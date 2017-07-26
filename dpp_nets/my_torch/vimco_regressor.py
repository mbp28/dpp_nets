import dpp_nets.my_torch
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dpp_nets.my_torch.controlvar import compute_alpha
from dpp_nets.my_torch.linalg import my_svd

from collections import defaultdict

class DPP(Function):
    
    def forward(self, vals, vecs):
        n = vecs.size(0)
        n_vals = vals.size(0)
        
        index = torch.rand(n_vals) < (vals / (vals + 1))
        k = torch.sum(index)
        print(k)
        if not k:
            subset = torch.zeros(n)
        
        if k == n:
            subset =  torch.ones(n)

        else:
            V = vecs[index.expand_as(vecs)].view(n, -1)
            subset = torch.zeros(n)

            for i in range(k):
                p = torch.sum(V**2, dim=1)
                p = torch.cumsum(p / torch.sum(p), 0) # item cumulative probabilities
                _, item = (torch.rand(1)[0] < p).max(0)
                subset[item[0,0]] = 1

                _, j = (torch.abs(V[item[0,0], :]) > 0).max(0)
                Vj = V[:, j[0]]
                if V.size(1) > 1:
                    V = delete_col(V, j[0])
                    V = V - torch.mm(Vj.unsqueeze(1), V[item[0,0], :].unsqueeze(1).t() / Vj[item[0,0]])
                    # find a new orthogonal basis
                    for a in range(V.size(1)):
                        for b in range(a):
                            V[:,a] = V[:,a] - V[:,a].dot(V[:,b]) * V[:,b]
                        V[:,a] = V[:,a] / torch.norm(V[:,a])
                    
        self.save_for_backward(vals, vecs, subset) 
        
        return subset
        
    def backward(self, grad_subset):
        
        vals, vecs, subset = self.saved_tensors
        matrix = vecs.mm(vals.diag()).mm(vecs.t())
        P = torch.eye(5).masked_select(subset.expand(5,5).t().byte()).view(subset.long().sum(),-1)
        submatrix = P.mm(matrix).mm(P.t())
        subinv = torch.inverse(submatrix)
        Pvecs = P.mm(vecs)
        
        
        grad_vals = 1 / vals
        grad_vals += Pvecs.t().mm(subinv).mm(Pvecs).diag()
        grad_vecs = P.t().mm(subinv).mm(Pvecs).mm(vals.diag())
        
        return grad_vals, grad_vecs

class custom_eig(Function):
    
    def forward(self, matrix):
        assert matrix.size(0) == matrix.size(1)
        e, v = torch.eig(matrix, eigenvectors=True)
        e = e[:,0]
        self.save_for_backward(e, v)
        return e, v

    def backward(self, grad_e, grad_v):
        e, v = self.saved_tensors
        dim = v.size(0)
        E = e.expand(dim, dim) - e.expand(dim, dim).t()
        I = E.new(dim, dim).copy_(torch.eye(dim))
        F = (1 / (E + I)) - I 
        M = grad_e.diag() + F * (v.t().mm(grad_v))
        grad_matrix = v.mm(M).mm(v.t())
        return grad_matrix

class DPPLayer(nn.Module):

    def __init__(self):
        super(DPPLayer, self).__init__()

    def forward(self, e, v):
        return DPP()(e, v)


class VIMCO_Regressor(nn.Module):
    
    def __init__(self, network_params, dtype):
        """
        Arguments:
        - network_params: see below for which parameters must specify
        - dtype: torch.DoubleTensor or torch.FloatTensor
        """
        super(VIMCO_Regressor, self).__init__()
        
        # Read in parameters
        self.set_size = network_params['set_size'] # 40
        self.emb_in = network_params['emb_in']
        self.emb_h = network_params['emb_h']
        self.emb_out = network_params['emb_out']
        self.pred_in = network_params['pred_in'] 
        self.pred_h = network_params['pred_h']
        self.pred_out = network_params['pred_out']
        assert int(self.emb_in / 2) == self.pred_in
        self.dtype = dtype
        self.ortho = True

        # Initialize Network
        self.emb_layer = torch.nn.Sequential(nn.Linear(self.emb_in, self.emb_h), nn.ELU(),
                                             nn.Linear(self.emb_h, self.emb_out))
        self.eig = custom_eig()
        self.dpp_sample = DPP()
        self.pred_layer = torch.nn.Sequential(nn.Linear(self.pred_in, self.pred_h), nn.ReLU(), 
                                                    nn.Linear(self.pred_h, self.pred_out))
        # Choose MSELoss as training criterion
        self.criterion = nn.MSELoss()

        # Useful intermediate variables 
        self.embedding = None
        self.subset = None
        self.pick = None
        self.pred = None

        self.type(self.dtype)

        # A varierty of convenience dictionaries
        # For alpha iteration
        self.alpha_dict = defaultdict(list)
        self.a_score_dict = defaultdict(list) # never delete
        self.a_reinforce_dict = defaultdict(list) # never delete
        self.a_loss_dict = defaultdict(list)

        # Gradients
        self.score_dict = defaultdict(list)
        self.reinforce_dict = defaultdict(list)
        self.control_dict = defaultdict(list)

        # Prediction & Loss
        self.embedding_dict = defaultdict(list)
        self.subset_dict = defaultdict(list)
        self.pred_dict = defaultdict(list)
        self.loss_dict = defaultdict(list)
        self.total_loss = defaultdict(list)

        # Network weights
        self.emb_w1_max = defaultdict(list)
        self.emb_w1_mean = defaultdict(list)
        self.emb_w2_max = defaultdict(list)
        self.emb_w2_mean = defaultdict(list)

        self.pred_w1_max = defaultdict(list)
        self.pred_w1_mean = defaultdict(list)
        self.pred_w2_max = defaultdict(list)
        self.pred_w2_mean = defaultdict(list)

        self.sample()


    def forward(self, words, context):
        """
        words: Tensor of dimension [set_size, word_emb_dim]
        contexts: Tensor of dimension [set_size, word_emb_dim]
        The rows of x2 are all identical and equal the sum
        across rows of x1 (context to predict DPP embedding)
        self.emb_dim must be 2 * word_emb_dim
        """
        # Concatenate individual words and set context
        x = torch.cat([words, context], dim = 1)

        # Compute embedding of DPP kernel
        self.embedding = self.emb_layer(x)

        # Sample a subset of words from the DPP
        
        self.subset = torch.diag(self.dpp_layer(self.embedding))
        
        # Filter out the selected words and combine them
        self.pick = self.subset.mm(words).sum(0)

        # Compute a prediction based on the sampled words
        self.pred = self.pred_layer(self.pick)

        return self.pred

    def generate(self, full=False):
        """
        Each training instance consists of a set of words (2D array) whose words come from a random
        number (between 1 and 20) of different clusters. In total, there exist self.pred_in / GLUE 
        different clusters. Each cluster contains standard random normal noise in most dimensions,
        except for GLUE dimensions, in which its entries are generated by a normal distribution around
        50. These signal dimensions differ across all clusters. 
        """

        if self.ortho:
            SCALE = 5
            OFFSET = 20
            STD = 0.5
            GLUE = 5 # how many dimensions carry the signal

            words = np.random.randn(self.set_size, self.pred_in)
            n_clusters = 1 + np.random.choice(20,1) 
            rep = (self.set_size // n_clusters) + 1 
            clusters = np.random.choice(self.pred_in, n_clusters, replace=False) # there are self.pred_in different clusters altogether
            clusters = np.tile(clusters, rep)[:(self.set_size)]
            cluster_means = SCALE * (clusters - OFFSET)
            cluster_means = np.tile(cluster_means,[GLUE, 1]).T
            words[:,:GLUE] = np.random.normal(cluster_means, STD)
            context = np.tile(np.sum(words, axis=0), (self.set_size, 1))

            # Shuffle
            shuffle_ix = np.random.permutation(self.set_size)
            copy_words = words.copy()
            copy_clusters = clusters.copy()
            words[shuffle_ix] = copy_words
            clusters[shuffle_ix] = copy_clusters
            clusters = self.dtype(clusters.astype(np.float64))

            # Wrap into Variables 
            words = Variable(self.dtype(words))
            context = Variable(self.dtype(context))
            target = Variable(self.dtype(n_clusters.astype(np.float64)))

            if not full: 
                return words, context, target
            else:
                return words, context, target, clusters


        else:
            GLUE = 1
            SIGNAL = 50

            words = np.random.randn(self.set_size, self.pred_in)

            # Sample a number of clusters (between 1 and 20) present in training instance
            n_clusters = 1 + np.random.choice(20,1) 
            # Will repeat cluster indices to fill upto set_size
            rep = (self.set_size // n_clusters) + 1 

            # Sample cluster indices 
            clusters = np.random.choice(self.pred_in // GLUE, n_clusters, replace=False)

            # Find column indices associated with cluster indices
            col_idx = np.array([np.arange(i*GLUE, i*GLUE + GLUE) for i in clusters]).flatten()
            # Repeat indices to fill upto set_size 
            col_idx = np.tile(col_idx, rep)[:(self.set_size * GLUE)]

            # Overwrite training data with signal according to column indices
            words[np.repeat(np.arange(self.set_size), GLUE), col_idx] = np.random.normal(SIGNAL, 1, (self.set_size * GLUE))

            # Create context 
            context = np.tile(np.sum(words, axis=0), (self.set_size, 1))

            # Shuffle, so it doesn't learn to always choose the first item for example.
            clusters = np.tile(clusters, rep)[:(self.set_size)]
            shuffle_ix = np.random.permutation(self.set_size)
            copy_words = words.copy()
            copy_clusters = clusters.copy()
            words[shuffle_ix] = copy_words
            clusters[shuffle_ix] = copy_clusters
            clusters = self.dtype(clusters.astype(np.float64))

            # Wrap into Variables 
            words = Variable(self.dtype(words))
            context = Variable(self.dtype(context))
            target = Variable(self.dtype(n_clusters.astype(np.float64)))

            if not full:
                return words, context, target
            else: 
                return words, context, target, clusters

    def sample(self):
        """
        Demonstrates how Network is currently performing
        by classifying a random instance 
        """

        # Sample
        words, context, target, clusters = self.generate(full=True)

        # Assigns prediction to self.pred
        self.forward(words, context)

        # How many true clusters existed?
        true_n = int(target.data[0])
        print("Number of different clusters was: ", true_n)

        # What did the network predict?
        pred = self.pred.data[0,0]
        print("Number of clusters predicted was: ", pred)

        # What was the loss?
        loss = self.criterion(self.pred, target).data[0]
        print("Resultant loss was: ", loss)

        # How many elements were retrieved?
        subset_size = int(self.subset.data.sum())
        print("Retrieved subset was of size: ", subset_size)

        # How many different clusters were detected?
        true_ix = clusters.numpy()
        retr_ix = torch.diag(self.subset.data).numpy()
        detected = true_ix[retr_ix.astype(bool)]
        n_detected = np.unique(detected).size
        print("Number of clusters detected by DPP was: ", n_detected)

        return (self.pred, target), (words, context), clusters

    def train_DPP_weak(self, train_iter, batch_size, lr):
        """
        NOTE TO MYSELF: IMPLEMENT THIS PROPERLY !! FOR THE FULL THING!! ? OR IS SIMPLE ENOUGH? 
        THIS IS WRONG: DOESN'T DO REINFORCE GRADIENT!!
        """
        self.pred_dict.clear()
        self.loss_dict.clear()

        self.weak_predictor = nn.Linear(self.pred_in, self.pred_out)

        optimizer = optim.SGD([{'params': self.emb_layer.parameters()},
                                   {'params': self.weak_predictor.parameters()}], 
                                  lr = lr / (batch_size))

        for t in range(train_iter):
            words, context, target = self.generate()
            x = torch.cat([words, context], dim = 1)
            self.embedding = self.emb_layer(x)
            self.subset = torch.diag(self.dpp_layer(self.embedding))
            self.pick = self.subset.mm(words).sum(0)
            self.pred = self.weak_predictor(self.pick)
            self.pred_dict[t].append(self.pred.data)
            loss = self.criterion(self.pred, target)
            self.loss_dict[t].append(loss.data) 
            loss.backward()

            # Update parameters
            if (t + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print loss
            if (t + 1) % 50 == 0:
                print(t + 1, loss.data[0])

    def train_DPP_strong(self, train_iter, batch_size, sample_iter, alpha_iter, lr, weight_decay, reg_exp, reg_var, 
        rajeesh=True, el_mean=False, overwrite=0):

        self.alpha_dict.clear()
        self.a_score_dict.clear()
        self.a_reinforce_dict.clear()
        self.a_loss_dict.clear()

        # Gradients
        self.score_dict.clear()
        self.reinforce_dict.clear()
        self.control_dict.clear()

        # Prediction & Loss
        self.embedding_dict.clear()
        self.subset_dict.clear()
        self.pred_dict.clear()
        self.loss_dict.clear()
        self.total_loss.clear()

        # Network weights
        self.emb_w1_max.clear()
        self.emb_w1_mean.clear()
        self.emb_w2_max.clear()
        self.emb_w2_mean.clear()

        self.pred_w1_max.clear()
        self.pred_w1_mean.clear()
        self.pred_w2_max.clear()
        self.pred_w2_mean.clear()

        def compute_miss(clusters, subset):
            # How many different clusters were detected?
            true_ix = clusters.numpy()
            retr_ix = torch.diag(subset.data).numpy()
            detected = true_ix[retr_ix.astype(bool)]
            n_detected = np.unique(detected).size
            target = np.unique(true_ix).size
            missed = float(target - n_detected)
            too_many = np.sum(retr_ix) - target
            maxi = max(missed, too_many)
            loss = Variable(self.dtype([maxi**2]))
            return loss

        # Prepare Optimizer
        optimizer = optim.SGD(self.emb_layer.parameters(), lr = lr / (batch_size * sample_iter))
            
        for t in range(train_iter):

            # Draw a Training Sample
            words, context, target, clusters = self.generate(full=True)
            
            # Save current weights
            self.emb_w1_max[t].append(self.dtype([torch.max(torch.abs(self.emb_layer[0].weight.data))]))
            self.emb_w1_mean[t].append(self.dtype([torch.mean(torch.abs(self.emb_layer[0].weight.data))]))
            self.emb_w2_max[t].append(self.dtype([torch.max(torch.abs(self.emb_layer[2].weight.data))]))
            self.emb_w2_mean[t].append(self.dtype([torch.mean(torch.abs(self.emb_layer[2].weight.data))]))

            self.pred_w1_max[t].append(self.dtype([torch.max(torch.abs(self.pred_layer[0].weight.data))]))
            self.pred_w1_mean[t].append(self.dtype([torch.mean(torch.abs(self.pred_layer[0].weight.data))]))
            self.pred_w2_max[t].append(self.dtype([torch.max(torch.abs(self.pred_layer[2].weight.data))]))
            self.pred_w2_mean[t].append(self.dtype([torch.mean(torch.abs(self.pred_layer[2].weight.data))]))

            # Estimate alpha
            # Save score, build reinforce gradient, save reinforce gradient
            save_score = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.a_score_dict[t].append(grad_in[0].data))
            reinforce_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (grad_in[0] * (loss.data[0]),))
            save_reinforce = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.a_reinforce_dict[t].append(grad_in[0].data))
            
            if alpha_iter:
                if rajeesh: 
                    raise NotImplementedError
                        
                else:
                    for i in range(alpha_iter):    
                        x = torch.cat([words, context], dim = 1)
                        self.embedding = self.emb_layer(x)
                        self.subset = torch.diag(self.dpp_layer(self.embedding))
                        pseudo_loss = torch.sum(self.subset)
                        loss = compute_miss(clusters, self.subset)
                        self.a_loss_dict[t].append(loss.data)

                    self.alpha = self.dtype([torch.mean(torch.stack(self.a_loss_dict[t]))]).expand_as(self.embedding)
                    self.alpha_dict[t].append(self.alpha)
            else:
                self.alpha = overwrite * torch.ones(self.embedding.size()).type(self.dtype)
                    
            save_score.remove()
            reinforce_grad.remove()
            save_reinforce.remove()
            
            # now actual training
            # save scores, reinforce, implement baseline gradient, save baseline gradient
            save_score = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.score_dict[t].append(grad_in[0].data))
            save_reinforce = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.reinforce_dict[t].append(grad_in[0].data *  loss.data[0]))
            modify_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (Variable(grad_in[0].data * (loss.data[0] - self.alpha)),))
            save_control = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.control_dict[t].append(grad_in[0].data))

            # sample multiple times from the DPP and backpropagate associated gradients!
            for i in range(sample_iter):
                x = torch.cat([words, context], dim = 1)
                self.embedding = self.emb_layer(x)
                self.subset = torch.diag(self.dpp_layer(self.embedding))
                pseudo_loss = torch.sum(self.subset)
                loss = compute_miss(clusters, self.subset)

                self.embedding_dict[t].append(self.embedding.data)
                self.subset_dict[t].append(self.subset.data)
                self.loss_dict[t].append(loss.data) # save_loss

                if not reg_exp and not reg_var:
                    pseudo_loss.backward()

                else: 
                   raise NotImplementedError

            # update parameters after processing a batch
            if (t + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print loss
            if (t + 1) % 50 == 0:
                print(t + 1, loss.data[0])
                
            save_score.remove()
            save_reinforce.remove()
            modify_grad.remove()
            save_control.remove()

    def train_predictor(self, train_iter, batch_size, lr, ground=False):
        """
        Instead of jointly training the network, we only train the predictor network based on the entire ground
        set. 

        """
        self.pred_dict.clear()
        self.loss_dict.clear()
        
        optimizer = optim.SGD(self.pred_layer.parameters(), lr=lr / batch_size)

        for t in range(train_iter):
            words, context, target = self.generate()

            if ground: # train based on all words
                self.pick = words.sum(0)
                self.pred = self.pred_layer(self.pick)
            else:  # train based on current DPP selection
                self.forward(words, context)

            self.pred_dict[t].append(self.pred.data)
            loss = self.criterion(self.pred, target)
            self.loss_dict[t].append(loss.data) 
            loss.backward()

            # Update parameters
            if (t + 1) % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # print loss
            if (t + 1) % 50 == 0:
                print(t + 1, loss.data[0])

    def train_with_baseline(self, train_iter, batch_size, sample_iter, alpha_iter, lr, weight_decay, reg_exp, reg_var, 
        rajeesh=True, el_mean=False, overwrite=0, all=True):
        """
        Standard training algorithm. 
        Arguments:
        - rajeesh: We try to estimate the optimal alpha by cov / var
        - el_mean: We average these alpha estimates across all dimensions for each element. 
        - if rajeesh is false, but alpha_iter is True, we will take the average loss as our alpha --> very simple baseline. 
        - if alpha_iter is false (0), then we take classic REINFORCE grad, respectively take a constant alpha = overwrite. 
        - all: indicates that we train the model jointly, with all = False, we only train the DPP embedding. This is useful
        if we have already trained a good predictor network and now don't want to make it bad again, by initially high losses. 
        In this way, the predictor network can't compensate for the DPP network, but we are sending a high quality error signal
        to the DPP network. 
        """
        
        # Clear dictionaries
        # For alpha iteration
        self.alpha_dict.clear()
        self.a_score_dict.clear()
        self.a_reinforce_dict.clear()
        self.a_loss_dict.clear()

        # Gradients
        self.score_dict.clear()
        self.reinforce_dict.clear()
        self.control_dict.clear()

        # Prediction & Loss
        self.embedding_dict.clear()
        self.subset_dict.clear()
        self.pred_dict.clear()
        self.loss_dict.clear()
        self.total_loss.clear()

        # Network weights
        self.emb_w1_max.clear()
        self.emb_w1_mean.clear()
        self.emb_w2_max.clear()
        self.emb_w2_mean.clear()

        self.pred_w1_max.clear()
        self.pred_w1_mean.clear()
        self.pred_w2_max.clear()
        self.pred_w2_mean.clear()

        # Prepare Optimizer
        if all:
            optimizer = optim.SGD([{'params': self.emb_layer.parameters()},
                                   {'params': self.pred_layer.parameters(),
                                   'weight_decay': weight_decay * batch_size * sample_iter}], 
                                  lr = lr / (batch_size * sample_iter))
        else: 
            optimizer = optim.SGD(self.emb_layer.parameters(), lr = lr / (batch_size * sample_iter))
            
        for t in range(train_iter):

            # Draw a Training Sample
            words, context, target = self.generate()
            
            # Save current weights
            self.emb_w1_max[t].append(self.dtype([torch.max(torch.abs(self.emb_layer[0].weight.data))]))
            self.emb_w1_mean[t].append(self.dtype([torch.mean(torch.abs(self.emb_layer[0].weight.data))]))
            self.emb_w2_max[t].append(self.dtype([torch.max(torch.abs(self.emb_layer[2].weight.data))]))
            self.emb_w2_mean[t].append(self.dtype([torch.mean(torch.abs(self.emb_layer[2].weight.data))]))

            self.pred_w1_max[t].append(self.dtype([torch.max(torch.abs(self.pred_layer[0].weight.data))]))
            self.pred_w1_mean[t].append(self.dtype([torch.mean(torch.abs(self.pred_layer[0].weight.data))]))
            self.pred_w2_max[t].append(self.dtype([torch.max(torch.abs(self.pred_layer[2].weight.data))]))
            self.pred_w2_mean[t].append(self.dtype([torch.mean(torch.abs(self.pred_layer[2].weight.data))]))

            # Estimate alpha
            # Save score, build reinforce gradient, save reinforce gradient
            save_score = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.a_score_dict[t].append(grad_in[0].data))
            reinforce_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (grad_in[0] * (loss.data[0]),))
            save_reinforce = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.a_reinforce_dict[t].append(grad_in[0].data))
            
            if alpha_iter:
                if rajeesh: 
                    for i in range(alpha_iter):                                      
                        self.forward(words, context)
                        loss = self.criterion(self.pred, target)
                        self.a_loss_dict[t].append(loss.data)

                        loss.backward()

                    self.alpha = compute_alpha(self.a_reinforce_dict[t], self.a_score_dict[t], el_mean).type(self.dtype)
                    self.zero_grad()
                    self.alpha_dict[t].append(self.alpha)
                else:
                    for i in range(alpha_iter):                                      
                        self.forward(words, context)
                        loss = self.criterion(self.pred, target)
                        self.a_loss_dict[t].append(loss.data)

                    self.alpha = self.dtype([torch.mean(torch.stack(self.a_loss_dict[t]))]).expand_as(self.embedding)
                    self.alpha_dict[t].append(self.alpha)
            else:
                self.alpha = overwrite * torch.ones(self.embedding.size()).type(self.dtype)
                    
            save_score.remove()
            reinforce_grad.remove()
            save_reinforce.remove()
            
            # now actual training
            # save scores, reinforce, implement baseline gradient, save baseline gradient
            save_score = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.score_dict[t].append(grad_in[0].data))
            save_reinforce = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.reinforce_dict[t].append(grad_in[0].data *  loss.data[0]))
            modify_grad = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: (Variable(grad_in[0].data * (loss.data[0] - self.alpha)),))
            save_control = self.dpp_layer.register_backward_hook(lambda module, grad_in, grad_out: self.control_dict[t].append(grad_in[0].data))

            # sample multiple times from the DPP and backpropagate associated gradients!
            for i in range(sample_iter):
                self.forward(words, context)
                loss = self.criterion(self.pred, target)

                self.embedding_dict[t].append(self.embedding.data)
                self.subset_dict[t].append(self.subset.data)
                self.pred_dict[t].append(self.pred.data)
                self.loss_dict[t].append(loss.data) # save_loss

                if not reg_exp and not reg_var:
                    loss.backward()

                else: 
                    TRUE_MEAN = 10.5
                    _, s, _ = my_svd()(self.embedding)
                    exp = torch.sum(s**2 / (s**2 + 1))
                    var = exp - torch.sum(s**4 / ((s**2 + 1) **2))
                    reg_loss = reg_exp * ((exp - TRUE_MEAN)**2) + reg_var * var
                    total_loss = loss + reg_loss
                    self.total_loss[t].append(total_loss)
                    total_loss.backward()

            # update parameters after processing a batch
            if (t + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print loss
            if (t + 1) % 50 == 0:
                print(t + 1, loss.data[0])
                
            save_score.remove()
            save_reinforce.remove()
            modify_grad.remove()
            save_control.remove()

    def evaluate(self, test_iter):

        loss_sum = 0.0
        true_n_sum = 0.0
        n_detected_sum = 0.0

        subset_mean = 0.0
        temp = 0.0

        for t in range(test_iter):
            # Sample
            words, context, target, clusters = self.generate(full=True)
        
            # Assigns prediction to self.pred
            self.forward(words, context)

            # How many true clusters existed?
            true_n = target.data[0]
            true_n_sum += true_n

            # What was the loss?
            loss = self.criterion(self.pred, target).data[0]
            loss_sum += loss

            # How many different clusters were detected?
            true_ix = clusters.numpy()
            retr_ix = torch.diag(self.subset.data).numpy()
            detected = true_ix[retr_ix.astype(bool)]
            n_detected = np.unique(detected).size
            n_detected_sum += n_detected

            # Gather subset statistics
            delta = self.subset.data.sum() - subset_mean
            subset_mean += delta / (t+1)
            delta2 = delta / (t+1)
            temp += delta * delta2
        
        subset_var = temp / test_iter
        loss_av = loss_sum / test_iter

        print("Average Loss is: ", loss_av) 
        print("Average Subset Size: ", subset_mean)
        print("Subset Variance: ", subset_var)
        print("Proportion of true clusters retrieved:", n_detected_sum/ true_n_sum)


    def evaluate_random_selector(self, test_iter, subset_size):

        loss_sum = 0.0
        true_n_sum = 0.0
        n_detected_sum = 0.0

        subset_mean = 0.0
        temp = 0.0

        p_elem = float(subset_size) / self.set_size

        for t in range(test_iter):
            # Sample
            words, context, target, clusters = self.generate(full=True)
        
            # Assigns prediction to self.pred
            self.subset = Variable(self.dtype(torch.diag(torch.rand(self.set_size) < p_elem)))
        
            # Filter out the selected words and combine them
            self.pick = self.subset.mm(words).sum(0)

            # Compute a prediction based on the sampled words
            self.pred = self.pred_layer(self.pick)

            # How many true clusters existed?
            true_n = target.data[0]
            true_n_sum += true_n

            # What was the loss?
            loss = self.criterion(self.pred, target).data[0]
            loss_sum += loss

            # How many different clusters were detected?
            true_ix = clusters.numpy()
            retr_ix = torch.diag(self.subset.data).numpy()
            detected = true_ix[retr_ix.astype(bool)]
            n_detected = np.unique(detected).size
            n_detected_sum += n_detected

            # Gather subset statistics
            delta = self.subset.data.sum() - subset_mean
            subset_mean += delta / (t+1)
            delta2 = delta / (t+1)
            temp += delta * delta2
        
        subset_var = temp / test_iter
        loss_av = loss_sum / test_iter

        print("Average Loss is: ", loss_av) 
        print("Average Subset Size: ", subset_mean)
        print("Subset Variance: ", subset_var)
        print("Proportion of true clusters retrieved:", n_detected_sum/ true_n_sum)


        
    def reset_parameter(self):
        self.emb_layer[0].reset_parameters()
        self.emb_layer[2].reset_parameters()
        self.pred_layer[0].reset_parameters()
        self.pred_layer[2].reset_parameters()
        
        self.optimizer = optim.SGD([{'params': self.emb_layer.parameters(), 'weight_decay': self.reg_kern},
                                    {'params': self.pred_layer.parameters()}],
                                    lr = self.lr / self.batch_size)
