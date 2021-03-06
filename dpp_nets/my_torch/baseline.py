import dpp_nets.my_torch
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dpp_nets.my_torch.controlvar import compute_alpha
from dpp_nets.my_torch.linalg import my_svd

from collections import defaultdict

class RegressorBaseline(nn.Module):
    
    def __init__(self, network_params, dtype):
        """
        Arguments:
        - network_params: see below for which parameters must specify
        - dtype: torch.DoubleTensor or torch.FloatTensor
        """
        super(RegressorBaseline, self).__init__()
        
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

        # Initialize Network
        self.pred_layer = torch.nn.Sequential(nn.Linear(self.pred_in, self.pred_h), nn.ReLU(), 
                                                    nn.Linear(self.pred_h, self.pred_out))
        self.weak_predictor = None
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

        # Compute embedding of DPP kernel

        # Sample a subset of words from the DPP
        
        # Filter out the selected words and combine them
        self.pick = words.sum(0)

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

        return (self.pred, target), (words, context), clusters

   

    def train(self, train_iter, batch_size, lr, weight_decay):
        """
        Instead of jointly training the network, we only train the predictor network based on the entire ground
        set. 

        """
        self.pred_dict.clear()
        self.loss_dict.clear()
        
        optimizer = optim.SGD(self.parameters(), lr=lr / batch_size, weight_decay=weight_decay)

        for t in range(train_iter):
            words, context, target = self.generate()
            self.pick = words.sum(0)
            self.pred = self.pred_layer(self.pick)
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

        loss_av = loss_sum / test_iter

        print("Average Loss is: ", loss_av) 

    def reset_parameter(self):
        self.emb_layer[0].reset_parameters()
        self.emb_layer[2].reset_parameters()
        self.pred_layer[0].reset_parameters()
        self.pred_layer[2].reset_parameters()
        
        self.optimizer = optim.SGD([{'params': self.emb_layer.parameters(), 'weight_decay': self.reg_kern},
                                    {'params': self.pred_layer.parameters()}],
                                    lr = self.lr / self.batch_size)
