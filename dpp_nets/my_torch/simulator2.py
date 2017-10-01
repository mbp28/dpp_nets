import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dpp_nets.my_torch.linalg import custom_decomp
from dpp_nets.my_torch.DPP import DPP
from dpp_nets.my_torch.utilities import compute_baseline
from dpp_nets.my_torch.utilities import pad_tensor

import dpp_nets.my_torch

import numpy as np 
from collections import defaultdict
import random


class SimulClassifier(object):

    def __init__(self, input_set_size, aspects_n, dtype):

        self.input_set_size = input_set_size
        self.aspects_n = aspects_n
        self.dtype = dtype
        
        self.kernel_in = 400
        self.kernel_h = 500
        self.kernel_out = 200

        self.pred_in = 200
        self.pred_h = 500
        self.pred_out = aspects_n


        self.kernel_net = torch.nn.Sequential(nn.Linear(self.kernel_in, self.kernel_h), nn.ReLU(), nn.Linear(self.kernel_h, self.kernel_h), 
                                                nn.ReLU(), nn.Linear(self.kernel_h, self.kernel_out))
        self.kernel_net.type(self.dtype)

        self.pred_net = torch.nn.Sequential(nn.Linear(self.pred_in, self.pred_h), nn.ReLU(), nn.Linear(self.pred_h, self.pred_h), nn.ReLU(), 
                                            nn.Linear(self.pred_h, self.pred_out), nn.Sigmoid())
        self.pred_net.type(self.dtype)

        # Deterministic Baseline
        self.baseline_model = nn.Sequential(nn.Linear(200, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU() ,nn.Linear(500, self.aspects_n), nn.Sigmoid())
        self.baseline_model.type(self.dtype)

        # Useful        
        self.saved_subsets = None
        self.saved_losses = None
        self.saved_baselines = None

        # Record loss
        self.loss_dict = defaultdict(list)
        self.prec_dict = defaultdict(list)
        self.rec_dict = defaultdict(list)
        self.ssize_dict = defaultdict(list)

        # Useful intermediate variables 
        self.embedding = None
        self.subset = None
        self.pick = None
        self.pred = None
 
        # Random Corners of Positive Hypercube for positive signal 
        self.signal_clusters = torch.rand(2 * aspects_n, self.pred_in)
        self.signal_cluster_var = 0.1

        # Counter for correct plotting
        self.counter = 0

    def generate(self):

        dtype = self.dtype

        words = torch.rand(self.input_set_size, self.pred_in)
        target = torch.FloatTensor(self.aspects_n).random_(2)

        # Compute signals
        signal = self.signal_clusters[target.long() + torch.LongTensor([2*i for i in range(self.aspects_n)])]
        ixs = torch.multinomial(torch.arange(0, self.input_set_size), self.aspects_n)
        words[ixs] = signal

        context = words.sum(0).expand_as(words)

        return Variable(words).type(dtype), Variable(context).type(dtype), ixs, Variable(target).type(dtype)


    def train(self, train_steps, batch_size=1, sample_iter=1, lr=1e-3, baseline=False, reg=0, reg_mean=0):

        if baseline:
            assert sample_iter > 1

        params = [{'params': self.kernel_net.parameters()}, {'params': self.pred_net.parameters()}]
        optimizer = optim.Adam(params, lr=lr)

        train_iter = train_steps * batch_size
        
        cum_loss = 0
        cum_prec = 0 
        cum_rec = 0
        cum_size = 0

        for t in range(train_iter):

            actions = self.saved_subsets = []
            rewards = self.saved_losses =  []
            picks = []


            words, context, ixs, target = self.generate()
            input_x = torch.cat([words, context], dim=1)
            kernel  = self.kernel_net(input_x)
            vals, vecs = custom_decomp()(kernel)

            pred_loss = 0

            for j in range(sample_iter):
                
                subset = DPP()(vals, vecs)
                actions.append(subset)
                pick = subset.diag().mm(words).sum(0, keepdim=True)
                self.pred = self.pred_net(pick).squeeze()
                loss = nn.BCELoss()(self.pred, target)
                rewards.append(loss.data[0])
                pred_loss += loss

                # For the statistics
                precision, recall, set_size = self.assess(subset.data, ixs)
                cum_loss += loss.data[0]
                cum_prec += precision
                cum_rec += recall 
                cum_size += set_size 

            # Compute baselines
            if baseline: 
                self.saved_baselines = compute_baseline(self.saved_losses)
            else:
                self.saved_baselines = self.saved_losses

            # Register rewards
            for action, reward in zip(self.saved_subsets, self.saved_baselines):
                action.reinforce(reward)

            # Apply Regularization
            total_loss = pred_loss

            if reg:
                card = (vals / (1 + vals)).sum()
                reg_loss = sample_iter * reg * ((card - reg_mean)**2)

                total_loss += reg_loss

            total_loss.backward()

            if not ((t + 1) % batch_size):
                optimizer.step()
                optimizer.zero_grad()
                
                if not ((t + 1) % (batch_size * 100)):
                    print(cum_loss / (batch_size * sample_iter))

                self.loss_dict[self.counter].append(cum_loss / (batch_size * sample_iter))
                self.prec_dict[self.counter].append(cum_prec / (batch_size * sample_iter))
                self.rec_dict[self.counter].append(cum_rec / (batch_size * sample_iter))
                self.ssize_dict[self.counter].append(cum_size / (batch_size * sample_iter))

                self.counter += 1

                cum_loss = 0
                cum_prec = 0 
                cum_rec = 0
                cum_size = 0

    def assess(self, subset, ixs):
        """
        This function compares a subset with ground-truth indices to assess whether junk
        or good stuff was selected. Don't pass variables, pass tensors
        """
        set_size = subset.sum()
        hits = subset[ixs].sum()
        misses = set_size - hits

        if set_size:
            precision = hits / set_size 
            recall = hits / self.aspects_n
            return precision, recall, set_size 
        else: 
            return 0, 0, 0


    def evaluate(self, test_iter):

        cum_loss = 0
        cum_acc = 0
        cum_prec = 0 
        cum_rec = 0
        cum_size = 0

        for t in range(test_iter):

            words, context, ixs, target = self.generate()
            input_x = torch.cat([words, context], dim=1)
            kernel  = self.kernel_net(input_x)
            vals, vecs = custom_decomp()(kernel)
            subset = DPP()(vals, vecs)
            pick = subset.diag().mm(words).sum(0, keepdim=True)
            self.pred = self.pred_net(pick).squeeze()
            loss = nn.BCELoss()(self.pred, target)

            # Classification Accuracy
            label = torch.round(self.pred.data)
            cum_acc += ((label == target.data).sum() / self.aspects_n)

            # Subset Statistics 
            precision, recall, set_size = self.assess(subset.data, ixs)
            cum_loss += loss.data[0]
            cum_prec += precision
            cum_rec += recall 
            cum_size += set_size 


        print('Loss:', cum_loss / test_iter, 'Pred Acc:', cum_acc / test_iter, 
            'Precision:', cum_prec / test_iter, 'Recall:', cum_rec / test_iter, 
            'Set Size:', cum_size / test_iter)


    def sample(self):

        words, context, ixs, target = self.generate()
        input_x = torch.cat([words, context], dim=1)
        kernel  = self.kernel_net(input_x)
        vals, vecs = custom_decomp()(kernel)
        subset = DPP()(vals, vecs)
        pick = subset.diag().mm(words).sum(0, keepdim=True)
        self.pred = self.pred_net(pick).squeeze()
        loss = nn.BCELoss()(self.pred, target)

        # Classification Accuracy
        label = torch.round(self.pred.data)
        acc = ((label == target.data).sum() / self.aspects_n)

        # Subset Statistics 
        precision, recall, set_size = self.assess(subset.data, ixs)

        # Print
        print('Target is: ', target.data)
        print('Pred is: ', self.pred.data)
        print('Loss is:', loss.data[0])
        print('Acc is:', ((label == target.data).sum() / self.aspects_n))
        print('Subset is:', subset.data)
        print('Ix is:', ixs)
        print('Subset statistics are:', precision, recall, set_size)

        return words, context, ixs, target, self.pred, loss, subset


    def train_deterministic(self, train_steps, batch_size=1, lr=1e-3):

        

        optimizer = optim.Adam(self.baseline_model.parameters(), lr=lr)

        train_iter = train_steps * batch_size
        
        cum_loss = 0

        for t in range(train_iter):

            words, context, ixs, target = self.generate()

            pred = self.baseline_model(words.sum(0))
            loss = nn.BCELoss()(pred, target.squeeze())
            cum_loss += loss.data[0]
            
            loss.backward()

            if not ((t + 1) % batch_size):
                optimizer.step()
                optimizer.zero_grad()
                


                if not ((t + 1) % (batch_size * 100)):
                    print(cum_loss / batch_size)

                self.loss_dict[self.counter].append(cum_loss / (batch_size))
                self.counter += 1
                cum_loss = 0


    def evaluate_deterministic(self, test_iter):


        cum_loss = 0
        cum_acc = 0

        for t in range(test_iter):

            words, context, ixs, target = self.generate()

            pred = self.baseline_model(words.sum(0))
            loss = nn.BCELoss()(pred, target.squeeze())
            cum_loss += loss.data[0]

            # Classification Accuracy
            label = torch.round(pred.data)
            cum_acc += ((label == target.data).sum() / self.aspects_n)

        print(cum_loss / test_iter, cum_acc / test_iter)

    def sample_deterministic(self):


        words, context, ixs, target = self.generate()
        pred = self.baseline_model(words.sum(0))
        loss = nn.BCELoss()(pred, target.squeeze())
        label = torch.round(pred.data)

        print('Target is: ', target.data)
        print('Pred is: ', pred.data)
        print('Loss is:', loss.data[0])
        print('Acc is:', ((label == target.data).sum() / self.aspects_n))



        return words, context, ixs, target, pred, loss


class SimulRegressor(object):

    def __init__(self, input_set_size, n_clusters, dtype):

        self.input_set_size = input_set_size
        self.n_clusters = n_clusters
        self.dtype = dtype
        
        self.kernel_in = 400
        self.kernel_h = 500
        self.kernel_out = 200

        self.pred_in = 200
        self.pred_h = 500
        self.pred_out = 1


        self.kernel_net = torch.nn.Sequential(nn.Linear(self.kernel_in, self.kernel_h), nn.ReLU(), nn.Linear(self.kernel_h, self.kernel_h), 
                                                nn.ReLU(), nn.Linear(self.kernel_h, self.kernel_out))
        self.kernel_net.type(self.dtype)

        self.pred_net = torch.nn.Sequential(nn.Linear(self.pred_in, self.pred_h), nn.ReLU(), nn.Linear(self.pred_h, self.pred_h), nn.ReLU(), 
                                            nn.Linear(self.pred_h, self.pred_out))
        self.pred_net.type(self.dtype)

        # Deterministic Baseline
        self.baseline_model = nn.Sequential(nn.Linear(200, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU() ,nn.Linear(500, self.pred_out))
        self.baseline_model.type(self.dtype)

        # Useful        
        self.saved_subsets = None
        self.saved_losses = None
        self.saved_baselines = None

        # Record loss
        self.loss_dict = defaultdict(list)
        self.prec_dict = defaultdict(list)
        self.rec_dict = defaultdict(list)
        self.ssize_dict = defaultdict(list)

        # Useful intermediate variables 
        self.embedding = None
        self.subset = None
        self.pick = None
        self.pred = None
 
        # Random Corners of Positive Hypercube for positive signal 
        self.signal_clusters = torch.rand(n_clusters, self.pred_in)
        self.signal_cluster_var = 0.1

        # Counter for correct plotting
        self.counter = 0


    def generate(self, n_cl_sample=None):


        n_clusters = self.n_clusters
        input_set_size = self.input_set_size
        dtype = self.dtype


        # Generate target (How many clusters?) # uniform between 1 and n_clusters
        if not n_cl_sample:
            target = random.randint(1,n_clusters)
        else:
            target = n_cl_sample

        # Which clusters shall be present? # Make sure each index appears at least once
        cluster_iden = torch.ones(n_clusters).multinomial(target, replacement=False)
        cluster_ixs = torch.cat([torch.arange(0,target).long(), torch.ones(target).multinomial(input_set_size - target, replacement=True)]) #
        cluster_ixs = cluster_ixs[torch.randperm(input_set_size)] # shuffle the ixs.

        
        words = torch.normal(self.signal_clusters[cluster_iden[cluster_ixs]], self.signal_cluster_var).type(dtype)
        context = words.sum(0).expand_as(words)

        target = torch.FloatTensor([target]).type(dtype)

        return Variable(words), Variable(context), Variable(target), cluster_iden, cluster_ixs

    def train_deterministic(self, train_steps, batch_size=1, lr=1e-3):

        optimizer = optim.Adam(self.baseline_model.parameters(), lr=lr)
        train_iter = train_steps * batch_size
        cum_loss = 0

        for t in range(train_iter):

            words, context, target, cluster_iden, cluster_ixs = self.generate()

            pred = self.baseline_model(words.sum(0))
            loss = nn.MSELoss()(pred, target.squeeze())
            cum_loss += loss.data[0]
            
            loss.backward()

            if not ((t + 1) % batch_size):
                optimizer.step()
                optimizer.zero_grad()
    
                if not ((t + 1) % (batch_size * 100)):
                    print(cum_loss / batch_size)

                self.loss_dict[self.counter].append(cum_loss / (batch_size))
                self.counter += 1
                cum_loss = 0

    def evaluate_deterministic(self, test_iter):


        cum_loss = 0

        for t in range(test_iter):

            words, context, target, cluster_iden, cluster_ixs = self.generate()

            pred = self.baseline_model(words.sum(0))
            loss = nn.MSELoss()(pred, target.squeeze())
            cum_loss += loss.data[0]

        print(cum_loss / test_iter)

    def sample_deterministic(self):


        words, context, target, cluster_iden, cluster_ixs = self.generate()
        pred = self.baseline_model(words.sum(0))
        loss = nn.MSELoss()(pred, target.squeeze())

        print('Target is: ', target.data)
        print('Pred is: ', pred.data)
        print('Loss is:', loss.data[0])


        return words, context, target, pred, loss

    def train(self, train_steps, batch_size=1, sample_iter=1, lr=1e-3, baseline=False, reg=0, reg_mean=0):

        if baseline:
            assert sample_iter > 1

        params = [{'params': self.kernel_net.parameters()}, {'params': self.pred_net.parameters()}]
        optimizer = optim.Adam(params, lr=lr)

        train_iter = train_steps * batch_size
        
        cum_loss = 0
        cum_prec = 0 
        cum_rec = 0
        cum_size = 0

        for t in range(train_iter):

            actions = self.saved_subsets = []
            rewards = self.saved_losses =  []
            picks = []

            words, context, target, cluster_iden, ixs = self.generate()
            input_x = torch.cat([words, context], dim=1)
            kernel  = self.kernel_net(input_x)
            vals, vecs = custom_decomp()(kernel)

            pred_loss = 0

            for j in range(sample_iter):
                
                subset = DPP()(vals, vecs)
                actions.append(subset)
                pick = subset.diag().mm(words).sum(0, keepdim=True)
                self.pred = self.pred_net(pick).squeeze()
                loss = nn.MSELoss()(self.pred, target)
                rewards.append(loss.data[0])
                pred_loss += loss

                # For the statistics
                precision, recall, set_size = self.assess(subset.data, ixs) 
                cum_loss += loss.data[0]
                cum_prec += precision
                cum_rec += recall 
                cum_size += set_size 

            # Compute baselines
            if baseline: 
                self.saved_baselines = compute_baseline(self.saved_losses)
            else:
                self.saved_baselines = self.saved_losses

            # Register rewards
            for action, reward in zip(self.saved_subsets, self.saved_baselines):
                action.reinforce(reward)

            # Apply Regularization
            total_loss = pred_loss

            if reg:
                card = (vals / (1 + vals)).sum()
                reg_loss = sample_iter * reg * ((card - reg_mean)**2)

                total_loss += reg_loss

            total_loss.backward()

            if not ((t + 1) % batch_size):
                optimizer.step()
                optimizer.zero_grad()
                
                if not ((t + 1) % (batch_size * 100)):
                    print(cum_loss / (batch_size * sample_iter))

                self.loss_dict[self.counter].append(cum_loss / (batch_size * sample_iter))
                self.prec_dict[self.counter].append(cum_prec / (batch_size * sample_iter))
                self.rec_dict[self.counter].append(cum_rec / (batch_size * sample_iter))
                self.ssize_dict[self.counter].append(cum_size / (batch_size * sample_iter))
                self.counter += 1

                cum_loss = 0
                cum_prec = 0 
                cum_rec = 0
                cum_size = 0

    def assess(self, subset, ixs):
        """
        This function compares a subset with ground-truth indices to assess whether junk
        or good stuff was selected. Don't pass variables, pass tensors
        """

        selection = torch.masked_select(ixs, subset.byte()).numpy()
        set_size = subset.sum()

        n_total = len(np.unique(ixs.numpy()))
        n_caught = len(np.unique(selection))
        n_missed = n_total - n_caught
        n_duplicate = set_size - n_caught

        if set_size:
            precision = n_caught / set_size 
            recall = n_caught / n_total
            return precision, recall, set_size 
        else: 
            return 0, 0, 0

    def evaluate(self, test_iter):

        cum_loss = 0
        cum_prec = 0 
        cum_rec = 0
        cum_size = 0

        for t in range(test_iter):

            words, context, target, cluster_iden, ixs  = self.generate()
            input_x = torch.cat([words, context], dim=1)
            kernel  = self.kernel_net(input_x)
            vals, vecs = custom_decomp()(kernel)
            subset = DPP()(vals, vecs)
            pick = subset.diag().mm(words).sum(0, keepdim=True)
            self.pred = self.pred_net(pick).squeeze()
            loss = nn.MSELoss()(self.pred, target)

            # Subset Statistics 
            precision, recall, set_size = self.assess(subset.data, ixs)
            cum_loss += loss.data[0]
            cum_prec += precision
            cum_rec += recall 
            cum_size += set_size 


        print(cum_loss / test_iter, cum_prec / test_iter, cum_rec / test_iter, cum_size / test_iter)
        return(cum_loss / test_iter, cum_prec / test_iter, cum_rec / test_iter, cum_size / test_iter)

    def sample(self):

        words, context, target, cluster_iden, ixs = self.generate()
        input_x = torch.cat([words, context], dim=1)
        kernel  = self.kernel_net(input_x)
        vals, vecs = custom_decomp()(kernel)
        subset = DPP()(vals, vecs)
        pick = subset.diag().mm(words).sum(0, keepdim=True)
        self.pred = self.pred_net(pick).squeeze()
        loss = nn.MSELoss()(self.pred, target)

        # Subset Statistics 
        precision, recall, set_size = self.assess(subset.data, ixs)

        # Print
        print('Target is: ', target.data)
        print('Pred is: ', self.pred.data)
        print('Loss is:', loss.data[0])
        print('Subset is:', subset.data)
        print('Ix is:', ixs)
        print('Subset statistics are:', precision, recall, set_size)

    def evaluate_fixed(self, test_iter, n_cl_sample):

        cum_loss = 0
        cum_prec = 0 
        cum_rec = 0
        cum_size = 0

        for t in range(test_iter):

            words, context, target, cluster_iden, ixs  = self.generate(n_cl_sample)
            input_x = torch.cat([words, context], dim=1)
            kernel  = self.kernel_net(input_x)
            vals, vecs = custom_decomp()(kernel)
            subset = DPP()(vals, vecs)
            pick = subset.diag().mm(words).sum(0, keepdim=True)
            self.pred = self.pred_net(pick).squeeze()
            loss = nn.MSELoss()(self.pred, target)

            # Subset Statistics 
            precision, recall, set_size = self.assess(subset.data, ixs)
            cum_loss += loss.data[0]
            cum_prec += precision
            cum_rec += recall 
            cum_size += set_size 


        print(cum_loss / test_iter, cum_prec / test_iter, cum_rec / test_iter, cum_size / test_iter)
        return(cum_loss / test_iter, cum_prec / test_iter, cum_rec / test_iter, cum_size / test_iter)




