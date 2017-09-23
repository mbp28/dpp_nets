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



class SimKDPP(object):

        # Customizable parameters
        self.set_size = set_size = network_params['set_size'] # 40
        self.n_clusters = n_clusters = network_params['n_clusters'] # 
        self.dtype = dtype

        # Fixed parameters
        self.kernel_in = kernel_in = 100
        self.kernel_h = kernel_h = 500
        self.kernel_out = kernel_out = 100

        self.pred_in = pred_in = 50 # kernel_in / 2
        self.pred_h = pred_h = 500
        self.pred_out = pred_out = 100
        
        # 2-Hidden-Layer Networks 
        self.kernel_net = torch.nn.Sequential(nn.Linear(kernel_in, kernel_h), nn.ELU(),
                                              nn.Linear(kernel_h, kernel_h), nn.ELU(), nn.Linear(kernel_h, kernel_out))
        self.kernel_net.type(self.dtype)

        # Data
        np.random.seed(0)
        self.means = dtype(np.random.randint(-50,50,[n_clusters, int(pred_in)]).astype("float"))
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


    def generate(self, batch_size):
        """sdf"
        Arguments:
        means: Probs best to make this an attribute of the class, 
        so that repeated training works with the same data distribution.


        """
        batch_size = batch_size
        n_clusters = self.n_clusters
        set_size = self.set_size
        embd_dim = self.pred_in
        dtype = self.dtype
        means = self.means

        # Generate index
        index = torch.cat([torch.arange(0, float(n_clusters)).expand(batch_size, n_clusters).long(), 
                          torch.multinomial(torch.ones(batch_size, n_clusters), set_size - n_clusters, replacement=True)]
                         ,dim=1)
        index = index.t()[torch.randperm(set_size)].t().contiguous()

        # Generate words, context, target
        words = dtype(torch.normal(means.index_select(0,index.view(index.numel()))).view(batch_size, set_size, embd_dim))
        context = dtype(words.sum(1, keepdim=True).expand_as(words))
        target = index

        return words, context, target 

    def train(self, train_iter, batch_size, lr, alpha_iter=1, baseline=True, reg=0):
        """
        Training the model. 
        Doesn't use the forward pass as want to sample repeatedly!

        """
        set_size = self.set_size
        n_clusters = self.n_clusters
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out
        dtype = self.dtype

        loss_log = 100
        optimizer = optim.Adam(self.kernel_net.parameters(), lr=lr)

        self.loss_dict.clear()
        self.prec_dict.clear()
        self.rec_dict.clear()
        self.ssize_dict.clear()

        for t in range(train_iter):
            actions = self.saved_subsets = [[] for i in range(batch_size)]
            rewards = self.saved_losses =  [[] for i in range(batch_size)]

            cum_loss = 0.
            cum_prec = 0.
            cum_rec = 0.
            cum_ssize = 0.
            words, context, target = self.generate(batch_size)

            # Concatenate individual words and set context
            # Dimensions are batch_size x set_size x kernel_in
            batch_x = Variable(torch.cat([words, context], dim = 2))

            # Compute embedding of DPP kernel
            batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
            batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

            if reg:
                reg_loss = 0

            for i, kernel in enumerate(batch_kernel):
                
                for j in range(alpha_iter):
                    subset = DPP()(vals, vecs)
                    actions[i].append(subset)
                    loss, prec, rec, ssize  = self._big_assess(target[i], subset.data)
                    rewards[i].append(loss)
                    cum_loss += loss
                    cum_prec += prec
                    cum_rec += rec
                    cum_ssize += ssize

                if reg:
                    exp_ssize = (vals / (1 + vals)).sum()
                    reg_loss += reg * (exp_ssize - n_clusters)**2

            if baseline:
                self.saved_baselines = [compute_baseline(i) for i in self.saved_losses]
            else: 
                self.saved_baselines = self.saved_losses

            # Register the baselines
            for actions, rewards in zip(self.saved_subsets, self.saved_baselines):
                for action, reward in zip(actions, rewards):
                    action.reinforce(reward)

            pseudo_loss = torch.stack([torch.stack(subsets) for subsets in self.saved_subsets]).sum()
            if reg:
                (pseudo_loss + reg_loss).backward()
            else:
                pseudo_loss.backward(None)
            optimizer.step()
            optimizer.zero_grad()

            self.loss_dict[t].append(cum_loss / (batch_size * alpha_iter))
            self.prec_dict[t].append(cum_prec / (batch_size * alpha_iter))
            self.rec_dict[t].append(cum_rec / (batch_size * alpha_iter))
            self.ssize_dict[t].append(cum_ssize / (batch_size * alpha_iter))

            if not ((t + 1) % loss_log):
                print("Loss at it ", t+1, " is: ", cum_loss / (batch_size * alpha_iter)) 

    def _big_assess(self, target, subset):

        set_size = self.set_size
        n_clusters = self.n_clusters

        target = target.expand(set_size, set_size)
        target_mat = (target == target.t()).type(self.dtype)
        target_sums = target_mat.sum(1, keepdim=True)

        subset_mat = subset.expand_as(target_mat).type(self.dtype)

        loss = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).abs()
        loss.div_(target_sums)
        loss = loss.sum()**2

        eval_vec = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).sign() / target_sums
        n_missed = (eval_vec * (eval_vec.sign() < 0).type(self.dtype)).abs().sum()
        n_many = (eval_vec * (eval_vec.sign() > 0).type(self.dtype)).sum()
        n_one = n_clusters - n_missed - n_many 

        if not n_missed and not n_many:
            n_perfect = 1.
        else: 
            n_perfect = 0.

        subset_size = subset.sum()
        prec = (n_one + n_many) / subset_size 
        rec = 1 - (n_missed / 10)

        return loss, prec, rec, subset_size
       
    def _assess(self, target, subset):

        set_size = target.size(0)
        target = target.expand(set_size, set_size)
        target_mat = (target == target.t()).type(self.dtype)
        target_sums = target_mat.sum(1, keepdim=True)
    
        subset_mat = subset.expand_as(target_mat).type(self.dtype)
    
        loss = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).abs()
        loss.div_(target_sums)
        loss = loss.sum()**2
        
        return loss

    def _eval_assess(self, target, subset):

        set_size = self.set_size
        n_clusters = self.n_clusters
        target = target.expand(set_size, set_size)
        target_mat = (target == target.t()).type(self.dtype)
        target_sums = target_mat.sum(1, keepdim=True)
    
        subset_mat = subset.expand_as(target_mat).type(self.dtype)
    
        eval_vec = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).sign() / target_sums
        n_missed = (eval_vec * (eval_vec.sign() < 0).type(self.dtype)).abs().sum()
        n_many = (eval_vec * (eval_vec.sign() > 0).type(self.dtype)).sum()
        n_one = n_clusters - n_missed - n_many 

        if not n_missed and not n_many:
            n_perfect = 1.
        else: 
            n_perfect = 0.

        return n_missed, n_one, n_many, n_perfect

    def run(self, words, context, batch_size, alpha_iter):
        """
        This may be used by sample and by evaluate. 
        Samples once from DPP. 
        Can be used with any batch_size. 
        Returns a tensor of many subsets
        """
        set_size = self.set_size
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out

        actions = self.saved_subsets = [[] for i in range(batch_size)]
        rewards = self.saved_losses =  [[] for i in range(batch_size)]
        cum_loss = 0.

        # Concatenate individual words and set context
        # Dimensions are batch_size x set_size x kernel_in
        batch_x = Variable(torch.cat([words, context], dim = 2))

        # Compute embedding of DPP kernel
        batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
        batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

        for i, kernel in enumerate(batch_kernel):
            vals, vecs = custom_decomp()(kernel)
            for j in range(alpha_iter):
                subset = DPP()(vals, vecs)
                actions[i].append(subset)

        subset_tensor = torch.stack([torch.stack(subsets) for subsets in actions])

        return subset_tensor

    def evaluate(self, test_iter):

        n_missed  = 0.0
        n_one     = 0.0
        n_many    = 0.0
        n_perfect = 0.0

        loss = 0.0

        mean = 0.0 
        temp = 0.0
        var  = 0.0

        # We could avoid the loop and instead process one big batch?
        for t in range(test_iter):
            words, context, target  = self.generate(1)
            subset_tensor = self.run(words, context, 1, 1)
            missed, one, many, perfect = self._eval_assess(target, subset_tensor.squeeze().data)
            loss += self._assess(target[0], subset_tensor.squeeze().data)
            n_missed += missed
            n_one += one
            n_many += many
            n_perfect += perfect 

            # Gather subset statistics
            size = subset_tensor.data.sum()
            delta =  size - mean
            mean += delta / (t+1)
            delta2 = size - mean
            temp += delta * delta2
        
        var = temp / test_iter

        print("Average Subset Size: ", mean)
        print("Subset Variance: ", var)
        print("Average Loss", loss / test_iter)
        print("n_missed share", n_missed / (test_iter * self.n_clusters))
        print("n_one share", n_one / (test_iter * self.n_clusters))
        print("n_many share", n_many / (test_iter * self.n_clusters))
        print("n_perfect share", n_perfect/ test_iter)

    def random_benchmark(self, test_iter):

        set_size = self.set_size
        dtype = self.dtype
        p = self.n_clusters / self.set_size

        n_missed  = 0.0
        n_one     = 0.0
        n_many    = 0.0
        n_perfect = 0.0

        loss = 0.0

        mean = 0.0 
        temp = 0.0
        var  = 0.0

        # We could avoid the loop and instead process one big batch?
        for t in range(test_iter):
            _, _, target  = self.generate(1)
            random_subset = torch.bernoulli(p * torch.ones(set_size)).type(dtype)
            missed, one, many, perfect = self._eval_assess(target, random_subset)
            
            loss += self._assess(target[0], random_subset)
            n_missed += missed
            n_one += one
            n_many += many
            n_perfect += perfect 

            # Gather subset statistics
            size = random_subset.sum()
            delta =  size - mean
            mean += delta / (t+1)
            delta2 = size - mean
            temp += delta * delta2
        
        var = temp / test_iter

        print("Average Subset Size: ", mean)
        print("Subset Variance: ", var)
        print("Average Loss", loss / test_iter)
        print("n_missed share", n_missed / (test_iter * self.n_clusters))
        print("n_one share", n_one / (test_iter * self.n_clusters))
        print("n_many share", n_many / (test_iter * self.n_clusters))
        print("n_perfect share", n_perfect/ test_iter)

class SimKDPPDeepSet(object):

    def __init__(self, network_params, dtype):

        # Customizable parameters
        self.set_size = set_size = network_params['set_size'] # 40
        self.n_clusters = n_clusters = network_params['n_clusters'] # 
        self.dtype = dtype

        # Fixed parameters
        self.kernel_in = kernel_in = 100
        self.kernel_h = kernel_h = 500
        self.kernel_out = kernel_out = 100

        self.pred_in = pred_in = 50 # kernel_in / 2
        self.pred_h = pred_h = 500
        self.pred_out = pred_out = 100
        
        # 2-Hidden-Layer Networks 
        self.kernel_net = torch.nn.Sequential(nn.Linear(kernel_in, kernel_h), nn.ELU(),
                                              nn.Linear(kernel_h, kernel_h), nn.ELU(), nn.Linear(kernel_h, kernel_out))
        self.kernel_net.type(self.dtype)

        self.pred_net = torch.nn.Sequential(nn.Linear(pred_in, pred_h), nn.ELU(),
                                            nn.Linear(pred_h, pred_h), nn.ELU(), nn.Linear(pred_h, pred_in))

        self.pred_net.type(self.dtype)

        # Data
        np.random.seed(0)
        self.means = dtype(np.random.randint(-50,50,[n_clusters, int(pred_in)]).astype("float"))
        self.saved_subsets = None
        self.saved_losses = None
        self.saved_baselines = None

        # 
        self.criterion = nn.MSELoss()

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


    def generate(self, batch_size):
        """sdf"
        Arguments:
        means: Probs best to make this an attribute of the class, 
        so that repeated training works with the same data distribution.


        """
        batch_size = batch_size
        n_clusters = self.n_clusters
        set_size = self.set_size
        embd_dim = self.pred_in
        dtype = self.dtype
        means = self.means

        # Generate index
        index = torch.cat([torch.arange(0, float(n_clusters)).expand(batch_size, n_clusters).long(), 
                          torch.multinomial(torch.ones(batch_size, n_clusters), set_size - n_clusters, replacement=True)]
                         ,dim=1)
        index = index.t()[torch.randperm(set_size)].t().contiguous()

        # Generate words, context, target
        words = dtype(torch.normal(means.index_select(0,index.view(index.numel()))).view(batch_size, set_size, embd_dim))
        context = dtype(words.sum(1, keepdim=True).expand_as(words))

        target = torch.pow(torch.log1p(words.abs()).mean(1),2).squeeze()

        return words, context, target, index 

    def train(self, train_iter, batch_size, lr, alpha_iter=1, baseline=True, reg=0, reg_mean=0):
        """
        Training the model. 
        Doesn't use the forward pass as want to sample repeatedly!

        """
        set_size = self.set_size
        n_clusters = self.n_clusters
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out
        embd_dim = self.pred_in
        dtype = self.dtype
        criterion = self.criterion

        loss_log = 100
        optimizer = optim.Adam(self.kernel_net.parameters(), lr=lr)

        self.loss_dict.clear()
        self.prec_dict.clear()
        self.rec_dict.clear()
        self.ssize_dict.clear()

        for t in range(train_iter):
k
            cum_loss = 0.
            cum_prec = 0.
            cum_rec = 0.
            cum_ssize = 0.
            words, context, target, index = self.generate(batch_size)

            # Concatenate individual words and set context
            # Dimensions are batch_size x set_size x kernel_in
            batch_x = Variable(torch.cat([words, context], dim = 2))

            # Compute embedding of DPP kernel
            batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
            batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

            if reg:
                reg_loss = 0

            for i, kernel in enumerate(batch_kernel):
                vals, vecs = custom_decomp()(kernel)
                for j in range(alpha_iter):
                    subset = DPP()(vals, vecs)
                    pick = subset.diag().mm(.a(words[i])).sum(0, keepdim=True) 
                    actions[i].append(subset)
                    picks[i].append(pick)
                    _, prec, rec, ssize  = self._big_assess(index[i], subset.data)
                    cum_prec += prec
                    cum_rec += rec
                    cum_ssize += ssize

                if reg:
                    exp_ssize = (vals / (1 + vals)).sum()
                    reg_loss += reg * (exp_ssize - reg_mean)**2

            picks = torch.stack([torch.stack(pick) for pick in picks]).view(-1, embd_dim)
            preds = self.pred_net(picks).view(batch_size, alpha_iter, -1)

            targets = target.unsqueeze(1).expand_as(preds)
            loss = criterion(preds, Variable(targets))

            # Compute indivdual losses and baseline
            losses = ((preds - Variable(targets))**2).mean(2)
            self.saved_losses = [[i.data[0] for i in row] for row in losses]

            if baseline:
                self.saved_baselines = [compute_baseline(i) for i in self.saved_losses]
            else: 
                self.saved_baselines = self.saved_losses

            # Register the baselines
            for actions, rewards in zip(self.saved_subsets, self.saved_baselines):
                for action, reward in zip(actions, rewards):
                    action.reinforce(reward)

            if reg:
                (loss + reg_loss).backward()
            else:
                loss.backward(None)

            optimizer.step()
            optimizer.zero_grad()

            self.loss_dict[t].append(loss.data[0])
            self.prec_dict[t].append(cum_prec / (batch_size * alpha_iter))
            self.rec_dict[t].append(cum_rec / (batch_size * alpha_iter))
            self.ssize_dict[t].append(cum_ssize / (batch_size * alpha_iter))

            if not ((t + 1) % loss_log):
                print("Loss at it ", t+1, " is: ", loss.data[0])

    def _big_assess(self, target, subset):

        set_size = self.set_size
        n_clusters = self.n_clusters

        target = target.expand(set_size, set_size)
        target_mat = (target == target.t()).type(self.dtype)
        target_sums = target_mat.sum(1, keepdim=True)

        subset_mat = subset.expand_as(target_mat).type(self.dtype)

        loss = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).abs()
        loss.div_(target_sums)
        loss = loss.sum()**2

        eval_vec = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).sign() / target_sums
        n_missed = (eval_vec * (eval_vec.sign() < 0).type(self.dtype)).abs().sum()
        n_many = (eval_vec * (eval_vec.sign() > 0).type(self.dtype)).sum()
        n_one = n_clusters - n_missed - n_many 

        if not n_missed and not n_many:
            n_perfect = 1.
        else: 
            n_perfect = 0.

        subset_size = subset.sum()
        prec = (n_one + n_many) / subset_size 
        rec = 1 - (n_missed / 10)

        return loss, prec, rec, subset_size
       
    def _assess(self, target, subset):

        set_size = target.size(0)
        target = target.expand(set_size, set_size)
        target_mat = (target == target.t()).type(self.dtype)
        target_sums = target_mat.sum(1, keepdim=True)
    
        subset_mat = subset.expand_as(target_mat).type(self.dtype)
    
        loss = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).abs()
        loss.div_(target_sums)
        loss = loss.sum()**2
        
        return loss

    def _eval_assess(self, target, subset):

        set_size = self.set_size
        n_clusters = self.n_clusters
        target = target.expand(set_size, set_size)
        target_mat = (target == target.t()).type(self.dtype)
        target_sums = target_mat.sum(1, keepdim=True)
    
        subset_mat = subset.expand_as(target_mat).type(self.dtype)
    
        eval_vec = ((target_mat * subset_mat).sum(1, keepdim=True) - torch.ones(set_size).type(self.dtype)).sign() / target_sums
        n_missed = (eval_vec * (eval_vec.sign() < 0).type(self.dtype)).abs().sum()
        n_many = (eval_vec * (eval_vec.sign() > 0).type(self.dtype)).sum()
        n_one = n_clusters - n_missed - n_many 

        if not n_missed and not n_many:
            n_perfect = 1.
        else: 
            n_perfect = 0.

        return n_missed, n_one, n_many, n_perfect

    def run(self, words, context, batch_size, alpha_iter):
        """
        This may be used by sample and by evaluate. 
        Samples once from DPP. 
        Can be used with any batch_size. 
        Returns a tensor of many subsets
        """
        set_size = self.set_size
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out

        actions = self.saved_subsets = [[] for i in range(batch_size)]
        rewards = self.saved_losses =  [[] for i in range(batch_size)]
        cum_loss = 0.

        # Concatenate individual words and set context
        # Dimensions are batch_size x set_size x kernel_in
        batch_x = Variable(torch.cat([words, context], dim = 2))

        # Compute embedding of DPP kernel
        batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
        batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

        for i, kernel in enumerate(batch_kernel):
            vals, vecs = custom_decomp()(kernel)
            for j in range(alpha_iter):
                subset = DPP()(vals, vecs)
                actions[i].append(subset)

        subset_tensor = torch.stack([torch.stack(subsets) for subsets in actions])

        return subset_tensor

    def evaluate(self, test_iter):

        set_size = self.set_size
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out        
        set_size = self.set_size
        n_clusters = self.n_clusters
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out
        embd_dim = self.pred_in
        dtype = self.dtype
        criterion = self.criterion
        
        cum_loss = 0.
        cum_prec = 0.
        cum_rec = 0.
        cum_ssize = 0.

        n_missed = 0.
        n_one = 0
        n_many = 0.
        n_perfect = 0.

        mean = 0.
        temp = 0.
        var = 0.

        criterion = self.criterion
        batch_size = 1
        embd_dim = self.pred_in

        for t in range(test_iter):
            picks = [[] for i in range(batch_size)]
            words, context, target, index = self.generate(batch_size)

            # Concatenate individual words and set context
            # Dimensions are batch_size x set_size x kernel_in
            batch_x = Variable(torch.cat([words, context], dim = 2))

            # Compute embedding of DPP kernel
            batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
            batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

            for i, kernel in enumerate(batch_kernel):
                vals, vecs = custom_decomp()(kernel)
                subset = DPP()(vals, vecs)
                pick = subset.diag().mm(Variable(words[i])).sum(0, keepdim=True) 
                picks[i].append(pick)

                _, prec, rec, ssize  = self._big_assess(index[i], subset.data)
                missed, one, many, perfect = self._eval_assess(index, subset.data)
                cum_prec += prec
                cum_rec += rec
                cum_ssize += ssize
                n_missed += missed
                n_one += one
                n_many += many
                n_perfect += perfect 

            picks = torch.stack([torch.stack(pick) for pick in picks]).view(-1, embd_dim)
            preds = self.pred_net(picks).view(1, -1)

            targets = target.unsqueeze(0).expand_as(preds)
            loss = criterion(preds, Variable(targets, volatile=True))
            cum_loss += loss.data[0]

            delta = ssize - mean
            mean += delta / (t+1)
            delta2 = ssize - mean
            temp += delta * delta2
        
            var = temp / test_iter

        print("Average Subset Size: ", mean)
        print("Subset Variance: ", var)
        print("Average Loss", cum_loss / test_iter)
        print("n_missed share", n_missed / (test_iter * self.n_clusters))
        print("n_one share", n_one / (test_iter * self.n_clusters))
        print("n_many share", n_many / (test_iter * self.n_clusters))
        print("n_perfect share", n_perfect/ test_iter)

    def random_benchmark(self, test_iter):

        set_size = self.set_size
        dtype = self.dtype
        p = self.n_clusters / self.set_size

        n_missed  = 0.0
        n_one     = 0.0
        n_many    = 0.0
        n_perfect = 0.0

        loss = 0.0

        mean = 0.0 
        temp = 0.0
        var  = 0.0

        # We could avoid the loop and instead process one big batch?
        for t in range(test_iter):
            _, _, target  = self.generate(1)
            random_subset = torch.bernoulli(p * torch.ones(set_size)).type(dtype)
            missed, one, many, perfect = self._eval_assess(target, random_subset)
            
            loss += self._assess(target[0], random_subset)
            n_missed += missed
            n_one += one
            n_many += many
            n_perfect += perfect 

            # Gather subset statistics
            size = random_subset.sum()
            delta =  size - mean
            mean += delta / (t+1)
            delta2 = size - mean
            temp += delta * delta2
        
        var = temp / test_iter

        print("Average Subset Size: ", mean)
        print("Subset Variance: ", var)
        print("Average Loss", loss / test_iter)
        print("n_missed share", n_missed / (test_iter * self.n_clusters))
        print("n_one share", n_one / (test_iter * self.n_clusters))
        print("n_many share", n_many / (test_iter * self.n_clusters))
        print("n_perfect share", n_perfect/ test_iter)

class SimFilter(object):

    def __init__(self, network_params, dtype):
        
        # Customizable parameters
        self.set_size = set_size = network_params['set_size'] # 40
        self.n_clusters = n_clusters = network_params['n_clusters'] # 
        self.max_sig = network_params['max_sig']

        self.dtype = dtype

        # Fixed parameters
        self.kernel_in = kernel_in = 100
        self.kernel_h = kernel_h = 500
        self.kernel_out = kernel_out = 100

        self.pred_in = pred_in = 50 # kernel_in / 2
        self.pred_h = pred_h = 500
        self.pred_out = pred_out = 100
        
        # 2-Hidden-Layer Networks 
        self.kernel_net = torch.nn.Sequential(nn.Linear(kernel_in, kernel_h), nn.ELU(),
                                              nn.Linear(kernel_h, kernel_h), nn.ELU(), nn.Linear(kernel_h, kernel_out))
        self.kernel_net.type(self.dtype)

        # Data
        np.random.seed(0)
        self.s_means = dtype(np.random.randint(-50,50,[n_clusters, int(pred_in)]).astype("float"))
        self.noise_means = torch.zeros(1, int(pred_in)).type(dtype)
        self.means = torch.cat([self.noise_means, self.s_means], dim=0).type(dtype)

        # Convenience
        self.saved_subsets = None
        self.saved_losses = None
        self.saved_baselines = None

        # Record loss
        self.loss_dict = defaultdict(list)

        # Useful intermediate variables 
        self.embedding = None
        self.subset = None
        self.pick = None
        self.pred = None


    def generate(self, batch_size):


        set_size = self.set_size
        n_clusters = self.n_clusters
        max_sig = self.max_sig
        embd_dim = int(self.pred_in)

        means = self.means
        dtype = self.dtype

        words = torch.zeros(batch_size, set_size, embd_dim)
        targets = torch.zeros(batch_size, set_size).long()

        for i in range(batch_size):
            n_sig = 1 + torch.multinomial(torch.ones(max_sig), 1)[0]
            idx = torch.multinomial(torch.ones(n_clusters), n_sig, replacement=True)
            idx.add_(1) # since 0th row of means is noise
            idx = pad_tensor(idx,0,0,set_size) # fill idx up with noise
            idx = idx[torch.randperm(set_size)] # shuffle words in set
            words[i] = means.gather(0, idx.expand(embd_dim, set_size).t())
            targets[i] = idx 

        words = words.normal_().type(dtype)
        context = words.sum(1, keepdim=True).expand_as(words).type(dtype)

        return words, context, targets

    def _assess(self, target, subset, w_junk=1):

        selection = torch.masked_select(target, subset.byte())

        n_sig = (target != 0).sum()
        n_junk = (selection == 0).sum()
        n_caught = (selection != 0).sum()
        n_missed = (torch.masked_select(target, (1-subset.byte())) != 0).sum()
        assert n_missed == n_sig - n_caught
    
        n_junk *= w_junk # apply regularization
        loss = (n_missed + n_junk)**2

        return loss, n_junk, n_missed, n_caught, n_sig

    def train(self, train_iter, batch_size, lr, alpha_iter=1, baseline=True):
        """
        Training the model. 
        Doesn't use the forward pass as want to sample repeatedly!
        """
        set_size = self.set_size
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out

        loss_log = 100
        optimizer = optim.Adam(self.kernel_net.parameters(), lr=lr)


        for t in range(train_iter):
            actions = self.saved_subsets = [[] for i in range(batch_size)]
            rewards = self.saved_losses =  [[] for i in range(batch_size)]

            cum_loss = 0.
            words, context, target = self.generate(batch_size)

            # Concatenate individual words and set context
            # Dimensions are batch_size x set_size x kernel_in
            batch_x = Variable(torch.cat([words, context], dim = 2))

            # Compute embedding of DPP kernel
            batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
            batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

            for i, kernel in enumerate(batch_kernel):
                vals, vecs = custom_decomp()(kernel)
                for j in range(alpha_iter):
                    subset = DPP()(vals, vecs)
                    actions[i].append(subset)
                    loss, _, _, _, _ = self._assess(target[i], subset.data)
                    rewards[i].append(loss)
                    cum_loss += loss

            if baseline:
                self.saved_baselines = [compute_baseline(i) for i in self.saved_losses]
            else: 
                self.saved_baselines = self.saved_losses

            # Register the baselines
            for actions, rewards in zip(self.saved_subsets, self.saved_baselines):
                for action, reward in zip(actions, rewards):
                    action.reinforce(reward)

            pseudo_loss = torch.stack([torch.stack(subsets) for subsets in self.saved_subsets]).sum()
            pseudo_loss.backward(None)
            optimizer.step()
            optimizer.zero_grad()

            self.loss_dict[t].append(cum_loss / (batch_size * alpha_iter))

            if not ((t + 1) % loss_log):
                print("Loss at it ", t+1, " is: ", cum_loss / (batch_size * alpha_iter))  

    def run(self, words, context, batch_size, alpha_iter):
        """
        This may be used by sample and by evaluate. 
        Samples once from DPP. 
        Can be used with any batch_size. 
        Returns a tensor of many subsets
        """
        set_size = self.set_size
        kernel_in = self.kernel_in
        kernel_out = self.kernel_out

        actions = self.saved_subsets = [[] for i in range(batch_size)]
        rewards = self.saved_losses =  [[] for i in range(batch_size)]
        cum_loss = 0.

        # Concatenate individual words and set context
        # Dimensions are batch_size x set_size x kernel_in
        batch_x = Variable(torch.cat([words, context], dim = 2))

        # Compute embedding of DPP kernel
        batch_kernel = self.kernel_net(batch_x.view(-1, kernel_in))
        batch_kernel = batch_kernel.view(-1, set_size, kernel_out)

        for i, kernel in enumerate(batch_kernel):
            vals, vecs = custom_decomp()(kernel)
            for j in range(alpha_iter):
                subset = DPP()(vals, vecs)
                actions[i].append(subset)

        subset_tensor = torch.stack([torch.stack(subsets) for subsets in actions])

        return subset_tensor

    def evaluate(self, test_iter):

        n_junk  = 0.0
        n_missed    = 0.0
        n_caught  = 0.0
        n_sig = 0.0

        t_loss = 0.0

        mean = 0.0 
        temp = 0.0
        var  = 0.0

        # We could avoid the loop and instead process one big batch?
        for t in range(test_iter):

            words, context, target  = self.generate(1)
            subset_tensor = self.run(words, context, 1, 1)
            loss, junk, missed, caught, sig = self._assess(target, subset_tensor.squeeze().data)
            t_loss += loss
            n_junk += junk
            n_missed += missed
            n_caught += caught
            n_sig += sig

            # Gather subset statistics
            size = subset_tensor.data.sum()
            delta =  size - mean
            mean += delta / (t+1)
            delta2 = size - mean
            temp += delta * delta2
        
        var = temp / test_iter

        print("Average Subset Size: ", mean)
        print("Subset Variance: ", var)
        print("Average Loss", t_loss / test_iter)
        print("How much junk per sample: ", n_junk / test_iter)
        print("How many misses per sample: ", n_missed / test_iter)
        print("How many catches per sample:", n_caught / test_iter)

        print("How many signals missed totally:", n_missed / n_sig)
        print("How many signals caught totally:", n_caught / n_sig)


    def random_benchmark(self, test_iter):

        set_size = self.set_size
        dtype = self.dtype

        n_junk  = 0.0
        n_missed    = 0.0
        n_caught  = 0.0
        n_sig = 0.0

        t_loss = 0.0

        mean = 0.0 
        temp = 0.0
        var  = 0.0

        # We could avoid the loop and instead process one big batch?
        for t in range(test_iter):

            _, _, target  = self.generate(1)
            p = torch.nonzero(target).size(0) / set_size # this even gives the random benchmark an unfair advantage
            random_subset = torch.bernoulli(p * torch.ones(set_size)).type(dtype)
            loss, junk, missed, caught, sig = self._assess(target, random_subset)
            
            t_loss += loss
            n_junk += junk
            n_missed += missed
            n_caught += caught
            n_sig += sig 

            # Gather subset statistics
            size = random_subset.sum()
            delta =  size - mean
            mean += delta / (t+1)
            delta2 = size - mean
            temp += delta * delta2
        
        var = temp / test_iter

        print("Average Subset Size: ", mean)
        print("Subset Variance: ", var)
        print("Average Loss", t_loss / test_iter)
        print("How much junk per sample: ", n_junk / test_iter)
        print("How many misses per sample: ", n_missed / test_iter)
        print("How many catches per sample:", n_caught / test_iter)

        print("How many signals missed totally:", n_missed / n_sig)
        print("How many signals caught totally:", n_caught / n_sig)




######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################

#### Make a prediction based on the subset using pred net!!
