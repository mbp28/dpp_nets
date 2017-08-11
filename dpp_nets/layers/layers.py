import torch
import torch.nn as nn
from torch.autograd import Variable
from dpp_nets.my_torch.linalg import custom_decomp, custom_inverse 
from dpp_nets.my_torch.DPP import DPP, AllInOne
from dpp_nets.my_torch.utilities import compute_baseline
from itertools import accumulate

class KernelFixed(nn.Module):

    def __init__(self, embd_dim, hidden_dim, kernel_dim):
        """
        Currently, this creates a 2-hidden-layer network 
        with ELU non-linearities.

        """
        super(KernelFixed, self).__init__()

        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.kernel_dim = kernel_dim

        self.layer1 = nn.Linear(2 * embd_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, kernel_dim)

        self.net = nn.Sequential(self.layer1, nn.tanh(), self.layer2, nn.tanh(), self.layer3)


    def forward(self, words):
        """
        Given words returns kernel of 
        dimension [batch_size, set_size, kernel_dim]

        """

        batch_size, set_size, embd_dim = words.size()
        context = words.mean(1).expand_as(words)
        batch_x = torch.cat([words, context], dim = 2).view(-1, 2 * embd_dim)

        batch_kernel = self.net(batch_x)
        kernels = batch_kernel.view(batch_size, set_size, -1)

        return kernels

class KernelVar(nn.Module):

    def __init__(self, embd_dim, hidden_dim, kernel_dim):
        """
        Currently, this creates a 2-hidden-layer network 
        with ELU non-linearities.

        """
        super(KernelVar, self).__init__()
        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.kernel_dim = kernel_dim

        self.layer1 = nn.Linear(2 * embd_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, kernel_dim)

        self.net = nn.Sequential(self.layer1, nn.tanh(), self.layer2, nn.tanh(), self.layer3)

        self.s_ix = []
        self.e_ix = []


    def forward(self, words):
        """
        Given words, returns batch_kernel of dimension
        [-1, kernel_dim]
        """

        batch_size, max_set_size, embd_dim = words.size()

        # Create context
        lengths = words.sum(2).abs().sign().sum(1)
        context = (words.sum(1) / lengths.expand_as(words.sum(1))).expand_as(words)

        # Filter out zero words 
        mask = words.sum(2).abs().sign().expand_as(words).byte()
        words = words.masked_select(mask).view(-1, embd_dim)
        context = context.masked_select(mask).view(-1, embd_dim)

        # Concatenate and compute kernel
        batch_x = torch.cat([words, context], dim=1)
        batch_kernel = self.net(batch_x)

        # Register indices for individual kernels
        self.s_ix = list(lengths.squeeze().cumsum(0).long().data - lengths.squeeze().long().data)
        self.e_ix = list(lengths.squeeze().cumsum(0).long().data)

        return batch_kernel , words 

class DeepSetPred(nn.Module):

    def __init__(self, embd_dim, hidden_dim, enc_dim, target_dim):

        super(DeepSetPred, self).__init__()

        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.target_dim = target_dim

        # Encodes each word into a different vector
        self.enc_layer1 = nn.Linear(embd_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_layer3 = nn.Linear(hidden_dim, enc_dim)
        self.enc_net = nn.Sequential(self.enc_layer1, nn.tanh(), self.enc_layer2, nn.tanh(), self.enc_layer3)

        # Uses the sum of the encoded vectors to make a final prediction
        self.pred_layer1 = nn.Linear(enc_dim ,hidden_dim)
        self.pred_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_layer3 = nn.Linear(hidden_dim, target_dim)
        self.pred_net = nn.Sequential(self.pred_layer1, nn.tanh(), self.pred_layer2, nn.tanh(), self.pred_layer3)

    def forward(self, word_picks):
        """
        word_picks is a list of lists, which contains word
        tensors, 
        len(outer_list = word_picks) = batch_size, 
        len(inner_list = word_samples) = alpha_iter, 
        word_tensors are of dim = [flex, embd_dim]

        """
        embd_dim = self.embd_dim
        hidden_dim = self.hidden_dim
        enc_dim = self.enc_dim
        target_dim = self.target_dim

        batch_size = len(word_picks)
        encodings = [[] for i in range(batch_size)] 

        for i, word_samples in enumerate(word_picks):
            for words in word_samples:
                enc = self.enc_net(words).sum(0)
                encodings[i].append(enc)

        encodings = torch.stack([torch.stack(enc_samples) for enc_samples in encodings]).squeeze()
        batch_size, alpha_iter, enc_dim = encodings.size()

        batch_encodings = encodings.view(-1, enc_dim)
        batch_pred = self.pred_net(batch_encodings)

        pred = batch_pred.view(batch_size, alpha_iter, target_dim)

        return pred 

class SampleFixed(object):
    """
    No need to wrap this in PyTorch
    module, but do it for convenience? 
    """
    def __init__(self, alpha_iter):

        self.alpha_iter = alpha_iter
        self.saved_subsets = None
        self.exp_sizes = None
        self.saved_picks = None

    def __call__(self, kernels, batched_words):

        batch_size, set_size, kernel_dim = kernels.size()
        batch_size, set_size, embd_dim = batched_words.size()
        alpha_iter = self.alpha_iter

        self.exp_sizes = exp_sizes = []
        self.saved_subsets = actions = [[] for i in range(batch_size)]
        self.saved_picks = picks = [[] for i in range(batch_size)]

        for i, (kernel, words) in enumerate(zip(kernels, batched_words)):
            vals, vecs = custom_decomp()(kernel)
            exp_size = (vals / (1 + vals)).sum()
            exp_sizes.append(exp_size)
            for j in range(alpha_iter):
                while True: # to avoid zero subsets, problematic as not unbiased anymore, temp fix.
                    subset = DPP()(vals, vecs)
                    if subset.data.sum() >= 1:
                        break
                    else:
                        print("Zero Subset was produced. Re-sample")
                        continue
                actions[i].append(subset)
                pick = words.masked_select(Variable(subset.data.byte().expand_as(words.t())).t())
                pick = pick.view(-1, embd_dim)
                picks[i].append(pick)

        return picks 

class SampleVar(object):
    """
    No need to wrap this in PyTorch
    module, but do it for convenience?
    """
    def __init__(self, alpha_iter):

        self.alpha_iter = alpha_iter
        self.saved_subsets = None
        self.saved_picks = None

    def __call__(self, kernels, batched_words, s_ix, e_ix):

        _, kernel_dim = kernels.size()
        batch_size, max_set_size, embd_dim = batched_words.size()

        alpha_iter = self.alpha_iter
        self.saved_subsets = actions = [[] for i in range(batch_size)]
        self.saved_picks = picks = [[] for i in range(batch_size)]

        # Mask words again to make same length as kernel
        mask = batched_words.sum(2).abs().sign().expand_as(batched_words).byte()
        batched_words = batched_words.masked_select(mask).view(-1, embd_dim)
        assert batched_words.size(0) == kernels.size(0)

        for i, (s, e) in enumerate(zip(s_ix, e_ix)):
            words = batched_words[s:e]
            kernel = kernels[s:e]
            #vals, vecs = custom_decomp()(kernel)
            for j in range(alpha_iter):
                subset = AllInOne()(kernel)
                #subset = DPP()(vals, vecs)
                actions[i].append(subset)
                pick = words.masked_select(Variable(subset.data.byte().expand_as(words.t())).t()) # this isn't ideal.
                pick = pick.view(-1, embd_dim)
                picks[i].append(pick)

        return picks 

class custom_backprop(object):
    """
    pred: batch_size x alpha_iter x target_dim
    target: batch_size x target_dim
    """
    def __init__(self, reg=0, reg_mean=0):

        self.saved_losses = None
        self.saved_baselines = None
        self.reg = reg
        self.reg_mean = reg_mean

    def __call__(self, pred, target, action_list, exp_sizes):

        batch_size, alpha_iter, target_dim = pred.size()
        target = target.unsqueeze(1).expand_as(pred)

        losses = (pred - target).pow(2).mean(2)
        self.saved_losses = [[i.data[0] for i in row] for row in losses]
        self.saved_baselines = [compute_baseline(i) for i in self.saved_losses]

        for actions, rewards in zip(action_list, self.saved_baselines):
            for action, reward in zip(actions, rewards):
                action.reinforce(reward)

        if self.reg:
            reg_loss = self.reg * ((torch.stack(exp_sizes) - self.reg_mean)**2).sum()
            reg_loss.backward(retain_variables=True)

        # backpropagate through kernel network
        pseudo_loss = torch.stack([torch.stack(actions) for actions in action_list]).sum()
        pseudo_loss.backward(None)

        # backpropgate through prediction network
            
        loss.backward()

        return loss


"""
Issues: - zero subsets, - , - test var set module,  
"""

class DeepSetBaseline(nn.Module):
    """
    Works with different set sizes, i.e. it does masking!
    """

    def __init__(self, embd_dim, hidden_dim, enc_dim, target_dim):

        super(DeepSetBaseline, self).__init__()

        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.target_dim = target_dim

        # Encodes each word into a different vector
        self.enc_layer1 = nn.Linear(embd_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_layer3 = nn.Linear(hidden_dim, enc_dim)
        self.enc_net = nn.Sequential(self.enc_layer1, nn.tanh(), self.enc_layer2, nn.tanh(), self.enc_layer3)

        # Uses the sum of the encoded vectors to make a final prediction
        self.pred_layer1 = nn.Linear(enc_dim ,hidden_dim)
        self.pred_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_layer3 = nn.Linear(hidden_dim, target_dim)
        self.pred_net = nn.Sequential(self.pred_layer1, nn.tanh(), self.pred_layer2, nn.tanh(), self.pred_layer3)

    def forward(self, words):
        """
        words is a 3D tensor with dimension: batch_size x max_set_size x embd_dim

        """
        embd_dim = self.embd_dim
        hidden_dim = self.hidden_dim
        enc_dim = self.enc_dim
        target_dim = self.target_dim

        batch_size, max_set_size, embd_dim = words.size()

        # Unpacking to send through encoder network
        # Register indices of individual instances in batch for reconstruction
        lengths = words.sum(2).abs().sign().sum(1)
        s_ix = list(lengths.squeeze().cumsum(0).long().data - lengths.squeeze().long().data)
        e_ix = list(lengths.squeeze().cumsum(0).long().data)

        # Filter out zero words 
        mask = words.sum(2).abs().sign().expand_as(words).byte()
        words = words.masked_select(mask).view(-1, embd_dim)

        # Send through encoder network
        enc_words = self.enc_net(words)

        # Compilation of encoded words for each instance in sample
        # Produce summed representation (code) for each instance in batch using encoded words:
        codes = []

        for i, (s, e) in enumerate(zip(s_ix, e_ix)):
            code = enc_words[s:e].sum(0)
            codes.append(code)

        codes = torch.stack(codes).squeeze(1)
        assert batch_size == codes.size(0)
        assert enc_dim == codes.size(1)

        # Produce predictions using codes
        pred = self.pred_net(codes)

        return pred 

class MarginalSampler(nn.Module):
    """
    No sampling, because this module just weights all words by its
    marginal probabilities. 
    """
    def __init__(self):

        super(MarginalSampler, self).__init__()

        self.s_ix = None
        self.e_ix = None

        self.exp_sizes = None

    def forward(self, kernels, words):
        """
        both kernel and words should be 2D, 
        the zero words have been filtered.
        """
        assert kernels.size(0) == words.size(0)
        
        # these need to be set beforehand
        s_ix = self.s_ix
        e_ix = self.e_ix 
        self.exp_sizes = []

        assert s_ix != None and e_ix != None

        output = []
        lengths = []
        for i, (s, e) in enumerate(zip(s_ix, e_ix)):
            # unpack kernel and words
            V = kernels[s:e]
            word = words[s:e]
            # compute marginal kernel K
            #vals, vecs = custom_decomp()(V)
            #K = vecs.mm((1 / (vals + 1)).diag()).mm(vecs.t()) # actually K = (identity - expression) 
            #marginals = (1 - K.diag()).diag() ## need to rewrite custom_decomp to return full svd + correct gradients. 
            # so this is the inefficient way
            identity = Variable(torch.eye(word.size(0)).type(words.data.type()))
            L = V.mm(V.t())
            K = identity - custom_inverse()(L + identity)
            marginals = (K.diag()).diag()
            exp_size = marginals.sum()

            # compute weighted or filtered words for REINFORCE
            weighted_word = marginals.mm(word)

            # append output and lengths for record!
            output.append(weighted_word)
            lengths.append(weighted_word.size(0))
            self.exp_sizes.append(exp_size)

        output = torch.cat(output, dim=0)
        cum_lengths = list(accumulate(lengths))
        
        self.s_ix = [ix1 - ix2 for (ix1, ix2) in zip(cum_lengths, lengths)]
        self.e_ix = cum_lengths

        return output

class ReinforceSampler(nn.Module):

    def __init__(self, alpha_iter):

        super(ReinforceSampler, self).__init__()
        
        self.alpha_iter = alpha_iter
        self.saved_subsets = None
        self.saved_picks = None
        self.s_ix = None
        self.e_ix = None

        self.exp_sizes = None

    def forward(self, kernels, words):

        # need to set beforehand!
        s_ix = self.s_ix
        e_ix = self.e_ix

        self.exp_sizes = []

        assert s_ix != None and e_ix != None

        alpha_iter = self.alpha_iter
        batch_size = len(s_ix)
        embd_dim = words.size(1)

        output = []
        lengths = []
        actions = self.saved_subsets = [[] for i in range(batch_size)]

        for i, (s, e) in enumerate(zip(s_ix, e_ix)):
            V = kernels[s:e]
            word = words[s:e]
            vals, vecs = custom_decomp()(V) # check if gradients work for non-square matrix 
            exp_size = (vals / (1 + vals)).pow(2).sum()
            for j in range(alpha_iter):
                while True: # to avoid zero subsets, problematic as not unbiased anymore, temp fix.
                    subset = AllInOne()(V) # scrap this after gradient check!
                    #subset = DPP()(vals, vecs)
                    if subset.data.sum() >= 1:
                        break
                    else:
                        print("Zero Subset was produced. Re-sample")
                        continue
                actions[i].append(subset)
                pick = subset.diag().mm(word) # but this creates zero rows, however it links the two graphs :)
                pick = pick.masked_select(Variable(subset.data.byte().expand_as(pick.t())).t())
                pick = pick.view(-1, embd_dim)

                output.append(pick)
                lengths.append(pick.size(0))
                self.exp_sizes.append(exp_size)

        output = torch.cat(output, dim=0)
        cum_lengths = list(accumulate(lengths))
        
        self.s_ix = [ix1 - ix2 for (ix1, ix2) in zip(cum_lengths, lengths)]
        self.e_ix = cum_lengths
        
        return output


class PredNet(nn.Module):

    def __init__(self, embd_dim, hidden_dim, enc_dim, target_dim):

        super(PredNet, self).__init__()

        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.target_dim = target_dim

        # Encodes each word into a different vector
        self.enc_layer1 = nn.Linear(embd_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_layer3 = nn.Linear(hidden_dim, enc_dim)
        self.enc_net = nn.Sequential(self.enc_layer1, nn.tanh(), self.enc_layer2, nn.tanh(), self.enc_layer3)

        # Uses the sum of the encoded vectors to make a final prediction
        self.pred_layer1 = nn.Linear(enc_dim ,hidden_dim)
        self.pred_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_layer3 = nn.Linear(hidden_dim, target_dim)
        self.pred_net = nn.Sequential(self.pred_layer1, nn.tanh(), self.pred_layer2, nn.tanh(), self.pred_layer3)

        self.s_ix = None
        self.e_ix = None

    def forward(self, words):

        # these need to be set beforehand!
        s_ix = self.s_ix
        e_ix = self.e_ix

        assert s_ix != None and e_ix != None
        enc_dim = self.enc_dim

        # Send through encoder network. 
        enc_words = self.enc_net(words)
        
        codes = []
        for i, (s, e) in enumerate(zip(s_ix, e_ix)):
            code = enc_words[s:e].sum(0)
            codes.append(code)

        codes = torch.stack(codes).squeeze(1)
        assert enc_dim == codes.size(1)

        # Produce predictions using codes
        pred = self.pred_net(codes)

        return pred 

class MarginalTrainer(nn.Module):

    def __init__(self, Embedding, hidden_dim, kernel_dim, enc_dim, target_dim):

        super(MarginalTrainer, self).__init__()

        self.embd = Embedding
        
        self.embd_dim = self.embd.weight.size(1)
        self.hidden_dim = hidden_dim
        self.kernel_dim = kernel_dim
        self.enc_dim = enc_dim
        self.target_dim = target_dim

        self.kernel_net = KernelVar(self.embd_dim, self.hidden_dim, self.kernel_dim)
        self.sampler = MarginalSampler()
        self.pred_net = PredNet(self.embd_dim, self.hidden_dim, self.enc_dim, self.target_dim)

        self.criterion = nn.MSELoss()
        self.activation = None
        
        self.pred = None

        self.pred_loss = None 
        self.reg_loss = None
        self.loss = None

        self.reg = None
        self.reg_mean = None

    def forward(self, reviews, target):

        words = self.embd(reviews)

        kernel, words = self.kernel_net(words) # returned words are masked now!

        self.sampler.s_ix = self.kernel_net.s_ix
        self.sampler.e_ix = self.kernel_net.e_ix
        
        weighted_words = self.sampler(kernel, words) 
        
        self.pred_net.s_ix = self.sampler.s_ix
        self.pred_net.e_ix = self.sampler.e_ix
        
        self.pred = self.pred_net(weighted_words)

        if self.activation:
            self.pred = self.activation(self.pred)

        self.pred_loss = self.criterion(self.pred, target)

        if self.reg:
            self.reg_loss = self.reg * (torch.stack(self.sampler.exp_sizes) - self.reg_mean).pow(2).mean()
            self.loss = self.pred_loss + self.reg_loss
        else:
            self.loss = self.pred_loss

        return self.loss

class ReinforceTrainer(nn.Module):

    def __init__(self, KernelNet, Sampler, PredNet):

        super(ReinforceTrainer, self).__init__()

        self.kernel_net = KernelNet
        self.sampler = Sampler
        self.pred_net = PredNet

        self.alpha_iter = self.sampler.alpha_iter

        self.criterion = nn.MSELoss()
        self.activation = None

        self.reg = None
        self.reg_mean = None

        # Register
        self.pred = None

        self.pred_loss = None 
        self.reg_loss = None
        self.loss = None

        self.saved_subsets = None
        self.saved_losses = None
        self.saved_baselines = None


    def forward(self, words, target):

        batch_size = target.size(0)
        alpha_iter = self.alpha_iter
        target_dim = target.size(1)

        target = target.unsqueeze(1).expand(batch_size, alpha_iter, target_dim).contiguous().view(batch_size * alpha_iter, target_dim)
        kernel, words = self.kernel_net(words)

        self.sampler.s_ix = self.kernel_net.s_ix
        self.sampler.e_ix = self.kernel_net.e_ix
        
        weighted_words = self.sampler(kernel, words)
        
        self.pred_net.s_ix = self.sampler.s_ix
        self.pred_net.e_ix = self.sampler.e_ix
        
        self.pred = self.pred_net(weighted_words)

        if self.activation:
            self.pred = self.activation(self.pred)

        self.pred_loss = self.criterion(self.pred, target)

        if self.reg:
            self.reg_loss = self.reg * (torch.stack(self.sampler.exp_sizes) - self.reg_mean).pow(2).mean()
            self.loss = pred_loss + reg_loss
            # print("reg_loss is:", reg_loss.data[0])
        else:
            self.loss = self.pred_loss

        # add computation of baselines and registering reawards here!!
        losses = (self.pred - target).pow(2).view(batch_size, alpha_iter, target_dim).mean(2)

        self.saved_losses = [[i.data[0] for i in row] for row in losses]
        if self.alpha_iter > 1:
            self.saved_baselines = [compute_baseline(i) for i in self.saved_losses]
        else:
            self.saved_baselines = self.saved_losses

        self.saved_subsets = self.sampler.saved_subsets

        for actions, rewards in zip(self.saved_subsets, self.saved_baselines):
            for action, reward in zip(actions, rewards):
                action.reinforce(reward)
                
        return loss

## Kernel Network
# Input: words (batch_size x max_set_size x embd_dim)
# will need to perform masking and produce indices
# Output: kernel (#true words x kernel_dim)

## MarginalTrainer
# Input: needs masked words and kernel
# Iterates through instance in batch, using indices 
# for each instance computes marginal probabilities, 
# creates a diag matrice and multiplies words with it
# then cat all words  to produce output
# Output: weighted words of dim (#true words x embd_dim)

## ReInforce Trainer
# kernel, words
# iteratres through instance in batch, using indices
# for each instance samples word indices
# separates graphs by masking using word indices
# cats selected words to produce output
# Output: selected words of dim (#selected words x embd_dim)

## Prediction Network
# Input: Weighted/ selected words (ideally: #true words x embd_dim)
# can then send through encoder
# need INDICES to mean correctly and produce batch_size x enc_dim
# can then send again to produce predictions
# Output: Predictions (batch_size x target_dim)

# Challenges:
# 1) Need to feed right indices into prediction network
# particularly indices will change then using ReInforce trainer
# because some words are now not fed in anymore. 
# 2) Then using ReInforceTrainer, will need to adjust training 
# procedure because graphs are disjoint, because of mask
# 3) compute marginal probabilities and back

# as a last step combine these layers into a marginal network 
# and a reinforce network,

# So far,
# got reinforce trainer and marginal trainer and can plug-and-play
# need to do custom_decomp now to test gradients.!! Do later!
# Write ready-script for training and validation of
# a) baseline
# b) DPP marginal trainer
# c) REINFORCE trainer
# --> think about which settings to change - regularization_mean + lr probably
# submit to EULER and start installing directory on LAS group. 



