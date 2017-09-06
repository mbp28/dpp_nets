import torch
import torch.nn as nn
from torch.autograd import Variable

from dpp_nets.my_torch.linalg import custom_decomp, custom_inverse 
from dpp_nets.my_torch.DPP import DPP, AllInOne
from dpp_nets.my_torch.utilities import compute_baseline
from itertools import accumulate

class NetBaseline(nn.Module):

    def __init__(self, embd_dim, hidden_dim, target_dim):

        super(NetBaseline, self).__init__()

        self.embd_dim = embd_dim 
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # Prediction 
        self.pred_layer1 = nn.Linear(embd_dim ,hidden_dim)      
        self.pred_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_layer3 = nn.Linear(hidden_dim, target_dim)
        self.pred_net = nn.Sequential(self.pred_layer1, nn.ReLU(), self.pred_layer2, nn.ReLU(), self.pred_layer3)
        
    def forward(self, words):

        batch_size, max_set_size, embd_dim = words.size()
        word_sums = words.sum(1) 
        lengths = Variable(words.data.sum(2, keepdim=True).abs().sign().sum(1).expand_as(word_sums))
        word_means = word_sums / lengths
        
        pred = self.pred_net(word_means)

        return pred

class SetNetBaseline(nn.Module):
    """
    Works with different set sizes, i.e. it does masking!
    """

    def __init__(self, embd_dim, hidden_dim, enc_dim, target_dim):

        super(SetNetBaseline, self).__init__()

        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.target_dim = target_dim

        # Encodes each word into a different vector
        self.enc_layer1 = nn.Linear(embd_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_layer3 = nn.Linear(hidden_dim, enc_dim)
        self.enc_net = nn.Sequential(self.enc_layer1, nn.ReLU(), self.enc_layer2, nn.ReLU(), self.enc_layer3)

        # Uses the sum of the encoded vectors to make a final prediction
        self.pred_layer1 = nn.Linear(enc_dim ,hidden_dim)
        self.pred_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_layer3 = nn.Linear(hidden_dim, target_dim)
        self.pred_net = nn.Sequential(self.pred_layer1, nn.ReLU(), self.pred_layer2, nn.ReLU(), self.pred_layer3)

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
        lengths = words.data.sum(2, keepdim=True).abs().sign().sum(1, keepdim=True)
        s_ix = list(lengths.squeeze().cumsum(0).long() - lengths.squeeze().long())
        e_ix = list(lengths.squeeze().cumsum(0).long())

        # Filter out zero words 
        mask = words.data.sum(2, keepdim=True).abs().sign().expand_as(words).byte()
        words = words.masked_select(Variable(mask)).view(-1, embd_dim)

        # Send through encoder network
        enc_words = self.enc_net(words)

        # Compilation of encoded words for each instance in sample
        # Produce summed representation (code) for each instance in batch using encoded words:
        codes = []

        for i, (s, e) in enumerate(zip(s_ix, e_ix)):
            code = enc_words[s:e].mean(0, keepdim=True)
            codes.append(code)

        codes = torch.stack(codes).squeeze(1)
        assert batch_size == codes.size(0)
        assert enc_dim == codes.size(1)

        # Produce predictions using codes
        pred = self.pred_net(codes)

        return pred 

class AttentionBaseline(nn.Module):
    """
    Works with different set sizes, i.e. it does masking!
    """

    def __init__(self, embd_dim, hidden_dim, target_dim):

        super(AttentionBaseline, self).__init__()

        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # Attention Network 
        self.attention_layer = nn.Sequential(nn.Linear(2 * embd_dim, hidden_dim), nn.Tanh())
        self.v = nn.Parameter(torch.randn(hidden_dim, 1))

        # Uses the sum of the encoded vectors to make a final prediction
        self.pred_layer1 = nn.Linear(embd_dim ,hidden_dim)
        self.pred_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_layer3 = nn.Linear(hidden_dim, target_dim)
        self.pred_net = nn.Sequential(self.pred_layer1, nn.ReLU(), self.pred_layer2, nn.ReLU(), self.pred_layer3)

        self.s_ix = []
        self.e_ix = []

        self.attention = []

    def forward(self, words):
        """
        words is a 3D tensor with dimension: batch_size x max_set_size x embd_dim

        """
        embd_dim = self.embd_dim
        hidden_dim = self.hidden_dim
        target_dim = self.target_dim

        batch_size, max_set_size, embd_dim = words.size()

        # Create context
        lengths = words.sum(2, keepdim=True).abs().sign().sum(1, keepdim=True)
        context = (words.sum(1, keepdim=True) / lengths.expand_as(words.sum(1, keepdim=True))).expand_as(words)

        # Filter out zero words 
        mask = words.data.sum(2, keepdim=True).abs().sign().expand_as(words).byte()
        words = words.masked_select(Variable(mask)).view(-1, embd_dim)
        context = context.masked_select(Variable(mask)).view(-1, embd_dim)

        # Concatenate and compute attention
        batch_x = torch.cat([words, context], dim=1)
        attention_unnorm = self.attention_layer(batch_x).mm(self.v)

        self.s_ix = list(lengths.squeeze().cumsum(0).long().data - lengths.squeeze().long().data)
        self.e_ix = list(lengths.squeeze().cumsum(0).long().data)

        # Apply attention
        reps = []
        for i, (s, e) in enumerate(zip(self.s_ix, self.e_ix)):
            attention = (nn.Softmax()(attention_unnorm[s:e].t())).t()
            self.attention.append(attention.data)
            rep = (attention * words[s:e]).sum(0)
            reps.append(rep)

        weighted_words = torch.stack(reps)
        assert weighted_words.size(0) == batch_size


        pred = self.pred_net(weighted_words)

        return pred 
