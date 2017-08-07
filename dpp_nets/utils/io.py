import torch
import gzip
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import json
import numpy as np

from torch.utils.data import TensorDataset
from dpp_nets.my_torch.utilities import pad_tensor


def embd_iterator(embd_path):
    with gzip.open(embd_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                embd = np.array([float(x) for x in parts[1:]])
                yield word, embd

def data_iterator(data_path):
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            target, sep, words = line.partition("\t")
            words, target = words.split(), target.split()
            if len(words):
                target = torch.Tensor([float(v) for v in target])
                yield words, target

def make_embd(embd_path, only_index_dict=False, save_path=None):
    
    # Create dictionaries
    ix_to_word = {}
    ix_to_vecs = {}
    word_to_ix = {}
    
    for ix, (word, vecs) in enumerate(embd_iterator(embd_path)):
        ix_to_word[ix] = word
        ix_to_vecs[ix] = vecs
        word_to_ix[word] = ix
    
    vocab_size, embd_dim = len(ix_to_word), len(ix_to_vecs[0])
    
    if only_index_dict:
        return word_to_ix
    
    embd = torch.zeros(1 + vocab_size, embd_dim)
    for i, vec in enumerate(ix_to_vecs.values(), 1): 
        embd[i] = torch.FloatTensor(vec)

    embd_weight_dict = OrderedDict([('weight', embd)])

    if save_path:
        save_path = os.path.join(save_path,'embeddings.pt')
        torch.save(embd_weight_dict, save_path)
    else:
        embd_layer = nn.Embedding(1 + vocab_size, embd_dim, padding_idx=0)
        embd_layer.load_state_dict(embd_weight_dict)
        embd_layer.weight.requires_grad = False
        return embd_layer, word_to_ix  

def load_embd(embd_dict_path):
    
    embd_weight_dict = torch.load(embd_dict_path)
    vocab_size, embd_dim = embd_weight_dict['weight'].size()
    embd_layer = nn.Embedding(vocab_size, embd_dim)
    embd_layer.load_state_dict(embd_weight_dict)
    embd_layer.weight.requires_grad = False

    return embd_layer


def make_tensor_dataset(data_path, word_to_ix, max_set_size=0, save_path=None):
        
    if not max_set_size:
        for (review, target) in data_iterator(data_path):
            review = [(word in word_to_ix) for word in review]
            max_set_size = max(sum(review),max_set_size)
            
    reviews, targets = [], []

    for (review, target) in data_iterator(data_path):
        review = [word_to_ix[word] + 1 for word in review if word in word_to_ix]
        review = torch.LongTensor(review)
        review = pad_tensor(review, 0, 0, max_set_size)
        reviews.append(review)
        targets.append(target)
    
    reviews = torch.stack(reviews)
    targets = torch.stack(targets)

    dataset = TensorDataset(reviews, targets)

    if save_path:
        torch.save(dataset, save_path)    
    else:
        return dataset

def load_tensor_dataset(data_set_path):

	dataset = torch.load(data_set_path)

	return dataset

def missing_embd(data_path, word_to_ix):
    missing = []
    for (review, target) in data_iterator(data_path):
        missed = [word for word in review if word not in word_to_ix]
        missing.append(missed)
    
    # if interested in creating a set
    # missing.extend(missed)
    # missing = list(set(missing))
    return missing
    