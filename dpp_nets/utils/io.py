import torch
import gzip
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import json
import numpy as np


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
                target = np.asarray([float(v) for v in target])  
                yield words, target

def make_embd(embd_path, word_to_ix=False, save_path=None):
    
    # Create dictionaries
    ix_to_word = {}
    ix_to_vecs = {}
    word_to_ix = {}
    
    for ix, (word, vecs) in enumerate(embd_iterator(embd_path)):
        ix_to_word[ix] = word
        ix_to_vecs[ix] = vecs
        word_to_ix = {word: ix}
    
    vocab_size, embd_dim = len(ix_to_word), len(ix_to_vecs[0])
    
    if word_to_ix:
        return word_to_ix
    
    embd = torch.zeros(1 + vocab_size, embd_dim)
    for i, vec in enumerate(ix_to_vecs.values(), 1): 
        embd[i] = vec

    embd_weight_dict = OrderedDict([('weight', embd)])

    if save:
        torch.save(embd_weight_dict, 'embeddings.pt')    
    else:
        embd_layer = nn.Embedding(1 + vocab_size, embd_dim, padding_idx=0)
        embd_layer.load_state_dict(embd_weight_dict)
        embd.weight.requires_grad = False
        return embd_layer

def load_embd(embd_dict_path):
    
    embd_weight_dict = torch.load(embd_dict_path)
    vocab_size, embd_dim = embd_weight_dict['weight'].size()
    embd_layer = nn.Embedding(vocab_size, embd_dim)
    embd_layer.load_state_dict(embd_weight_dict)
    embd.weight.requires_grad = False

    return embd_layer

