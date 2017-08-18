from collections import OrderedDict, namedtuple
import json
import re
import csv
import gzip
import os
import numpy as np
from nltk import word_tokenize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from dpp_nets.utils.io import embd_iterator, make_embd
from dpp_nets.my_torch.utilities import pad_tensor

def create_ubuntu(embd_path, database_path, train_path, val_path):
    embd_layer, word_to_ix  = make_embd(embd_path)
    id_to_content, word_to_ix, max_title_size, max_body_size = create_id_to_content(database_path, word_to_ix)
    embd = update_embd(embd_layer, word_to_ix)
    train_title_ds = create_train_set(train_path, id_to_content, max_title_size, max_body_size)
    val_title_ds = create_validate_set(val_path, id_to_content, max_title_size, max_body_size)

    return embd, train_title_ds, val_title_ds #body_ds 

def create_id_to_content(database_path, word_to_ix):
    max_title_size = 0
    max_body_size = 0
    c_ix = len(word_to_ix)
    question = namedtuple('question', 'id title body title_ix body_ix')
    id_to_content = {}
    
    with gzip.open(database_path, 'rt') as f:
        for line in f:
            id, title, body = line.split("\t")
            id = int(id)
            if len(title) == 0:
                print(id)
                empty_cnt += 1
                continue  
            title = word_tokenize(title)
            body = word_tokenize(body)

            title_ix = []
            for word in title:
                if word in word_to_ix.keys():
                    title_ix.append(word_to_ix[word] + 1)
                else:
                    word_to_ix[word] = c_ix
                    title_ix.append(c_ix + 1)
                    c_ix += 1
            max_title_size = max(max_title_size, len(title_ix))

            body_ix = []
            for word in body:
                if word in word_to_ix.keys():
                    body_ix.append(word_to_ix[word] + 1)
                else:
                    word_to_ix[word] = c_ix
                    body_ix.append(c_ix + 1)
                    c_ix += 1
            max_body_size = max(max_body_size, len(body_ix))

            content = question(id, title, body, title_ix, body_ix)
            id_to_content[id] = content
    return id_to_content, word_to_ix, max_title_size, max_body_size

def update_embd(old_embd, word_to_ix):
    old_vocab, embd_dim = old_embd.weight.size()
    
    random_embd = torch.randn(len(word_to_ix) - old_vocab + 1, embd_dim)
    new_embd = nn.Embedding(len(word_to_ix) + 1, embd_dim)
    new_embd.weight = nn.Parameter(torch.cat([old_embd.weight.data, random_embd]))
    new_embd = nn.Embedding(len(word_to_ix), 200)
    new_embd.weight = nn.Parameter(torch.cat([old_embd.weight.data, random_embd]))
    
    return new_embd


def create_train_set(train_path, id_to_content, max_title_size, max_body_size):
    titles = []
    bodies = []
    targets = []
    
    with open(train_path) as f:
        for line in f:
            q_id, pos, neg = line.split("\t")
            q_id = int(q_id)
            q_content = id_to_content[q_id]
            q_title = pad_tensor(torch.LongTensor(np.array(q_content.title_ix)),0,0, max_title_size)
            q_body = pad_tensor(torch.LongTensor(np.array(q_content.body_ix)), 0,0, max_body_size)
            pos = [id_to_content[int(id)] for id in pos.split()]
            neg = [id_to_content[int(id)] for id in neg.split()]
            targets.extend([1] * len(pos))
            targets.extend([-1] * len(neg))

            pos_pairs_title = [torch.stack([q_title, pad_tensor(torch.LongTensor(np.array(q.title_ix)),0,0,max_title_size)]) 
                               for q in pos]
            neg_pairs_title = [torch.stack([q_title, pad_tensor(torch.LongTensor(np.array(q.title_ix)),0,0,max_title_size)])
                                for q in neg]

            titles.extend(pos_pairs_title)
            titles.extend(neg_pairs_title)        

            #pos_pairs_body = [torch.stack([q_body, pad_tensor(torch.LongTensor(np.array(q.body_ix)),0,0,max_body_size)]) 
            #                   for q in pos]
            #neg_pairs_body = [torch.stack([q_body, pad_tensor(torch.LongTensor(np.array(q.body_ix)),0,0,max_body_size)]) 
            #                   for q in neg]

            #bodies.extend(pos_pairs_body)
            #bodies.extend(neg_pairs_body)

        titles = torch.stack(titles)
        #bodies = torch.stack(bodies)
        targets = torch.FloatTensor(targets)
        
    title_ds = TensorDataset(titles, targets)
    #body_ds = TensorDataset(bodies, targets)
    
    return title_ds #, body_ds

def create_validate_set(val_path, id_to_content, max_title_size, max_body_size):
    
    data_tensor = []
    target_tensor = []

    with open(val_path) as f:
    
        for line in f:
            q_id, pos, candidates, _ = line.split("\t")
            q_id = int(q_id)
            pos = [int(id) for id in pos.split()]
            candidates = [int(id) for id in candidates.split()]

            # Check
            if len(pos) == len(candidates):
                continue

            # Create target
            target = [1 if i in pos else 0 for i in candidates]
            target = [-1] + target
            target = torch.ByteTensor(target)

            # Create x Tensor 
            all_ids = [q_id] + candidates 
            all_ids = [id_to_content[id] for id in all_ids]
            all_ids = [pad_tensor(torch.LongTensor(np.array(id.title_ix)),0,0,max_title_size) for id in all_ids]
            all_ids = torch.stack(all_ids)
            data_tensor.append(all_ids)
            target_tensor.append(target)

    data_tensor = torch.stack(data_tensor)
    target_tensor = torch.stack(target_tensor)
    
    title_ds = TensorDataset(data_tensor, target_tensor)
    
    return title_ds #, body_ds    #body_ds = TensorDataset(bodies, targets)

def compute_ap(scores, target):
    
    if not target.sum():
        return 0 
    
    # sort the targets according to scores
    order = scores.argsort()
    copy = target.copy()
    target = copy[order]
    
    # compute average precision
    tp_k = target.cumsum() * target
    tot_k = np.arange(1, len(target) + 1)
    p_k = tp_k / tot_k
    ap = np.mean(p_k[p_k > 0])
    
    return ap