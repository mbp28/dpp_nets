import numpy as np
import gzip
import string
import spacy
import torch 
import torch.nn as nn

from collections import OrderedDict
from torch.utils.data import Dataset
from dpp_nets.my_torch.utilities import pad_tensor
from torch.autograd import Variable

def filter_stops(tree, vocab):
    return (token.text for token in tree if not token.is_stop and token.text in vocab.word2index)

def yield_chunks(doc, vocab, MAX_CHUNK_LENGTH):
    seen = set()
    for token in doc:
        t = tuple((filter_stops(token.subtree, vocab)))
        if t and t not in seen:
            seen.add(t)
            #ixs = [vocab.word2index[word] if word in vocab.word2index else print(word) for word in t]
            ixs = torch.LongTensor([vocab.word2index[word] for word in t])
            ixs = pad_tensor(ixs,0,0,MAX_CHUNK_LENGTH)
            yield ixs

def yield_chunk_vec(doc, vocab, embd):
    seen = set()
    for token in doc:
        t = tuple((filter_stops(token.subtree, vocab)))
        if t and t not in seen:
            seen.add(t)
            ixs = torch.LongTensor([vocab.word2index[word] for word in t])
            embd_mat = embd(Variable(ixs)).mean(0)
            yield embd_mat

def process_batch(nlp, vocab, embd, batch):

    MAX_CHUNK_LENGTH = 271
    MAX_CHUNK_NO = 397

    # maxi = 0
    # for review in batch['review']:
     #   doc = nlp(review)
     #   rep = torch.stack(list(yield_chunk_vec(doc, vocab, embd))).squeeze()
     #   maxi = max(maxi, rep.size(0))

    reps = []
    for review in batch['review']:
        doc = nlp(review)
        rep = torch.stack(list(yield_chunk_vec(doc, vocab, embd))).squeeze()
        rep = torch.cat([rep, Variable(torch.zeros(MAX_CHUNK_NO + 1 - rep.size(0), rep.size(1)))], dim=0)
        reps.append(rep)

    data_tensor =  torch.stack(reps)
    target_tensor = Variable(torch.stack(batch['target']).t().float())
    
    return data_tensor, target_tensor

def yield_sen_vec(doc, vocab, embd):
    seen = set()
    for s in doc.sents:
        t = tuple((filter_stops(s, vocab)))
        if t and t not in seen:
            seen.add(t)
            ixs = torch.LongTensor([vocab.word2index[word] for word in t])
            embd_mat = embd(Variable(ixs)).mean(0)
            yield embd_mat

def process_batch_sens(nlp, vocab, embd, batch):

    MAX_CHUNK_LENGTH = 271
    MAX_SENS_NO = 105

    # maxi = 0
    # for review in batch['review']:
     #   doc = nlp(review)
     #   rep = torch.stack(list(yield_chunk_vec(doc, vocab, embd))).squeeze()
     #   maxi = max(maxi, rep.size(0))

    reps = []
    for review in batch['review']:
        doc = nlp(review)
        rep = torch.stack(list(yield_sen_vec(doc, vocab, embd))).squeeze(1)
        rep = torch.cat([rep, Variable(torch.zeros(MAX_SENS_NO + 1 - rep.size(0),rep.size(1)))],dim=0)
        reps.append(rep)

    data_tensor =  torch.stack(reps)
    target_tensor = Variable(torch.stack(batch['target']).t().float())
    
    return data_tensor, target_tensor


class Vocabulary:
    
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.word2vec = {}
        self.index2vec = {}
        
        self.vocab_size = 0  # Count SOS and EOS
        
    def load_pretrained(self, embd_path):
        
        # Create dictionaries
        for ix, (word, vec) in enumerate(self.embd_iterator(embd_path)):
            
            self.vocab_size += 1            
            self.word2index[word] = self.vocab_size
            self.word2count[word] = 1
            self.index2word[self.vocab_size] = word
            self.word2vec[word] = vec
            self.index2vec[self.vocab_size] = vec
        
    def embd_iterator(self, embd_path):
        with gzip.open(embd_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    word = parts[0]
                    embd = np.array([float(x) for x in parts[1:]])
                    yield word, embd   
                    
    def addLofSentences(self, lofsentences):

        for sentence in lofsentences:
            self.addSentence(sentence)

    def addSentence(self, sentence):
        """
        sentence is a list of words
        """
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            pass
            # self.vocab_size += 1
            # self.word2count[word] = 1
            # self.word2index[word] = self.vocab_size
            # self.index2word[self.vocab_size] = word
            # Create random normal vector
            # vec = np.random.normal(size=200)
            # self.word2vec[word] = vec
            # self.index2vec[self.vocab_size] = vec            
        else:
            self.word2count[word] += 1


def create_clean_vocabulary(embd_path, data_path):
    
    vocab = Vocabulary()
    vocab.load_pretrained(embd_path)
    MIN_WORD_OCCURENCE = 10
    EMBD_DIM = 200

    # Create random word vectors for words that are missing
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            target, sep, review = line.partition("\t")
            review, target = review.split(), target.split()
            if len(review):
                vocab.addSentence(review)

    # Use spacy
    nlp = spacy.load('en')

    # Re-define stop words
    nlp.vocab["not"].is_stop = False
    nlp.vocab["no"].is_stop = False
    nlp.vocab['...'].is_stop = True
    nlp.vocab["\n"].is_stop = True
    nlp.vocab["\t"].is_stop = True
    for symbol in list(string.punctuation):
        nlp.vocab[symbol].is_stop = True

    for word, count in vocab.word2count.items():
        if count < 10:
            nlp.vocab[word].is_stop = True

    # Create an embedding
    embd_matrix = torch.zeros(len(vocab.index2vec) + 1, EMBD_DIM)    
    for ix, vec in vocab.index2vec.items():
        embd_matrix[ix] = torch.FloatTensor(vec)

    embd_dict = OrderedDict([('weight', embd_matrix)])
    embd = nn.Embedding(len(vocab.index2vec) + 1, EMBD_DIM, padding_idx=0)
    embd.load_state_dict(embd_dict)


    return nlp, vocab, embd

class BeerDataset(Dataset):
    """BeerDataset."""

    def __init__(self, data_path, aspect='all', transform=None):
        
        # Compute size of the data set      
        self.aspect = aspect
        
        with gzip.open(data_path, 'rt') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        
        target, sep, review = self.lines[idx].partition('\t')

        target = tuple(float(t) for t in target.split())
        
        if self.aspect == 'aspect 1':
            target = target[0]
        elif self.aspect == 'aspect 2':
            target = target[1]
        elif self.aspect == 'aspect 3':
            target = target[2]
        else:
            target = target[:3]
            
        sample = {'review': review, 'target': target}

        return sample