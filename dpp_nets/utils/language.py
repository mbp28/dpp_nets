import numpy as np
import gzip
import string
import spacy
import torch 
import torch.nn as nn
import nltk 

from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset
from dpp_nets.my_torch.utilities import pad_tensor
from torch.autograd import Variable

class Vocabulary:
    
    def __init__(self):
        
        # Basic Indexing
        self.word2index = {}
        self.index2word = {}
        
        # Keeping track of vocabulary
        self.vocab_size = 0 
        self.word2count = {}
        
        # Vector Dictionaries
        self.pretrained = {}
        self.random = {}
        self.word2vec = {}
        self.index2vec = {}

        # Set of Stop Words
        self.stop_words = set()
        
        self.Embedding = None
        self.EmbeddingBag = None

        self.cuda = False
    
    def setStops(self):
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        make_stops = set(string.punctuation + '\n' + '\t' + '...')
        unmake_stops = set(('no', 'not'))

        self.stop_words = self.stop_words.union(make_stops)
        self.stop_words = self.stop_words.difference(unmake_stops)      
        
    def loadPretrained(self, embd_path):
        
        self.pretrained = {}
        with gzip.open(embd_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if line:
                    word, *embd = line.split()
                    vec = torch.FloatTensor([float(dim) for dim in embd])            
                    self.pretrained[word]  = vec
                    
    def loadCorpus(self, word_path):
        
        with gzip.open(word_path, 'rt') as f:

            for line in f:
                _, review = line.split('\D')
                review = tuple(tuple(chunk.split('\W')) for chunk in review.split('\T'))

                for words in review:
                    self.addWords(words)
            
    def addWords(self, words):
        """
        words: seq containing variable no of words
        """
        for word in words:
            self.addWord(word)

    def addWord(self, word):

        if word not in self.word2index:
            
            # Keeping track of vocabulary
            self.vocab_size += 1
            self.word2count[word] = 1
            
            # Basic Indexing
            self.word2index[word] = self.vocab_size
            self.index2word[self.vocab_size] = word
            
            # Add word vector
            if word in self.pretrained:
                vec = self.pretrained[word]
                self.word2vec[word] = vec
                self.index2vec[self.vocab_size] = vec
                
            else:
                vec = torch.randn(200)
                self.random[word] = vec
                self.word2vec[word] = vec
                self.index2vec[self.vocab_size] = vec
        else:
            self.word2count[word] += 1
            
    def updateEmbedding(self):
        
        vocab_size = len(self.index2vec) + 1
        EMBD_DIM = 200
        
        self.Embedding = nn.Embedding(vocab_size, EMBD_DIM, padding_idx=0)
        self.EmbeddingBag = nn.EmbeddingBag(vocab_size, EMBD_DIM)
        embd_matrix = torch.zeros(vocab_size, EMBD_DIM)
        
        for ix, vec in self.index2vec.items():
            embd_matrix[ix] = vec
        
        embd_dict = OrderedDict([('weight', embd_matrix)])
        self.Embedding.load_state_dict(embd_dict)
        self.EmbeddingBag.load_state_dict(embd_dict)
    
    def checkWord(self, word, min_count):

        if word not in self.stop_words and word in self.word2index and self.word2index[word] > min_count:
            return word
            
    def filterReview(self, review):
        """
        review should be like our data set
        """
        f_review = []
        seen = set()
        
        for tup in review:
            f_tuple = []
            
            for word in tup:
                word = self.checkWord(word, 10)
                if word:
                    f_tuple.append(word)
            
            f_tuple = tuple(f_tuple)    
            
            if f_tuple and f_tuple not in seen:
                seen.add(f_tuple)
                f_review.append(f_tuple)
                
        return f_review

    def createDicts(self, defdict):
        """
        sentences is a list of tuples that contain 
        0: spacy sentence
        1: set of label(s)
        """
        newdict = defaultdict(set)
        reverse_dict = defaultdict(list)

        for tup, label in defdict.items():
            
            f_tuple = []

            for word in tup:
                word = self.checkWord(word, 10)
                if word:
                    f_tuple.append(word)
            
            f_tuple = tuple(f_tuple)    
            
            if f_tuple: 
                newdict[f_tuple].update(*label)
                reverse_dict[f_tuple].append(tup)
                
        return newdict, reverse_dict

    def mapIndicesBatch(self, reviews):
        
        f_review = []
        offset = []
        i = 0

        for review in reviews:
            seen = set()
            
            for tup in review: 
                f_tuple = []
                
                for word in tup:
                    word = self.checkWord(word, 10)
                    if word:
                        f_tuple.append(word)

                f_tuple = tuple(f_tuple)    

                if f_tuple and f_tuple not in seen:
                    seen.add(f_tuple)
                    f_review.extend([self.word2index[word] for word in f_tuple])
                    offset.append(i)
                    i += len(f_tuple)
            
        f_review, offset = torch.LongTensor(f_review), torch.LongTensor(offset)   
        return f_review, offset
    
    def mapIndices(self, review):
        
        f_review = []
        offset = []
        seen = set()
        i = 0

        for tup in review:
            f_tuple = []

            for word in tup:
                word = self.checkWord(word, 10)
                if word:
                    f_tuple.append(word)

            f_tuple = tuple(f_tuple)    

            if f_tuple and f_tuple not in seen:
                seen.add(f_tuple)
                f_review.extend([self.word2index[word] for word in f_tuple])
                offset.append(i)
                i += len(f_tuple)

        f_review, offset = torch.LongTensor(f_review), torch.LongTensor(offset)   
        return f_review, offset
    
    def returnEmbds(self, review):
        
        f_review = []
        offset = []
        seen = set()
        i = 0

        for tup in review:
            f_tuple = []

            for word in tup:
                word = self.checkWord(word, 10)
                if word:
                    f_tuple.append(word)

            f_tuple = tuple(f_tuple)    

            if f_tuple and f_tuple not in seen:
                seen.add(f_tuple)
                f_review.extend([self.word2index[word] for word in f_tuple])
                offset.append(i)
                i += len(f_tuple)

        
        if self.cuda: 
            f_review, offset = Variable(torch.cuda.LongTensor(f_review)), Variable(torch.cuda.LongTensor(offset))

        else:
            f_review, offset = Variable(torch.LongTensor(f_review)), Variable(torch.LongTensor(offset))

        embd = self.EmbeddingBag(f_review, offset)

        return embd

    def setCuda(self, Yes=True):

        if Yes:
            self.cuda = True
            self.EmbeddingBag.cuda()
        else:
            self.cuda = False
            self.EmbeddingBag.float()


class BeerDataset(Dataset):
    """BeerDataset."""

    def __init__(self, data_path, aspect='all'):
        
        # Compute size of the data set      
        self.aspect = aspect
        
        with gzip.open(data_path, 'rt') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        
        # Decode
        target, review = self.lines[idx].split('\D')
        
        # Target
        target = torch.FloatTensor([float(t) for t in target.split()[:3]])
        
        # Review
        review = tuple(tuple(chunk.split('\W')) for chunk in review.split('\T'))
        #ixs, offset = self.vocab.mapIndices(review)
        
        #sample = {'ixs': ixs, 'offset': offset, 'target': target}
        sample = {'review': review, 'target': target}
        
        return sample

def custom_collate(vocab, cuda=False):

    def collate(batch):

        # Count sizes - no tensor operations
        max_no_chunks = 0
        
        for d in batch:
            max_no_chunks = max(max_no_chunks, len(vocab.filterReview(d['review'])))
        
        # Map to Embeddings
        reps = []
        for d in batch:
            rep = vocab.returnEmbds(d['review'])
            if cuda:
                rep = torch.cat([rep, Variable(torch.zeros(max_no_chunks + 1 - rep.size(0), rep.size(1))).cuda()], dim=0)
            else:
                rep = torch.cat([rep, Variable(torch.zeros(max_no_chunks + 1 - rep.size(0), rep.size(1)))], dim=0)
            reps.append(rep)
        
        data_tensor = torch.stack(reps) 
        
        # Create target vector
        # target_tensor = Variable(torch.stack([d['target'] for d in batch]))
        if cuda: 
            target_tensor = Variable(torch.stack([d['target'] for d in batch]).cuda())
        else:
            target_tensor = Variable(torch.stack([d['target'] for d in batch]))
        
        return data_tensor, target_tensor

    return collate
