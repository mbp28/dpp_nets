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

import json
import gzip
import random
import ast
from collections import defaultdict
from collections import namedtuple
import spacy
from dpp_nets.dpp.map import computeMAP


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
        make_stops = set(string.punctuation).union(set(['\n','\t','...','beer','glass'])) 
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

def simple_collate(batch):
    return batch

def custom_collate(batch, vocab, cuda=False):

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

def custom_collate_reinforce(batch, vocab, alpha_iter, cuda=False):

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
        target_tensor = Variable(torch.stack([d['target'] for d in batch for _ in range(alpha_iter)]).cuda())
    else:
        target_tensor = Variable(torch.stack([d['target'] for d in batch for _ in range(alpha_iter)]))
    
    return data_tensor, target_tensor



class EvalSet():
    
    def __init__(self, rat_path, vocab, mode='words'):
        
        nlp = spacy.load('en')
        self.reviews = []
        self.targets = []
        self.labelled_docs = []
        self.vocab = vocab
        self.words  = []
        self.chunks = []
        self.sents  = []
                
        DictCollect = namedtuple('DictCollect', ['all', 'clean', 'rev'])
        
        with open(rat_path) as f:
            for line in f:
                review = json.loads(line)
                self.reviews.append(review)
                
        for review in self.reviews:
            target = torch.FloatTensor([float(t) for t in review['y'][:3]])
            self.targets.append(target)
                        
        for review in self.reviews:
            doc = nlp(ast.literal_eval(review['raw'])['review/text'])
            sens_with_labels = self.__gatherLabels(review)
            labelled_doc = self.__curateDoc(doc, sens_with_labels)
            self.labelled_docs.append(labelled_doc)
        
        for labelled_doc in self.labelled_docs:
            words, chunks, sents = defaultdict(set), defaultdict(set), defaultdict(set)            
            for review, label in labelled_doc:
                for word in review:
                    words[tuple([word.text])].update([*label])
                    chunks[tuple([w.text for w in word.subtree if w.text != '\n' and w.text != '\t'])].update([*label])
                sents[tuple([word.text for word in review])].update([*label])
            self.words.append(words)
            self.chunks.append(chunks)
            self.sents.append(sents)
            
        self.words = [self.__complementDict(defdict) for defdict in self.words]
        self.chunks = [self.__complementDict(defdict) for defdict in self.chunks]
        self.sents = [self.__complementDict(defdict) for defdict in self.sents]
        
    def __complementDict(self, def_dict):
    
            DictCollect = namedtuple('DictCollect', ['all', 'clean', 'rev'])
            cleanDict, revDict = self.vocab.createDicts(def_dict)
            dictcollect = DictCollect(def_dict, cleanDict, revDict)
            
            return dictcollect
        
    def __curateDoc(self, doc, sens_with_labels):
    
        labelled_doc = [(sen, set()) for sen in doc.sents]
        
        for words, label in sens_with_labels:
            scores = ([self.__computeScore(sen, words) for sen, _ in labelled_doc])
            labelled_doc[scores.index(max(scores))][1].add(label)
        
        return labelled_doc    

    def __computeScore(self, sen, words):
        
        sen = set([str(token).lower() for token in sen])
        n = len(sen) 
        s = len(sen & words)
        score = s / n
        return score
    
    def __gatherLabels(self, review, labels=['0','1','2']):
        sens_with_labels = []
        all_words = review['x']
        for label in labels:
            for s, e in review[label]:
                sens_with_labels.append((set(all_words[s:e]), label))
        return sens_with_labels
    
    def computeLoss(self, trainer, mode):
    
        trainer.eval()
        
        total_loss = 0.0
        total_pred_loss = 0.0
        total_reg_loss = 0.0
        
        if mode == 'words':
            reviews = self.words
        elif mode == 'chunks':
            reviews = self.chunks
        elif mode == 'sents':
            reviews = self.sents
        else:
            raise
            
        for i, (review, target) in enumerate(zip(reviews, self.targets), 1):


            
            trainer(self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0), Variable(target))
            loss = trainer.loss.data[0]
            pred_loss = trainer.pred_loss.data[0]
            reg_loss = trainer.reg_loss.data[0]

            delta = loss - total_loss
            total_loss += (delta / i)
            delta = pred_loss - total_pred_loss 
            total_pred_loss += (delta / i)
            delta = reg_loss - total_reg_loss
            total_reg_loss += (delta / i)
        
        return total_loss, total_pred_loss, total_reg_loss

    def computeMAPPredLoss(self, trainer, mode):
    
        trainer.eval()
        
        total_pred_loss = 0.0
        total_extract = 0.0
        
        if mode == 'words':
            reviews = self.words
        elif mode == 'chunks':
            reviews = self.chunks
        elif mode == 'sents':
            reviews = self.sents
        else:
            raise
            
        for i, (review, target) in enumerate(zip(reviews, self.targets), 1):

            words = self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0)

            kernel, _ = trainer.kernel_net(words)
            L = (kernel.data.mm(kernel.data.t())).numpy()
            return_ixs = computeMAP(L)
            fwords = words[:, return_ixs, :].squeeze(0)
            
            trainer.pred_net.s_ix = [0]
            trainer.pred_net.e_ix = [len(return_ixs)]

            trainer.pred = trainer.pred_net(fwords)

            if trainer.activation:
                trainer.pred = trainer.activation(trainer.pred)

            trainer.pred_loss = trainer.criterion(trainer.pred, Variable(target))

            pred_loss = trainer.pred_loss.data[0]

            delta = pred_loss - total_pred_loss 
            total_pred_loss += (delta / i)

            # Compute Extract %
            labelled_doc = self.labelled_docs[i - 1]
            rats = [list(review.rev.values())[ix] for ix in return_ixs]
            lens = [max([len(l) for l in rat]) for rat in rats]
            review_lens = [len(sen) for sen, _ in labelled_doc]
            extract = sum(lens) / sum(review_lens)

            delta = extract - total_extract
            total_extract += (delta / i)

        
        return total_pred_loss, total_extract

    def evaluatePrecision(self, trainer, mode):
        
        trainer.eval()
        mean_prec = 0.0
        text_extract = 0.0
        
        if mode == 'words':
            reviews = self.words
        elif mode == 'chunks':
            reviews = self.chunks
        elif mode == 'sents':
            reviews = self.sents
        else:
            raise
        
        for i, (review, labelled_doc) in enumerate(zip(reviews, self.labelled_docs), 1):
            
            kernel, _ = trainer.kernel_net(self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0))
            L = (kernel.data.mm(kernel.data.t())).numpy()
            return_ixs = computeMAP(L)

            labels = [list(review.clean.values())[ix] for ix in return_ixs]
            hits = [label for label in labels if label]
            prec = len(hits) / len(labels)
            
            rats = [list(review.rev.values())[ix] for ix in return_ixs]
            lens = [max([len(l) for l in rat]) for rat in rats]
            review_lens = [len(sen) for sen, _ in labelled_doc]
            extract = sum(lens) / sum(review_lens)
            
            delta = prec - mean_prec
            mean_prec += (delta / i)
            delta = extract - text_extract
            text_extract += (delta / i)
            
        return mean_prec, text_extract
    
    def sample(self, trainer, mode, ix=None):

        trainer.eval()
        
        if not ix:
            ix = random.randint(0,1000)
        print('index is:', ix)
        if mode == 'words':
            reviews = self.words
        elif mode == 'chunks':
            reviews = self.chunks
        elif mode == 'sents':
            reviews == self.sents
        else:
            raise

        review = reviews[ix]
        labelled_doc = self.labelled_docs[ix]
        target = self.targets[ix]

        # Compute MAP
        kernel, _ = trainer.kernel_net(self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0))
        L = (kernel.data.mm(kernel.data.t())).numpy()
        return_ixs = computeMAP(L)

        for i in return_ixs:
            txt = list(review.clean.keys())[i]
            label = list(review.clean.values())[i]
            rat = review.rev[txt]
            print(txt, label, rat)

        # Compute precision
        labels = [list(review.clean.values())[i] for i in return_ixs]
        hits = [label for label in labels if label]
        prec = len(hits) / len(labels)
        print('Precision is:', prec)
        
        # Compute Extract %
        rats = [list(review.rev.values())[ix] for ix in return_ixs]
        lens = [max([len(l) for l in rat]) for rat in rats]
        review_lens = [len(sen) for sen, _ in labelled_doc]
        extract = sum(lens) / sum(review_lens)
        print('Extraction Percentage is:', extract)
        
        print(self.labelled_docs[ix])

        # Prediction and target
        trainer(self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0), Variable(target))
        pred = trainer.pred.data
        loss = trainer.loss.data[0]
        pred_loss = trainer.pred_loss.data[0]
        reg_loss = trainer.reg_loss.data[0]

        print(pred, target)
        print('Loss:', loss, 'Pred Loss', pred_loss, 'Reg Loss', reg_loss)

    def computeMarginals(self, trainer, mode, ix=None):

            if not ix:
                ix = random.randint(0,1000)
            print('index is:', ix)
            if mode == 'words':
                reviews = self.words
            elif mode == 'chunks':
                reviews = self.chunks
            elif mode == 'sents':
                reviews == self.sents
            else:
                raise

            review = reviews[ix]
            labelled_doc = self.labelled_docs[ix]

            kernel, _ = trainer.kernel_net(self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0))
            L = (kernel.data.mm(kernel.data.t())).numpy()
            K = L.dot(np.linalg.inv(L + np.eye(L.shape[0])))
            ranking = list(np.argsort(-np.diag(K)))

            for i, rank in enumerate(ranking):
                print(i, np.diag(K)[rank], list(review.clean.keys())[rank])

    def evaluateBaseline(self, trainer, mode):
    
        trainer.eval()
        activation = nn.Sigmoid()
        criterion = nn.MSELoss()
        
        total_loss = 0.0
        
        if mode == 'words':
            reviews = self.words
        elif mode == 'chunks':
            reviews = self.chunks
        elif mode == 'sents':
            reviews = self.sents
        else:
            raise
            
        for i, (review, target) in enumerate(zip(reviews, self.targets), 1):
            
            pred = trainer(self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0))

            loss = criterion(activation(pred),Variable(target, volatile=True)).data[0]

            delta = loss - total_loss
            total_loss += (delta / i)
        
        return total_loss

    def computeMUEPredLoss(self, trainer, mode, n_runs):
    
        trainer.eval()
        trainer.alpha_iter = n_runs
        trainer.sampler.alpha_iter = n_runs

        total_pred_loss1 = 0.0
        total_pred_loss2 = 0.0
        
        if mode == 'words':
            reviews = self.words
        elif mode == 'chunks':
            reviews = self.chunks
        elif mode == 'sents':
            reviews = self.sents
        else:
            raise
            
        for i, (review, t) in enumerate(zip(reviews, self.targets), 1):

            words = self.vocab.returnEmbds(review.clean.keys()).unsqueeze(0)
            target = Variable(torch.stack([t for _ in range(n_runs)]))
            trainer(words, target)

            # Now pick a prediction using Maximum Utility Expectation Principle
            pred1 = trainer.pred.mean(0)
            _, pred_ix = torch.min((trainer.pred - pred1).pow(2).mean(1).data,0)
            pred2 = trainer.pred[pred_ix]

            # Evaluate that prediction (pred1)
            trainer.pred_loss = trainer.criterion(pred1, Variable(t))
            pred_loss = trainer.pred_loss.data[0]
            delta = pred_loss - total_pred_loss1
            total_pred_loss1 += (delta / i)

            # Evaluate that prediction (pred2)
            trainer.pred_loss = trainer.criterion(pred2, Variable(t))
            pred_loss = trainer.pred_loss.data[0]
            delta = pred_loss - total_pred_loss2
            total_pred_loss2 += (delta / i)
            
        trainer.alpha_iter = 1
        trainer.sampler.alpha_iter = 1

        return total_pred_loss1, total_pred_loss2


