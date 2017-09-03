import argparse
import os
import shutil
import nltk

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from dpp_nets.utils.language import Vocabulary, BeerDataset, custom_collate
from dpp_nets.layers.layers import ChunkTrainer


parser = argparse.ArgumentParser(description='marginal_chunk Krause Trainer')
parser.add_argument('-a', '--aspect', type=str, choices=['aspect1', 'aspect2', 'aspect3', 'all', 'short'],
                    help='what is the target?', required=True)
parser.add_argument('-m', '--mode', type=str, choices=['words', 'chunks', 'sents'],
                    help='what is the mode?', required=True)
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                    metavar='', help='initial learning rate')
parser.add_argument('--reg', type=float, required=True,
                    metavar='reg', help='regularization constant')
parser.add_argument('--reg_mean', type=float, required=True,
                    metavar='reg_mean', help='regularization_mean')
parser.add_argument('-r', '--remote', type=int,
                    help='training locally or on cluster?', required=True)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

def main():

    global vocab, args

    lowest_loss = 100 # arbitrary high number as upper bound for loss

    # Check for GPU
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    # Set Seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set-up data
    if args.remote:
        train_path = '/home/paulusm/data/beer_reviews/' + 'reviews.' + args.aspect + '.train.' + args.mode + '.txt.gz'
        val_path = '/home/paulusm/data/beer_reviews/' + 'reviews.' + args.aspect + '.heldout.' + args.mode + '.txt.gz'
        embd_path = '/home/paulusm/data/beer_reviews/' + 'review+wiki.filtered.200.txt.gz'
        word_path = '/home/paulusm/data/beer_reviews/' + 'reviews.' + args.aspect + '.train.' + 'words.txt.gz'
    else:
        train_path = '/Users/Max/data/beer_reviews/' + 'reviews.' + args.aspect + '.train.' + args.mode + '.txt.gz'
        val_path = '/Users/Max/data/beer_reviews/' + 'reviews.' + args.aspect + '.heldout.' + args.mode + '.txt.gz'
        embd_path = '/Users/Max/data/beer_reviews/' + 'review+wiki.filtered.200.txt.gz'
        word_path = '/Users/Max/data/beer_reviews/' + 'reviews.' + args.aspect + '.train.' + 'words.txt.gz'

    # Set-up vocabulary
    vocab = Vocabulary()
    vocab.loadPretrained(embd_path)
    vocab.setStops()
    vocab.loadCorpus(word_path)
    vocab.updateEmbedding()
    vocab.setCuda(args.cuda)

    # Set up datasets and -loader
    train_set = BeerDataset(train_path, vocab)
    val_set = BeerDataset(val_path, vocab)
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    my_collate = custom_collate(vocab, args.cuda)
    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=my_collate, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(train_set, collate_fn=my_collate, batch_size=args.batch_size, **kwargs)

    # Network parameters
    EMBD_DIM = 200
    KERNEL_DIM = 200
    HIDDEN_DIM = 500
    ENC_DIM = 200
    TARGET_DIM = 3 if args.aspect in set(['all', 'short']) else 1

    # Conventional trainer
    trainer = ChunkTrainer(EMBD_DIM, HIDDEN_DIM, KERNEL_DIM, ENC_DIM, TARGET_DIM)
    trainer.activation = nn.Sigmoid()
    trainer.reg = args.reg
    trainer.reg_mean = args.reg_mean

    print("created trainer")

    # Set-up optimizer
    params = [{'params': vocab.EmbeddingBag.parameters()}, {'params': trainer.parameters()}]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    ### Loop
    for epoch in range(args.epochs):

        for t, batch in enumerate(train_loader):
            pass


def train(loader, trainer, optimizer):

    trainer.train()

    for t, batch in enumerate(loader):

        reviews, target = batch

        loss  = trainer(reviews, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("trained one batch")

def validate(loader, trainer):

    trainer.eval()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_reg_loss = 0.0

    for i, batch in enumerate(loader, 1):

        review, target = batch

        trainer(review, target)

        loss = trainer.loss.data[0]
        pred_loss = trainer.pred_loss.data[0]
        reg_loss = trainer.reg_loss.data[0]

        delta = loss - total_loss
        total_loss += (delta / i)
        delta = pred_loss - total_pred_loss 
        total_pred_loss += (delta / i)
        delta = reg_loss - total_reg_loss
        total_reg_loss += (delta / i)

        # print("validated one batch")

    return total_loss, total_pred_loss, total_reg_loss

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by factor 0.1 for every 10 epochs"""
    if not ((epoch + 1) % 10):
        factor = 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor

def log(epoch, loss, pred_loss, reg_loss):

    string = str.join(" | ", ['Epoch: %d' % (epoch), 'V Loss: %.5f' % (loss), 
                              'V Pred Loss: %.5f' % (pred_loss), 'V Reg Loss: %.5f' % (reg_loss)])

    if args.remote:
        destination = '/home/paulusm/checkpoints/beer_reviews/' + str(args.aspect) + str(args.mode) + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 'lr' + str(args.lr) +  'log_marginal.txt'
    else:
        destination = '/Users/Max/checkpoints/beer_reviews/' + str(args.aspect) + str(args.mode) + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 'lr' + str(args.lr) +  'log_marginal.txt'


    with open(destination, 'a') as log:
        log.write(string + '\n')

def save_checkpoint(state, is_best, filename='marginal_chunk_checkpoint.pth.tar'):
    """
    State is a dictionary that cotains valuable information to be saved.
    """
    if args.remote:
        destination = '/home/paulusm/checkpoints/beer_reviews/' + str(args.aspect) + str(args.mode) + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 'lr' + str(args.lr) + 'marginal_ckp.pth.tar'
    else:
        destination = '/Users/Max/checkpoints/beer_reviews/' + str(args.aspect) + str(args.mode) + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 'lr' + str(args.lr) +  'marginal_ckp.pth.tar'

    torch.save(state, destination)

    if is_best:
        if args.remote:
            best_destination = '/home/paulusm/checkpoints/beer_reviews/' + str(args.aspect) + str(args.mode) + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 'lr' + str(args.lr) + 'marginal_best_ckp.pth.tar'
        else:
            best_destination = '/Users/Max/checkpoints/beer_reviews/' + str(args.aspect) + str(args.mode) + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 'lr' + str(args.lr) + 'marginal_best_ckp.pth.tar'

        shutil.copyfile(destination, best_destination)

if __name__ == '__main__':
    main()
    


