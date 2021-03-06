import argparse
import os
import shutil
import nltk

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from dpp_nets.utils.language import Vocabulary, BeerDataset, simple_collate, custom_collate
from dpp_nets.layers.baselines import AttentionBaseline, NetBaseline, SetNetBaseline

from collections import defaultdict

parser = argparse.ArgumentParser(description='Baselines')
parser.add_argument('-a', '--aspect', type=str, choices=['aspect1', 'aspect2', 'aspect3', 'all', 'short'],
                    help='what is the target?', required=True)
parser.add_argument('-m', '--mode', type=str, choices=['words', 'chunks', 'sents'],
                    help='what is the mode?', required=True)
parser.add_argument('-b', '--baseline', type=str, choices=['AttentionBaseline', 'Net', 'SetNet'],
                    help='which baseline model?', required=True)
parser.add_argument('-n', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument( '--start-epoch', default=0, type=int,
                    metavar='N', help='start epoch')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='', help='initial learning rate')
parser.add_argument('-r', '--remote', type=int,
                    help='training locally or on cluster?', required=True)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=str, default=None, metavar='S',
                    help='specify model name')
parser.add_argument('-e', '--euler', type=str, default=None, metavar='S',
                    help='specify model name')

def main():

    global vocab, args, activation

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
        word_path = '/home/paulusm/data/beer_reviews/' + 'reviews.' + 'all' + '.train.' + 'words.txt.gz'
        if args.euler:
            train_path = '/cluster' + train_path
            val_path = '/cluster' + val_path
            embd_path = '/cluster' + embd_path
            word_path = '/cluster' + word_path

    else:
        train_path = '/Users/Max/data/beer_reviews/' + 'reviews.' + args.aspect + '.train.' + args.mode + '.txt.gz'
        val_path = '/Users/Max/data/beer_reviews/' + 'reviews.' + args.aspect + '.heldout.' + args.mode + '.txt.gz'
        embd_path = '/Users/Max/data/beer_reviews/' + 'review+wiki.filtered.200.txt.gz'
        word_path = '/Users/Max/data/beer_reviews/' + 'reviews.' + 'all' + '.train.' + 'words.txt.gz'

    # Set-up vocabulary
    vocab = Vocabulary()
    vocab.loadPretrained(embd_path)
    vocab.setStops()
    vocab.loadCorpus(word_path)
    vocab.updateEmbedding()
    vocab.setCuda(args.cuda)

    # Set up datasets and -loader
    train_set = BeerDataset(train_path)
    val_set = BeerDataset(val_path)
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(train_set, collate_fn=simple_collate, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, collate_fn=simple_collate, batch_size=args.batch_size, **kwargs)

    # Network parameters
    EMBD_DIM = 200
    KERNEL_DIM = 200
    HIDDEN_DIM = 500
    ENC_DIM = 200
    TARGET_DIM = 3 if args.aspect in set(['all', 'short']) else 1

    # Conventional trainer
    if args.baseline == 'AttentionBaseline':
        trainer = AttentionBaseline(EMBD_DIM, HIDDEN_DIM, TARGET_DIM)
    if args.baseline == 'Net':
        trainer = NetBaseline(EMBD_DIM, HIDDEN_DIM, TARGET_DIM)
    if args.baseline == 'SetNet':
        trainer = SetNetBaseline(EMBD_DIM, HIDDEN_DIM, ENC_DIM, TARGET_DIM)

    if args.cuda:
        trainer.cuda()
    print("created trainer")

    # Prediction & Loss
    activation = nn.Sigmoid()
    criterion = nn.MSELoss()

    params = [{'params': vocab.EmbeddingBag.parameters()}, {'params': trainer.parameters()}]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    print("set up optimizer")

    # Set-up Savers
    train_loss  = defaultdict(list) 
    val_loss = defaultdict(list)

        # optionally resume from a checkpoint
    if args.resume:
        if args.remote:
            check_path = '/home/paulusm/checkpoints/beer_reviews/baseline/' + args.resume
            if args.euler:
                check_path = '/cluster' + check_path
        else:
            check_path = '/Users/Max/checkpoints/beer_reviews/baseline/' + args.resume

        if os.path.isfile(check_path):
            print("=> loading checkpoint '{}'".format(check_path))
            checkpoint = torch.load(check_path)
            args.start_epoch = len(checkpoint['train_loss'])
            lowest_loss = checkpoint['lowest_loss']
            
            trainer.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            vocab.EmbeddingBag.load_state_dict(checkpoint['embedding'])

            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']

            if not args.mode == checkpoint['mode']:
                print('wrong specs')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    ### Loop
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch)

        loss = train(train_loader, trainer, criterion, optimizer)    
        train_loss[epoch].append(loss)
    
        loss = validate(val_loader, trainer, criterion)
        val_loss[epoch].append(loss)
        
        is_best = loss < lowest_loss
        lowest_loss = min(loss, lowest_loss)    

        checkpoint = {'train_loss': train_loss, 
                      'val_loss': val_loss,
                      'embedding': vocab.EmbeddingBag.state_dict(),
                      'model': trainer.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'model_type': type(trainer),
                      'mode': args.mode,
                      'seed': args.seed,
                      'aspect': args.aspect,
                      'lowest_loss': lowest_loss}

        save_checkpoint(checkpoint, is_best)
        print("saved a checkpoint")

    print('*'*20, 'SUCCESS','*'*20)


def train(loader, trainer, criterion, optimizer):

    trainer.train()

    total_loss = 0.0

    for t, batch in enumerate(loader, 1):

        review, target = custom_collate(batch, vocab, args.cuda)

        pred = activation(trainer(review))
        loss = criterion(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("trained one batch")
        delta = loss.data[0] - total_loss
        total_loss += (delta / t)

    return total_loss

def validate(loader, trainer, criterion):


    trainer.eval()

    total_loss = 0.0

    for i, batch in enumerate(loader, 1):

        review, target = custom_collate(batch, vocab, args.cuda)

        pred = activation(trainer(review))
        loss = criterion(pred, target).data[0]

        delta = loss - total_loss
        total_loss += (delta / i)

    return total_loss


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by factor 0.1 for every 10 epochs"""
    if not ((epoch + 1) % 10):
        factor = 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor


def save_checkpoint(state, is_best, filename='XX.pth.tar'):
    """
    State is a dictionary that cotains valuable information to be saved.
    """
    if args.remote:
        destination = '/home/paulusm/checkpoints/beer_reviews/baseline/' + str(args.aspect) + str(args.mode)  + str(args.baseline)  + str(args.seed) + 'lr' + str(args.lr) + 'ckp.pth.tar'
        if args.euler:
            destination = '/cluster' + destination

    else:
        destination = '/Users/Max/checkpoints/beer_reviews/baseline/' + str(args.aspect) + str(args.mode) + str(args.baseline) + str(args.seed) + 'lr' + str(args.lr) +  'ckp.pth.tar'

    torch.save(state, destination)

    if is_best:
        if args.remote:
            best_destination = '/home/paulusm/checkpoints/beer_reviews/baseline/' + str(args.aspect) + str(args.mode) + str(args.baseline) + str(args.seed) + 'lr' + str(args.lr) + 'best_ckp.pth.tar'
            if args.euler:
                best_destination = '/cluster' + best_destination
        else:
            best_destination = '/Users/Max/checkpoints/beer_reviews/baseline/' + str(args.aspect) + str(args.mode)+ str(args.baseline)  + str(args.seed) + 'lr' + str(args.lr) + 'best_ckp.pth.tar'

        shutil.copyfile(destination, best_destination)

if __name__ == '__main__':
    main()