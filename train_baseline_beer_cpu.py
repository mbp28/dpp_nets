import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from dpp_nets.utils.io import make_embd, make_tensor_dataset
from dpp_nets.layers.layers import DeepSetBaseline

parser = argparse.ArgumentParser(description='Baseline (Deep Sets) Trainer')
parser.add_argument('-a', '--aspect', type=str, choices=['aspect1', 'aspect2', 'aspect3', 'all'],
                    help='what is the target?', required=True)
parser.add_argument('--remote', type=int,
                    help='training locally or on cluster?', required=True)

parser.add_argument('--data_path_local', type=str, default='/Users/Max/data/beer_reviews',
                    help='where is the data folder locally?')
parser.add_argument('--data_path_remote', type=str, default='/cluster/home/paulusm/data/beer_reviews',
                    help='where is the data folder?')

parser.add_argument('--ckp_path_local', type=str, default='/Users/Max/checkpoints/beer_reviews',
                    help='where is the data folder locally?')

parser.add_argument('--ckp_path_remote', type=str, default='/cluster/home/paulusm/checkpoints/beer_reviews',
                    help='where is the data folder?')

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
#parser.add_argument('--lr-k', '--learning-rate-k', default=0.1, type=float,
#                    metavar='LRk', help='initial learning rate for kernel net')
#parser.add_argument('--lr-p', '--learning-rate-p', default=0.1, type=float,
#                    metavar='LRp', help='initial learning rate for pred net')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for baseline')
#parser.add_argument('--reg', type=float, required=True,
#                    metavar='reg', help='regularization constant')
#parser.add_argument('--reg-mean', type=float, required=True,
#                    metavar='reg_mean', help='regularization_mean')

def main():

    global args, best_prec1

    args = parser.parse_args()
    lowest_loss = 100 # arbitrary high number as upper bound for loss

    ### Load data
    if args.remote:
        # print('training remotely')
        train_path = os.path.join(args.data_path_remote, str.join(".",['reviews', args.aspect, 'train.txt.gz']))
        val_path   = os.path.join(args.data_path_remote, str.join(".",['reviews', args.aspect, 'heldout.txt.gz']))
        embd_path = os.path.join(args.data_path_remote, 'review+wiki.filtered.200.txt.gz')

    else:
        # print('training locally')
        train_path = os.path.join(args.data_path_local, str.join(".",['reviews', args.aspect, 'train.txt.gz']))
        val_path   = os.path.join(args.data_path_local, str.join(".",['reviews', args.aspect, 'heldout.txt.gz']))
        embd_path = os.path.join(args.data_path_local, 'review+wiki.filtered.200.txt.gz')

    embd, word_to_ix = make_embd(embd_path)
    train_set = make_tensor_dataset(train_path, word_to_ix)
    val_set = make_tensor_dataset(val_path, word_to_ix)
    print("loaded data")

    torch.manual_seed(0)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size)
    print("loader defined")

    ### Build model
    # Network parameters
    embd_dim = embd.weight.size(1)
    hidden_dim = 500
    enc_dim = 200
    if args.aspect == 'all':
        target_dim = 3
    else: 
        target_dim = 1

    # Model
    torch.manual_seed(0)
    net = DeepSetBaseline(embd_dim, hidden_dim, enc_dim, target_dim)
    activation = nn.Sigmoid()
    model = nn.Sequential(embd, net, activation)
    print("created model")

    ### Set-up training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("set up optimizer")

    ### Loop
    torch.manual_seed(0)
    print("started loop")
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch)

        #train(train_loader, model, criterion, optimizer, args.aspect)
        #loss = 1
        
        loss = validate(val_loader, model, criterion, args.aspect)
        
        log(epoch, loss)
        print("logged")

        is_best = loss < lowest_loss
        lowest_loss = min(loss, lowest_loss)    
        save = {'epoch:': epoch + 1, 
                'model': 'Deep Set Baseline',
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict()} 

        save_checkpoint(save, is_best)
        print("saved a checkpoint")

    print('*'*20, 'SUCCESS','*'*20)


def train(loader, model, criterion, optimizer, aspect):

    for t, (review, target) in enumerate(loader):
        review = Variable(review)

        if args.aspect == 'all':
            target = Variable(target[:,:3])
        else:
            target = Variable(target[:,int(args.aspect[-1])])

        pred = model(review)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("trained one batch")

def validate(loader, model, criterion, aspect):

    total_loss = 0.0

    for i, (review, target) in enumerate(loader, 1):

        review = Variable(review, volatile=True)

        if args.aspect == 'all':
            target = Variable(target[:,:3], volatile=True)
        else:
            target = Variable(target[:,int(args.aspect[-1])], volatile=True)

        pred = model(review)
        loss = criterion(pred, target)
        
        delta = loss.data[0] - total_loss
        total_loss += (delta / i)
        
        print("validated one batch")

    return total_loss

def log(epoch, loss):
    string = str.join(" | ", ['Epoch: %d' % (epoch), 'Validation Loss: %.5f' % (loss)])

    if args.remote:
        destination = os.path.join(args.ckp_path_remote, args.aspect + 'DeepSetBaseline_log.txt')
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + 'DeepSetBaseline_log.txt')

    with open(destination, 'a') as log:
        log.write(string + '\n')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = args.lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='baseline_checkpoint.pth.tar'):
    """
    State is a dictionary that cotains valuable information to be saved.
    """
    if args.remote:
        destination = os.path.join(args.ckp_path_remote, filename)
    else:
        destination = os.path.join(args.ckp_path_local, filename)
    
    torch.save(state, destination)
    if is_best:
        if args.remote:
            best_destination = os.path.join(args.ckp_path_remote, args.aspect + 'baseline_model_best.pth.tar')
        else:
            best_destination = os.path.join(args.ckp_path_local, args.aspect + 'baseline_model_best.pth.tar')
        
        shutil.copyfile(destination, best_destination)

if __name__ == '__main__':
    main()
