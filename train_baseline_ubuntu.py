import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from dpp_nets.utils.ubuntu_io import compute_ap, create_ubuntu
from dpp_nets.layers.layers import DeepSetBaseline

import numpy as np

parser = argparse.ArgumentParser(description='Baseline (Deep Sets) Trainer')

parser.add_argument('--remote', type=int,
                    help='training locally or on cluster?', required=True)

parser.add_argument('--data_path_local', type=str, default='/Users/Max/data/askubuntu',
                    help='where is the data folder locally?')

parser.add_argument('--data_path_remote', type=str, default='/cluster/home/paulusm/data/askubuntu',
                    help='where is the data folder?')

parser.add_argument('--ckp_path_local', type=str, default='/Users/Max/checkpoints/askubuntu',
                    help='where is the data folder locally?')

parser.add_argument('--ckp_path_remote', type=str, default='/cluster/home/paulusm/checkpoints/askubuntu',
                    help='where is the data folder?')

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N', help='mini-batch size (default: 50)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
#parser.add_argument('--lr-k', '--learning-rate-k', default=0.1, type=float,
#                    metavar='LRk', help='initial learning rate for kernel net')
#parser.add_argument('--lr-p', '--learning-rate-p', default=0.1, type=float,
#                    metavar='LRp', help='initial learning rate for pred net')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate for baseline')
#parser.add_argument('--reg', type=float, required=True,
#                    metavar='reg', help='regularization constant')
#parser.add_argument('--reg-mean', type=float, required=True,
#                    metavar='reg_mean', help='regularization_mean')

def main():

    global args, lowest_loss

    args = parser.parse_args()
    lowest_map = 0 # arbitrary high number as upper bound for loss

    ### Load data
    if args.remote:
        # print('training remotely')
        train_path = os.path.join(args.data_path_remote, str.join(".",['reviews', args.aspect, 'train_random.txt']))
        val_path   = os.path.join(args.data_path_remote, str.join(".",['reviews', args.aspect, 'dev.txt']))
        embd_path  = os.path.join(args.data_path_remote, 'vectors_pruned.200.txt.gz')
        database_path = os.path.join(args.data_path_remote, 'text_tokenized.txt.gz')

    else:
        # print('training locally')
        train_path = os.path.join(args.data_path_local, str.join(".",['reviews', args.aspect, 'train_random.txt']))
        val_path   = os.path.join(args.data_path_local, str.join(".",['reviews', args.aspect, 'dev.txt']))
        embd_path  = os.path.join(args.data_path_local, 'review+wiki.filtered.200.txt.gz')
        database_path = os.path.join(args.data_path_local, 'text_tokenized.txt.gz')

    embd, train_set, val_set = create_ubuntu(embd_path, database_path, train_path, val_path)
    embd.requires_grad = False

    torch.manual_seed(0)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size)
    print("loader defined")

    ### Build model
    # Network parameters
    embd_dim = embd.weight.size(1)
    hidden_dim = 500
    enc_dim = 200
    target_dim = 200

    # Model
    torch.manual_seed(0)
    net = DeepSetBaseline(embd_dim, hidden_dim, enc_dim, target_dim)
    activation = nn.Sigmoid()
    model = nn.Sequential(embd, net, activation)
    print("created model")

    ### Set-up training
    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("set up optimizer")

    ### Loop
    torch.manual_seed(0)
    print("started loop")
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer)        
        loss, MAP = validate(val_loader, model, criterion)
        
        log(epoch, loss, MAP)
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


def train(loader, model, criterion, optimizer):

    model.train()

    for t, (qs, target) in enumerate(loader):
        
        q1 = Variable(qs[:,0,:])
        q2 = Variable(qs[:,1,:])
        target = Variable(target)

        pred1 = model(q1)
        pred2 = model(q2)
        loss = criterion(pred1, pred2, target)

        optimizer.zero_grad()
        loss.backward() # I'm currently not using the Hinge loss. 
        optimizer.step()

        print("trained one batch")

def validate(loader, model, criterion):

    model.eval()

    MAP = 0.0

    for t, (qs, target) in enumerate(loader, 1):

        q0 = Variable(qs[:,0,:])
        pred0 = model(q0)

        scores = []
        losses = []

        for i in range(1, len(qs.size(1))):

            q = Variable(qs[:,i,:], volatile=True)
            pred = model(q)
            target = Variable([target[i]], volatile=True)
            cos_dis = 1 - ((pred * pred0) / (pred0.pow(2).sum().sqrt() * pred.pow(2).sum().sqrt()))
            scores.append(cos_dis.data[0])
            loss = criterion(pred0, pred1, target)
            losses.append(loss.data[0])

        scores = np.array(scores)
        target = target.numpy()
        ap = compute_ap(scores, append)
        delta = ap - MAP
        MAP += (delta / t)

        average_loss = losses.sum() / len(loss)

        print("validated one batch")

    return average_loss, MAP

def log(epoch, loss, MAP):
    string = str.join(" | ", ['Epoch: %d' % (epoch), 'Validation Loss: %.5f' % (loss), 'Validation MAP: %.5f' % (MAP)])

    if args.remote:
        destination = os.path.join(args.ckp_path_remote, args.aspect + str(args.lr) + 'ubuntu_baseline_log.txt')
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + str(args.lr) + 'ubuntu_baseline_log.txt')

    with open(destination, 'a') as log:
        log.write(string + '\n')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by factor 0.1 for every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='ubuntu_baseline_checkpoint.pth.tar'):
    """
    State is a dictionary that cotains valuable information to be saved.
    """
    if args.remote:
        destination = os.path.join(args.ckp_path_remote, args.aspect + str(args.lr) + filename)
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + str(args.lr) + filename)
    
    torch.save(state, destination)
    if is_best:
        if args.remote:
            best_destination = os.path.join(args.ckp_path_remote, args.aspect + str(args.lr) + 'ubuntu_baseline_model_best.pth.tar')
        else:
            best_destination = os.path.join(args.ckp_path_local, args.aspect + str(args.lr) + 'ubuntu_baseline_model_best.pth.tar')
        
        shutil.copyfile(destination, best_destination)

if __name__ == '__main__':
    main()
