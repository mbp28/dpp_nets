import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from dpp_nets.utils.language import create_clean_vocabulary, BeerDataset, process_batch_sens

from dpp_nets.layers.layers import ChunkTrainer


parser = argparse.ArgumentParser(description='marginal_sens Krause Trainer')

parser.add_argument('-a', '--aspect', type=str, choices=['aspect1', 'aspect2', 'aspect3', 'all', 'short'],
                    help='what is the target?', required=True)

parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_k', '--learning_rate_k', default=1e-3, type=float,
                    metavar='LRk', help='initial learning rate for kernel net')
parser.add_argument('--lr_p', '--learning_rate_p', default=1e-3, type=float,
                    metavar='LRp', help='initial learning rate for pred net')
parser.add_argument('--reg', type=float, required=True,
                    metavar='reg', help='regularization constant')
parser.add_argument('--reg_mean', type=float, required=True,
                    metavar='reg_mean', help='regularization_mean')

# Train locally or remotely?
parser.add_argument('--remote', type=int,
                    help='training locally or on cluster?', required=True)
# Burnt in Paths..
parser.add_argument('--data_path_local', type=str, default='/Users/Max/data/beer_reviews',
                    help='where is the data folder locally?')
parser.add_argument('--data_path_remote', type=str, default='/cluster/home/paulusm/data/beer_reviews',
                    help='where is the data folder?')
parser.add_argument('--ckp_path_local', type=str, default='/Users/Max/checkpoints/beer_reviews',
                    help='where is the data folder locally?')
parser.add_argument('--ckp_path_remote', type=str, default='/cluster/home/paulusm/checkpoints/beer_reviews',
                    help='where is the data folder?')


def main():

    global args, lowest_loss, nlp, vocab, embd

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

    nlp, vocab, embd = create_clean_vocabulary(embd_path, train_path) # actualy train_path
    embd.weight.requires_grad = False

    train_set = BeerDataset(train_path, aspect=args.aspect) # actualy train_path
    val_set = BeerDataset(val_path, aspect=args.aspect)
    print("loaded data")

    torch.manual_seed(0)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size)
    print("loader defined")

    ### Build model
    # Network parameters
    embd_dim = embd.weight.size(1)
    kernel_dim = 200
    hidden_dim = 500
    enc_dim = 200

    if args.aspect == 'all' or args.aspect == 'short':
        target_dim = 3
    else: 
        target_dim = 1

    # Model
    torch.manual_seed(0)
    trainer = ChunkTrainer(embd_dim, hidden_dim, kernel_dim, enc_dim, target_dim)
    trainer.activation = nn.Sigmoid()
    trainer.reg = args.reg
    trainer.reg_mean = args.reg_mean
    print("created trainer")

    # Set-up Training
    params = [{'params': trainer.kernel_net.parameters(), 'lr': args.lr_k},
              {'params': trainer.pred_net.parameters(),   'lr': args.lr_p}]
    optimizer = torch.optim.Adam(params)
    print('set-up optimizer')

    ### Loop
    torch.manual_seed(0)
    print("started loop")
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch)

        train(train_loader, trainer, optimizer)        
        loss, pred_loss, reg_loss = validate(val_loader, trainer)
        
        log(epoch, loss, pred_loss, reg_loss)
        print("logged")

        is_best = pred_loss < lowest_loss
        lowest_loss = min(pred_loss, lowest_loss)    
        save = {'epoch:': epoch + 1, 
                'model': 'marginal_sens Trainer',
                'state_dict': trainer.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict()} 

        save_checkpoint(save, is_best)
        print("saved a checkpoint")

    print('*'*20, 'SUCCESS','*'*20)

def train(loader, trainer, optimizer):

    trainer.train()

    for t, batch in enumerate(loader):

        review, target = process_batch_sens(nlp, vocab, embd, batch)
        loss  = trainer(review, target)
        
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

        review, target = process_batch_sens(nlp, vocab, embd, batch)

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
        destination = os.path.join(args.ckp_path_remote, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
            'lr' + str(args.lr_k) + str(args.lr_p) + 'marginal_sens_log.txt')
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
            'lr' + str(args.lr_k) + str(args.lr_p) + 'marginal_sens_log.txt')

    with open(destination, 'a') as log:
        log.write(string + '\n')

def save_checkpoint(state, is_best, filename='marginal_sens_checkpoint.pth.tar'):
    """
    State is a dictionary that cotains valuable information to be saved.
    """
    if args.remote:
        destination = os.path.join(args.ckp_path_remote, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
            'lr' + str(args.lr_k) + str(args.lr_p) + filename)
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
            'lr' + str(args.lr_k) + str(args.lr_p) + filename)
    
    torch.save(state, destination)

    if is_best:
        if args.remote:
            best_destination = os.path.join(args.ckp_path_remote, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
                'lr' + str(args.lr_k) + str(args.lr_p) + 'marginal_sens_best.pth.tar')
        else:
            best_destination = os.path.join(args.ckp_path_local, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) +  
                'lr' + str(args.lr_k) + str(args.lr_p) + 'marginal_sens_best.pth.tar')
        
        shutil.copyfile(destination, best_destination)

if __name__ == '__main__':
    main()
    


