import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from dpp_nets.utils.io import make_embd, make_tensor_dataset
from dpp_nets.layers.layers import KernelVar, ReinforceSampler, PredNet, ReinforceTrainer


parser = argparse.ArgumentParser(description='REINFORCE VIMCO Trainer')

parser.add_argument('-a', '--aspect', type=str, choices=['aspect1', 'aspect2', 'aspect3', 'all'],
                    help='what is the target?', required=True)

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_k', '--learning_rate_k', default=1e-3, type=float,
                    metavar='LRk', help='initial learning rate for kernel net')
parser.add_argument('--lr_p', '--learning_rate_p', default=1e-4, type=float,
                    metavar='LRp', help='initial learning rate for pred net')
parser.add_argument('--reg', type=float, required=True,
                    metavar='reg', help='regularization constant')
parser.add_argument('--reg_mean', type=float, required=True,
                    metavar='reg_mean', help='regularization_mean')
parser.add_argument('--alpha_iter', type=int, required=True,
                    metavar='alpha_iter', help='How many subsets to sample from DPP? At least 2!')

# Pre-training
parser.add_argument('--pretrain_kernel', type=str, default="",
                    metavar='pretrain_kernel', help='Give name of pretrain_kernel')
parser.add_argument('--pretrain_pred', type=str, default="",
                    metavar='pretrain_pred', help='Give name of pretrain_pred')

# Train locally or remotely?
parser.add_argument('--remote', type=int,
                    help='training locally or on cluster?', required=True)

# Burnt in Paths..
parser.add_argument('--data_path_local', type=str, default='/Users/Max/data/beer_reviews',
                    help='where is the data folder locally?')
parser.add_argument('--data_path_remote', type=str, default='/cluster/home/paulusm/data/beer_reviews',
                    help='where is the data folder remotely?')
parser.add_argument('--ckp_path_local', type=str, default='/Users/Max/checkpoints/beer_reviews',
                    help='where is the checkpoints folder locally?')
parser.add_argument('--ckp_path_remote', type=str, default='/cluster/home/paulusm/checkpoints/beer_reviews',
                    help='where is the data folder remotely?')

parser.add_argument('--pretrain_path_local', type=str, default='/Users/Max/checkpoints/beer_reviews',
                    help='where is the pre_trained model? locally')
parser.add_argument('--pretrain_path_remote', type=str, default='/cluster/home/paulusm/pretrain/beer_reviews',
                    help='where is the data folder? remotely')


def main():

    global args, lowest_loss, dtype

    args = parser.parse_args()
    lowest_loss = 100 # arbitrary high number as upper bound for loss
    dtype = torch.DoubleTensor

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
    kernel_dim = 200
    hidden_dim = 500
    enc_dim = 200
    if args.aspect == 'all':
        target_dim = 3
    else: 
        target_dim = 1

    # Model
    torch.manual_seed(1)


    # Add pre-training here...
    kernel_net = KernelVar(embd_dim, hidden_dim, kernel_dim)
    sampler = ReinforceSampler(args.alpha_iter)
    pred_net = PredNet(embd_dim, hidden_dim, enc_dim, target_dim)

    if args.pretrain_kernel:
        if args.remote:
            state_dict = torch.load(args.pretrain_path_remote + args.pretrain_kernel)
        else:
            state_dict = torch.load(args.pretrain_path_local + args.pretrain_kernel)
        kernel_net.load_state_dict(state_dict)

    if args.pretrain_pred:
        if args.remote:
            state_dict = torch.load(args.pretrain_path_remote + args.pretrain_pred)
        else:
            state_dict = torch.load(args.pretrain_path_local + args.pretrain_pred)
        pred_net.load_state_dict(state_dict)

    # continue with trainer
    trainer = ReinforceTrainer(embd, kernel_net, sampler, pred_net)
    trainer.reg = args.reg
    trainer.reg_mean = args.reg_mean
    trainer.activation = nn.Sigmoid()
    trainer.type(dtype)

    print("created trainer")

    params = [{'params': trainer.kernel_net.parameters(), 'lr': args.lr_k},
              {'params': trainer.pred_net.parameters(), 'lr': args.lr_p}]
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
                'model': 'Marginal Trainer',
                'state_dict': trainer.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict()} 

        save_checkpoint(save, is_best)
        print("saved a checkpoint")

    print('*'*20, 'SUCCESS','*'*20)


def train(loader, trainer, optimizer):

    trainer.train()

    for t, (review, target) in enumerate(loader):
        review = Variable(review)

        if args.aspect == 'all':
            target = Variable(target[:,:3]).type(dtype)
        else:
            target = Variable(target[:,int(args.aspect[-1])]).type(dtype)

        loss  = trainer(review, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("trained one batch")

def validate(loader, trainer):
    """
    Note, we keep the sampling as before. 
    i.e what ever alpha_iter is, we take it. 
    """
    trainer.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_reg_loss = 0.0

    for i, (review, target) in enumerate(loader, 1):
        review = Variable(review, volatile=True)

        if args.aspect == 'all':
            target = Variable(target[:,:3], volatile=True).type(dtype)
        else:
            target = Variable(target[:,int(args.aspect[-1])], volatile=True).type(dtype)

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
          'alpha_iter' + str(args.alpha_iter) + str(args.pretrain_kernel) + str(args.pretrain_pred) + 'reinforce_log.txt')
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
          'alpha_iter' + str(args.alpha_iter) + str(args.pretrain_kernel) + str(args.pretrain_pred) + 'reinforce_log.txt')

    with open(destination, 'a') as log:
        log.write(string + '\n')

def save_checkpoint(state, is_best, filename='reinforce_checkpoint.pth.tar'):
    """
    State is a dictionary that cotains valuable information to be saved.
    """
    if args.remote:
        destination = os.path.join(args.ckp_path_remote, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
          'alpha_iter' + str(args.alpha_iter) + str(args.pretrain_kernel) + str(args.pretrain_pred) + str(args.filename))
    else:
        destination = os.path.join(args.ckp_path_local, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
          'alpha_iter' + str(args.alpha_iter) + str(args.pretrain_kernel) + str(args.pretrain_pred) + str(args.filename))
    
    torch.save(state, destination)

    if is_best:
        if args.remote:
            best_destination = os.path.join(args.ckp_path_remote, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) + 
               'alpha_iter' + str(args.alpha_iter) + str(args.pretrain_kernel) + str(args.pretrain_pred) + 'reinforce_best.pth.tar')
        else:
            best_destination = os.path.join(args.ckp_path_local, args.aspect + 'reg' + str(args.reg) + 'reg_mean' + str(args.reg_mean) +  
               'alpha_iter' + str(args.alpha_iter) + str(args.pretrain_kernel) + str(args.pretrain_pred) + 'reinforce_best.pth.tar')
        
        shutil.copyfile(destination, best_destination)

if __name__ == '__main__':
    main()