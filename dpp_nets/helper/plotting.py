import torch
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict

def moving_average(array, n):
	cumsum = np.cumsum(array, dtype=float)
	cumsum[n:] = cumsum[n:] - cumsum[:-n]
	return cumsum[n-1:] / n

def plot_floats(my_def_dict, ma=0, fname=None, title="Untitled", xlabel="Unlabelled", ylabel="Unlabelled"):

    if isinstance(my_def_dict, defaultdict):
        my_dict = {k: sum(v)/ len(v) for k,v in my_def_dict.items()}
    else:
    	my_dict = my_def_dict

    x, y = zip(*sorted(my_dict.items()))

    if ma:
    	x = moving_average(np.array(x), ma)
    	y = moving_average(np.array(y), ma)
    else: 
        x = np.array(x)
        y = np.array(y)

    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if fname: 
        plt.savefig(str(fname) + '.pdf', format='pdf')
    plt.show()

def plot_defaultdict(my_def_dict, ma=0, fname=None, title="Untitled", xlabel="Unlabelled", ylabel="Unlabelled"):

	my_dict = {k: torch.stack(v) for k, v in my_def_dict.items()}
	plot_dict(my_dict, ma, fname, title, xlabel, ylabel)

def plot_dict(my_dict, ma=0, fname=None, title="Untitled", xlabel="Unlabelled", ylabel="Unlabelled"):
	my_dict = {k: torch.mean(v) for k, v in my_dict.items()}
	x, y = zip(*sorted(my_dict.items()))

	if ma: 
		x = moving_average(np.array(x), ma)
		y = moving_average(np.array(y), ma)
	else:
		x = np.array(x)
		y = np.array(y)

	plt.plot(x, y)
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)

	if fname: 
		plt.savefig(str(fname) + '.pdf', format='pdf')
	plt.show()

def gen_matrix_from_cluster_ix(cluster_ix):
	if not isinstance(cluster_ix, np.ndarray):
		cluster_ix = cluster_ix.numpy()
	set_size = cluster_ix.shape[0]
	matrix = np.tile(cluster_ix, (set_size,1))
	matrix = matrix - matrix.T
	matrix = ~matrix.astype(bool)
	return matrix

def plot_matrix(matrix):
	if not isinstance(matrix, np.ndarray):
		matrix = matrix.numpy()

	plt.matshow(matrix, interpolation='nearest')
	plt.show()

def plot_embd(embd):
	"""
	Plots a colormap of the L-kernel given embd,
	such that L = embd * embd.T
	Arguments:
	- embd: numpy array or torch tensor
	"""
	if not isinstance(embd, np.ndarray):
		embd = embd.numpy()
	
	L = embd.dot(embd.T)
	plot_matrix(L)
