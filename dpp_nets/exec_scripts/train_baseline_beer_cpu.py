from dpp_nets.layers.layers import *
import torch
import torch.nn as nn
from collections import OrderedDict
import shutil
import time
import gzip
import os
import json
import numpy as np
from dpp_nets.utils.io import make_embd, make_tensor_dataset, load_tensor_dataset
from dpp_nets.utils.io import data_iterator, load_embd
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import time
from dpp_nets.my_torch.utilities import pad_tensor

# C 