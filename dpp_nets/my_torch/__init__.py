import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable

import dpp_nets.dpp as dpp

from dpp_nets.my_torch.DPP import DPP
from dpp_nets.my_torch.DPP import DPPLayer