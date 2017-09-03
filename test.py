import sys, warnings, traceback, torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from dpp_nets.my_torch.linalg import custom_decomp, custom_inverse 
from dpp_nets.my_torch.DPP import DPP, AllInOne
from dpp_nets.my_torch.utilities import compute_baseline
from itertools import accumulate
from dpp_nets.layers.layers import KernelVar


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
    
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

def main():
    kernel_net = KernelVar(200, 500, 200)
    words = Variable(torch.randn(200, 200, 200))
    kernel_net(words)

if __name__ == '__main__':
    main()