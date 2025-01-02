# import this when doing interactive stuff

# useful libs
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torcheval.metrics as ms
import torch.utils.data as td

# info
import logging
import importlib
from pprint import pprint
from tnibs.utils import *
if is_notebook():
    from IPython.display import display, clear_output

from tnibs.data import *
from tnibs.data.utils import dfs
from tnibs.modules.modules import *
from tnibs.metric import *
from tnibs.train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if vb(5):
    # Device configuration
    if torch.cuda.is_available():
        print(f"Using: {device}. Device: {torch.cuda.get_device_name()}")


