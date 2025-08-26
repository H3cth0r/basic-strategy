import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import os
import sys
from typing import List, Dict, Tuple

from datasets import load_dataset


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() "cpu")
