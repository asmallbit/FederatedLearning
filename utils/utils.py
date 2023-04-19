import torch
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.backends.cudnn.benchmark = False    # Close optimization
        torch.backends.cudnn.deterministic = True # Close optimization
        torch.cuda.manual_seed_all(seed) # All GPU (Optional)