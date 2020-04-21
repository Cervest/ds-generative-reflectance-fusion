from functools import wraps
import numpy as np
import random
import torch


def setseed(*types):
    """Wrap onto any function having a random seed as argument, e.g.
    fn(*args, seed, **kwargs) to set random seed
    """
    def seeded_fn(fn):
        @wraps(fn)
        def wrapper(*args, seed=None, **kwargs):
            if seed:
                if 'random' in types:
                    random.seed(seed)
                if 'numpy' in types:
                    np.random.seed(seed)
                if 'torch' in types:
                    torch.manual_seed(seed)
            return fn(*args, seed=seed, **kwargs)
        return wrapper
    return seeded_fn
