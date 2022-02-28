# ==========
# Utility functions
# =========
import random
import re
import time

import numpy as np
import torch


def set_seed_everywhere(seed):
    """
    Adapted from facebookresearch/drqv2
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Timer:
    """
    Adapted from facebookresearch/drqv2
    """
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

    def elapsed_time(self):
        return time.time() - self._last_time

