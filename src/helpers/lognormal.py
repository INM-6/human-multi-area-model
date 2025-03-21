# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    "mu_sigma_lognorm",
]

def mu_sigma_lognorm(mean, rel_sd):
    """
    Calculates parameters mu and sigma of a lognormal distribution with given
    mean and relative standard deviation.
    """
    return np.log(mean/np.sqrt(rel_sd**2 + 1)), np.sqrt(np.log(rel_sd**2 + 1))
