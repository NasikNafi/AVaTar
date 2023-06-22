"""Shared utilities for agents."""
import math
from scipy.stats import truncnorm

def generate_gammas(mean, sd, low, upp, num_gamma):
    truncated_normal = truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    return truncated_normal.rvs(num_gamma)
