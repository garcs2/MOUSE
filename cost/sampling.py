# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
#
# --- PERFORMANCE PATCH ---
# Original behavior/signatures are unchanged: sampler(distribution, **kwargs)
# still returns a single scalar draw, exactly as before. The only change is
# that create_lognormal_sampler no longer recomputes mu/sigma from scratch
# (np.log x3, np.mean, np.std, plus a new closure) on every single call.
#
# Why this is safe to cache: for a given cost-database row, low_cost/high_cost/
# class3_cost are the SAME on every one of the 1000 Monte Carlo samples — only
# the random draw differs. So mu/sigma are sample-invariant and can be computed
# once per unique (low, high, class3) triple and reused. lru_cache gives us
# that for free, keyed on the exact input values, with no changes needed at
# the call sites in cost_scaling.py.

import numpy as np
from functools import lru_cache


@lru_cache(maxsize=None)
def _lognormal_mu_sigma(low_cost, high_cost, class3_cost):
    """
    Compute (mu, sigma) for the lognormal distribution implied by a row's
    low/high/central cost estimate. Cached because these three inputs are
    identical across every Monte Carlo sample for a given row — only the
    draw itself should vary sample to sample.
    """
    ln_low_cost = np.log(low_cost)
    ln_high_cost = np.log(high_cost)
    ln_class3_cost = np.log(class3_cost)

    mu = np.mean([ln_low_cost, ln_high_cost, ln_class3_cost])
    sigma = np.std([ln_low_cost, ln_high_cost, ln_class3_cost], ddof=0)
    return mu, sigma


def create_lognormal_sampler(low_cost, high_cost, class3_cost):
    # Cast to plain floats so the cache key is stable regardless of whether
    # callers pass in numpy scalars, pandas cell values, or Python floats.
    mu, sigma = _lognormal_mu_sigma(float(low_cost), float(high_cost), float(class3_cost))
    return np.random.lognormal(mean=mu, sigma=sigma)


def sample_lognormal_batch(low_cost, high_cost, class3_cost, size):
    """
    New helper (not required by existing call sites): draws `size` samples at
    once instead of `size` separate Python-level calls. Useful if you later
    vectorize scale_cost() to loop over rows once and draw all N samples for
    each row in a single call, rather than looping over samples N times and
    calling the scalar sampler on each pass. See the vectorization note in
    cost_scaling_vectorized_example.py.
    """
    mu, sigma = _lognormal_mu_sigma(float(low_cost), float(high_cost), float(class3_cost))
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)


def truncated_normal_sample(mean, std, lower_bound, upper_bound):
    while True:
        sample = np.random.normal(mean, std)
        if lower_bound <= sample <= upper_bound:
            return sample


def sample_truncated_normal_batch(mean, std, lower_bound, upper_bound, size):
    """
    Vectorized batch version of truncated_normal_sample. Draws `size` normal
    samples at once and re-draws (in a vectorized loop) only the out-of-bound
    ones, instead of looping one Python-level rejection-sample at a time.
    """
    samples = np.random.normal(mean, std, size=size)
    mask = (samples < lower_bound) | (samples > upper_bound)
    while mask.any():
        samples[mask] = np.random.normal(mean, std, size=mask.sum())
        mask = (samples < lower_bound) | (samples > upper_bound)
    return samples


def uniform_sample(low, high):
    return np.random.uniform(low, high)


def sample_uniform_batch(low, high, size):
    return np.random.uniform(low, high, size=size)


def sampler(distribution, **kwargs):
    if distribution == "Lognormal":
        return create_lognormal_sampler(kwargs['low_cost'], kwargs['high_cost'], kwargs['class3_cost'])
    elif distribution == "Truncated Normal":
        return truncated_normal_sample(kwargs['mean'], kwargs['std'], kwargs['lower_bound'], kwargs['upper_bound'])
    elif distribution == "Uniform":
        return uniform_sample(kwargs['low'], kwargs['high'])
    else:
        raise ValueError("Unavailable Distribution")