# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:28:34 2019

@author: SimonThornewill
"""

# ----------------------------------------------------------------------------------------------------------------------
# Import Functions and Data
# ----------------------------------------------------------------------------------------------------------------------

import pymc as pm
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize


# ----------------------------------------------------------------------------------------------------------------------
# Define Functions
# ----------------------------------------------------------------------------------------------------------------------

def plot_posteriors(mcmc, p_true):
    
    figsize(12.5, 4)
    plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
    plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    plt.hist(mcmc.trace("p")[:], bins=25, histtype="stepfilled", density=True)
    plt.legend();

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def Main():
    
    p = pm.Uniform('p', lower=0, upper=1)

    # Define true parameters for experimental purposes
    p_true = 0.05
    N = 1_500
    
    # Generate fake Data using parameters
    occurrences = pm.rbernoulli(p_true, N)
    
    print(occurrences)
    print(len(occurrences), occurrences.sum())
    
    # Define observation random variavle
    obs = pm.Bernoulli("obs", p, value=occurrences, observed=True)
    
    # Solve using MCMC
    mcmc = pm.MCMC([p, obs])
    mcmc.sample(18_000, 1_000)
    
    plot_posteriors(mcmc, p_true)


if __name__ == '__main__':
    Main()
