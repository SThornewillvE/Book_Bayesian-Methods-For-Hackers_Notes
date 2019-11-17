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
# Define Bayesian Architecture
# ----------------------------------------------------------------------------------------------------------------------

# Define random variables for probs of A and B
p_A = pm.Uniform('p_A', lower=0, upper=1)
p_B = pm.Uniform('p_B', lower=0, upper=1)

# Define true parameters for experimental purposes
p_true_A = 0.05
p_true_B = 0.04

# Note: Unequal sample sizes are valid in bayesian analysis
N_A = 1_500
N_B = 750

# Generate fake Data using parameters
observations_A = pm.rbernoulli(p_true_A, N_A)
observations_B = pm.rbernoulli(p_true_B, N_B)

# Define observation random variavle
obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)


# ----------------------------------------------------------------------------------------------------------------------
# Define Functions
# ----------------------------------------------------------------------------------------------------------------------

@pm.deterministic
def delta(p_A=p_A, p_B=p_B):
    return p_A - p_B


def plot_posteriors(mcmc, true_p_A, true_p_B):
    
    p_A_samples = mcmc.trace("p_A")[:]
    p_B_samples = mcmc.trace("p_B")[:]
    delta_samples = mcmc.trace("delta")[:]
    
    figsize(12.5, 10)
    
    # histogram of posteriors
    
    ax = plt.subplot(311)
    
    plt.xlim(0, .1)
    plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="posterior of $p_A$", color="#A60628", normed=True)
    plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
    plt.legend(loc="upper right")
    plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
    
    ax = plt.subplot(312)
    
    plt.xlim(0, .1)
    plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="posterior of $p_B$", color="#467821", normed=True)
    plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
    plt.legend(loc="upper right")
    
    ax = plt.subplot(313)
    
    plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of delta", color="#7A68A6", normed=True)
    plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
               label="true delta (unknown)")
    plt.vlines(0, 0, 60, color="black", alpha=0.2)
    plt.legend(loc="upper right");

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def Main():
    
    # Solve using MCMC
    mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
    mcmc.sample(20000, 1000)
    
    plot_posteriors(mcmc, p_true_A, p_true_B)


if __name__ == '__main__':
    Main()
