# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:13:01 2019

@author: sthornewillvonessen
"""

# ----------------------------------------------------------------------------------------------------------------------
# Import Functions and Data
# ----------------------------------------------------------------------------------------------------------------------

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize

# Get data
count_data = np.loadtxt('./dat/txtdata.csv')


# ----------------------------------------------------------------------------------------------------------------------
# Define Bayesian Architecture
# ----------------------------------------------------------------------------------------------------------------------

# Get number of observations
n_count_data = len(count_data)

# Set prior param for exponential functions
alpha = 1 / count_data.mean()

# Create lambdas
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

# Create tau
tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)


# ----------------------------------------------------------------------------------------------------------------------
# Define Functions
# ----------------------------------------------------------------------------------------------------------------------

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    """
    Function that sets the lambda for each observation depending on tau.
    
    :Inputs:
        :tau: Int. The point where lambda_1 switches to lambda_2
        :lambda_1: Float. Rate of recieves before switch
        :lambda_2: Float. Rate of recieves after switch
        :n_count_data: Total number of observations to create in `out`
    :Returns:
        :out: Array. Setting of lambda_1 and lambda_2 depending on tau
    """
    
    # Create output vector
    out = np.zeros(n_count_data)
    
    out[:tau] = lambda_1  # lambda before tau is lambda1
    out[tau:] = lambda_2  # lambda after (and including) tau is lambda2
    return out


def plot_data(lambda_1_samples, lambda_2_samples, tau_samples):
    """
    Plot resulting data.
    """
    
    figsize(12.5, 10)
    # histogram of the samples:
    
    ax = plt.subplot(311)
    ax.set_autoscaley_on(False)
    
    plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_1$", color="#A60628", density=True)
    plt.legend(loc="upper left")
    plt.title(r"""Posterior distributions of the variables
        $\lambda_1,\;\lambda_2,\;\tau$""")
    plt.xlim([15, 30])
    plt.xlabel("$\lambda_1$ value")
    
    ax = plt.subplot(312)
    ax.set_autoscaley_on(False)
    plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_2$", color="#7A68A6", density=True)
    plt.legend(loc="upper left")
    plt.xlim([15, 30])
    plt.xlabel("$\lambda_2$ value")
    
    plt.subplot(313)
    w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
    plt.hist(tau_samples, bins=n_count_data, alpha=1,
             label=r"posterior of $\tau$",
             color="#467821", weights=w, rwidth=2.)
    plt.xticks(np.arange(n_count_data))
    
    plt.legend(loc="upper left")
    plt.ylim([0, .75])
    plt.xlim([35, len(count_data) - 20])
    plt.xlabel(r"$\tau$ (in days)")
    plt.ylabel("probability");


def solve_exercises(mcmc):
    
    print("Q: Using lambda_1_samples and lambda_2_samples, what is the mean of the posterior distributions of λ1 and λ2?")
    
    lambda_1_samples = mcmc.trace('lambda_1')[:]
    lambda_2_samples = mcmc.trace('lambda_2')[:]
    tau_samples = mcmc.trace('tau')[:]    
    
    print("A:\n λ1: {} \n λ2?: {} \n\n".format(lambda_1_samples.mean(), lambda_2_samples.mean()))
    
    print("Q: What is the expected percentage increase in text-message rates?")
    
    print("A: {} \n\n".format(((lambda_2_samples-lambda_1_samples)/lambda_1_samples).mean()))
    
    print("Q: What is the mean of λ1 given that we know τ is less than 45?")
    
    # Check this
    lambda_1_giv_tau_leq_45 = lambda_1_samples[tau_samples < 45]
    
    print("A: {} \n\n".format(lambda_1_giv_tau_leq_45.mean()))
    

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def Main():
    
    # Create observation
    observation = pm.Poisson("obs", 
                             lambda_, 
                             value=count_data, 
                             observed=True)
    
    # Create model
    model = pm.Model([observation, lambda_1, lambda_2, tau])
    
    # Solve using MCMC (Explained in Chapter 3)
    mcmc = pm.MCMC(model)
    mcmc.sample(40000, 10000, 1)
    
    # Get traces for parameters
    lambda_1_samples = mcmc.trace('lambda_1')[:]
    lambda_2_samples = mcmc.trace('lambda_2')[:]
    tau_samples = mcmc.trace('tau')[:]
    
    plot_data(lambda_1_samples, lambda_2_samples, tau_samples)

    solve_exercises(mcmc)


if __name__ == '__main__':
    Main()
