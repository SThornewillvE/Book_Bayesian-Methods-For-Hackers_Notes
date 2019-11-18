# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:37:49 2019

@author: sthornewillvonessen
"""

# ----------------------------------------------------------------------------------------------------------------------
# Import Functions and Data
# ----------------------------------------------------------------------------------------------------------------------

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize
from scipy.stats.mstats import mquantiles

challenger_data = np.genfromtxt("./dat/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")

# drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]


# ----------------------------------------------------------------------------------------------------------------------
# Define Bayesian Architecture
# ----------------------------------------------------------------------------------------------------------------------

temperature = challenger_data[:, 0]
D= challenger_data[:, 1]  # is defect=

# Create variables for parameters
beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)


# ----------------------------------------------------------------------------------------------------------------------
# Define Functions
# ----------------------------------------------------------------------------------------------------------------------

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


@pm.deterministic
def p(t=temperature, alpha=alpha, beta=beta):
    return 1.0 / (1 + np.exp(beta*t + alpha))


def plot_raw(challener_data):
 
    # First, plot challenger data
    plt.scatter(challenger_data[:, 0][challenger_data[:, 1] == 0], 
                challenger_data[:, 1][challenger_data[:, 1] == 0])
    plt.scatter(challenger_data[:, 0][challenger_data[:, 1] == 1], 
                challenger_data[:, 1][challenger_data[:, 1] == 1])
    plt.title("O-Ring Failures as a Function of Temperature")
    plt.ylabel("Bool of Failure")
    plt.xlabel("Temperature")
    plt.show()


def plot_posterior(alpha_samples, beta_samples):

    figsize(12.5, 6)
    
    plt.subplot(211)
    plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
    plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
             label=r"posterior of $\beta$", color="#7A68A6", density=True)
    plt.legend()
    
    plt.subplot(212)
    plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
             label=r"posterior of $\alpha$", color="#A60628", density=True)
    plt.legend();
    
    
def plot_against_data(t, mean_prob_t, qs):
    
    figsize(12.5, 4)
    
    plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

    plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)
    
    plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
             label="average posterior \nprobability of defect")
    
    plt.xlim(t.min(), t.max())
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower left")
    plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
    plt.xlabel("temp, $t$")
    
    plt.ylabel("probability estimate")
    plt.title("Posterior probability estimates given temp. $t$");
    
def plot_t_31(alpha_samples, beta_samples):
    
    figsize(12.5, 2.5)

    prob_31 = logistic(31, beta_samples, alpha_samples)
    
    plt.xlim(0.995, 1)
    plt.hist(prob_31, bins=1000, density=True, histtype='stepfilled')
    plt.title("Posterior distribution of probability of defect, given $t = 31$")
    plt.xlabel("probability of defect occurring in O-ring");
                 
# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def Main():
    
    # Plot Raw Data
    plot_raw(challenger_data)
    
    # Calculate observed values
    observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)
    
    # Create and sample from model
    model = pm.Model([observed, beta, alpha])
    
    # Once again, this code will be explored next chapter
    map_ = pm.MAP(model)
    map_.fit()
    mcmc = pm.MCMC(model)
    mcmc.sample(120000, 100000, 2)
    
    alpha_samples = mcmc.trace('alpha')[:, None]  # best to make them 1d
    beta_samples = mcmc.trace('beta')[:, None]

    # Do some plotting    
    plot_posterior(alpha_samples, beta_samples)
    
    t = np.linspace(temperature.min() - 5, temperature.max() + 5, 50)[:, None]
    p_t = logistic(t.T, beta_samples, alpha_samples)

    mean_prob_t = p_t.mean(axis=0)
    
    # vectorized bottom and top 2.5% quantiles for credible interval
    qs = mquantiles(p_t, [0.025, 0.975], axis=0)
    
    plot_against_data(t, mean_prob_t, qs)
    
    plot_t_31(alpha_samples, beta_samples)
    
    
if __name__ == '__main__':
    Main()
