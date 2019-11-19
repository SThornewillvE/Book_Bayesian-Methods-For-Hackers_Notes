# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 07:07:38 2019

@author: SimonThornewill
"""

# ======================================================================================================================
# Import Packages and Data
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

# from IPython.core.pylabtools import figsize
from scipy.stats import norm

data = np.loadtxt("./dat/mixture_data.csv", delimiter=",")


# ======================================================================================================================
# Define Bayesian Structure
# ======================================================================================================================

# Define probability of assigning one or another cluster
p = pm.Uniform("p", 0, 1)

assignment = pm.Categorical("assignment", [p, 1 - p], size=data.shape[0])

# Create priors for standard deviations and centers
sigmas = pm.Uniform("stds", 0, 100, size=2)
centers = pm.Normal("centers", [120, 190], [0.01, 0.01], size=2)


# ======================================================================================================================
# Define Functions
# ======================================================================================================================

@pm.deterministic
def taus(sigmas=sigmas):
   return 1.0 / sigmas **2


@pm.deterministic
def center_i(assignment=assignment, centers=centers):
    return centers[assignment]


@pm.deterministic
def tau_i(assignment=assignment, taus=taus):
    return taus[assignment]


def plot_initial_data(data):
    
    plt.hist(data, bins=20, color="k", histtype="stepfilled", alpha=0.8)
    plt.title("Histogram of the dataset")
    plt.ylim([0, None])
    plt.show()

def plot_mms(mcmc):

    # Get traces    
    center_trace = mcmc.trace("centers")[:]
    std_trace = mcmc.trace("stds")[:]

    # Set beautiful colors
    colors = ["#348ABD", "#A60628"]

    x = np.linspace(20, 300, 500)
    posterior_center_means = center_trace.mean(axis=0)
    posterior_std_means = std_trace.mean(axis=0)
    posterior_p_mean = mcmc.trace("p")[:].mean()
    
    plt.hist(data, bins=20, histtype="step", density=True, color="k",
         lw=2, label="histogram of data")
    y = posterior_p_mean * norm.pdf(x, loc=posterior_center_means[0],
                                    scale=posterior_std_means[0])
    plt.plot(x, y, label="Cluster 0 (using posterior-mean parameters)", lw=3)
    plt.fill_between(x, y, color=colors[1], alpha=0.3)
    
    y = (1 - posterior_p_mean) * norm.pdf(x, loc=posterior_center_means[1],
                                          scale=posterior_std_means[1])
    plt.plot(x, y, label="Cluster 1 (using posterior-mean parameters)", lw=3)
    plt.fill_between(x, y, color=colors[0], edgecolor=colors[0], alpha=0.3)
    
    plt.legend(loc="upper left")
    plt.title("Visualizing Clusters using posterior-mean parameters");
    plt.show()


# ======================================================================================================================
# Main
# ======================================================================================================================

def Main():
    
    plot_initial_data(data)
    
    # and to combine it with the observations:
    observations = pm.Normal("obs", center_i, tau_i, value=data, observed=True)
    
    # below we create a model class
    model = pm.Model([p, assignment, observations, taus, centers, sigmas])
    
    mcmc = pm.MCMC(model)
    mcmc.sample(50_000)
    
    plot_mms(mcmc)


if __name__ == '__main__':
    Main()
    