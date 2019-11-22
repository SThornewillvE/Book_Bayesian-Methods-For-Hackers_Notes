# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:37:36 2019

@author: sthornewillvonessen
"""

# ----------------------------------------------------------------------------------------------------------------------
# Import Functions and Data
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import scipy.stats as stats
import scipy.optimize as sop

from IPython.core.pylabtools import figsize

plt.style.use("seaborn")

# ----------------------------------------------------------------------------------------------------------------------
# Define Bayesian Architecture
# ----------------------------------------------------------------------------------------------------------------------
           
data_mu = [3e3, 12e3]
data_std = [5e2, 3e3]

mu_prior = 35e3
std_prior = 75e2

true_price = pm.Normal("true_price", mu_prior, 1.0 / std_prior ** 2)

prize_1 = pm.Normal("first_prize", data_mu[0], 1.0 / data_std[0] ** 2)
prize_2 = pm.Normal("second_prize", data_mu[1], 1.0 / data_std[1] ** 2)
price_estimate = prize_1 + prize_2


# ----------------------------------------------------------------------------------------------------------------------
# Define Functions
# ----------------------------------------------------------------------------------------------------------------------
   
@pm.potential
def error(true_price=true_price, price_estimate=price_estimate):
        return pm.normal_like(true_price, price_estimate, 1 / (3e3) ** 2)
    

def showdown_loss(guess, true_price, risk=80_000):
    
    # Create loss
    loss = np.zeros_like(true_price)
    
    # Create masks
    ix = true_price < guess
    close_mask = [abs(true_price - guess) < 250]
    
    # Fill loss
    loss[~ix] = np.abs(guess - true_price[~ix])
    loss[close_mask] = -2 * true_price[close_mask]
    loss[ix] = risk
    
    return loss


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def Main():
    
    mcmc = pm.MCMC([true_price, prize_1, prize_2, price_estimate, error])
    mcmc.sample(50000, 10000)
    
    price_trace = mcmc.trace("true_price")[:]    

    # Plotting
    figsize(12.5, 4)
    x = np.linspace(5000, 40000)
    plt.plot(x, stats.norm.pdf(x, 35000, 7500), c="k", lw=2,
             label="prior dist. of suite price")
    
    _hist = plt.hist(price_trace, bins=35, density=True, histtype="stepfilled")
    plt.title("Posterior of the true price estimate")
    plt.vlines(mu_prior, 0, 1.1 * np.max(_hist[0]), label="prior's mean",
               linestyles="--")
    plt.vlines(price_trace.mean(), 0, 1.1 * np.max(_hist[0]),
               label="posterior's mean", linestyles="-.")
    plt.legend(loc="upper left");
    plt.show()
    
    guesses = np.linspace(5000, 50000, 70)
    risks = np.linspace(30000, 150000, 6)
    expected_loss = lambda guess, risk: showdown_loss(guess, price_trace, risk).mean()
    
    figsize(12.5, 7)
    
    for _p in risks:
        results = [expected_loss(_g, _p) for _g in guesses]
        plt.plot(guesses, results, label="%d" % _p)
    
    # Plotting
    plt.title("Expected loss of different guesses, \nvarious risk-levels of \
    overestimating")
    plt.legend(loc="upper left", title="Risk parameter")
    plt.xlabel("price bid")
    plt.ylabel("expected loss")
    plt.xlim(5000, 30000);

    
    ax = plt.subplot(111)

    for _p in risks:
        _color = next(ax._get_lines.prop_cycler)
        _min_results = sop.fmin(expected_loss, 15000, args=(_p,),disp = False)
        _results = [expected_loss(_g, _p) for _g in guesses]
        
        plt.plot(guesses, _results , color = _color['color'])
        plt.scatter(_min_results, 0, s = 60, \
                    color= _color['color'], label = "%d"%_p)
        plt.vlines(_min_results, 0, 120000, color = _color['color'], linestyles="--")
    
        plt.title("Expected loss & Bayes actions of different guesses, \n \
        various risk-levels of overestimating")
        plt.legend(loc="upper left", scatterpoints=1, title="Bayes action at risk:")
        plt.xlabel("price guess")
        plt.ylabel("expected loss")
        plt.xlim(7000, 30000)
        plt.ylim(-1000, 80000);

if __name__ == '__main__':
    Main()
