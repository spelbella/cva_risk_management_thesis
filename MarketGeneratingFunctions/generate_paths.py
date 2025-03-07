# Math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
import random
from scipy.stats import norm
import scipy.optimize as optimize
import time

# Ez parallelization
from joblib import Parallel, delayed

# File Handling
from sys import getsizeof
import _pickle as pickle
# Alternatively use JSON which will be human readable
# import json

# Custom imports
from global_cache import Global_Cache
import base_from_gen as bg
import pricing_func as pf
from path_datatype import Path

# C magic?
#from numba import jit

"""
The Goal of this File is to generate a bunch of sample paths for the value of our tracked market through 30 years

I'm setting the preliminary goal of around 1600 paths per hour since this gives us a reasonably fast
rate (We can get a testable set in about an hour, and a set that should be large enough to be useable over 
a day 24*1600 = 38400 paths)

At the latest point I ran it for 1600 paths in 3926 seconds ie, 65 minutes. 

----------- What are we finding and saving -----------
We need to at all times know the value of our hedging instruments, these will be (as I understand it)
r_t,
Q(T_i > tau)_t           for i in 1 to 30
Alternatively 
Swaption(T_i,T_end)_t    for i in 1 to 30
Q(T_i > tau)_t           for i in 1 to 30
Alternatively 
Swap(T_i, T_end)_t
Q(T_i > tau)_t           for i in 1 to 30

We also need the value of the CVA at every timepoint 
CVA_t

And we might as well include the default intensity since we know it, so it's only taking up whatever the write
/ read time is for one parameter, and we might want to hedge in it idk.


----------- How are we Saving it -----------
While grasping the data a dataframe is a natural choice, alternatively as in the demos a numpy matrix,
to save it, preliminarily CSV feels like a natural choice but I need to think about it. Pickle is also a choice


----------- How can we get speedups -----------
The first and most important speedup in my eyes is that we can pre-calculate certain elements that remain unchanged
across runs. For instance rstar(t), A(t,T) and B(t,T) are all unchanged no matter when or where they are evaluated. 
Some of these are acsessed literally thousands of times per run, and over thousands of runs that means millions of 
evaulations, a dict has (loosely speaking) O(1) search which should be much faster than the function evaluation

The Second speedup is making sure that run specific elements are re-used as efficiently as possible. If we let each 
run occupy it's own thread we can let them share these local elements, which again saves unnessecary recalculation,
examples of where we can do this is for instance P(t,T,rt) which does depend on an observed interest rate, but will
likely occur multiple times, or the ZCB values which are time step dependant but called by many Swaptions.

Finally the last speedup is to use multithreading to wrap the above. With all of these in play it should be possible
to efficiently generate our paths 
"""

########################################################
## Defining Global Params and info                    ##
########################################################
# Set parameters
params = dict()

params["t0"] = t0 = 0
params["T"] = T = 10

# Intensity params
params["lambda0"] = lambda0 = 0.01
params["mu"] = mu = 0.01
params["kappa"] = kappa = 1
params["v"] = v = 0.1

params["j_alpha"] = j_alpha = T/4 # We should expect to see about 4 jumps,
params["gamma"] = gamma = mu/5 # With expected size mu/5

# Short rate params
params["r0"] = r0 = 0.045
params["alpha"] = alpha = 1
params["theta"] = theta = r0*alpha
params["sigma"] = sigma = np.sqrt(r0)/5

# Covariance
params["rho"] = 0.8

# Number of steps and Number of Paths total
params["N"] = N = (T-t0)*252
N_paths = 10

# Global base time grid, pre jump insertion
t_s_base = np.linspace(t0,T,N)
T_s = np.arange(0,11,1)

#########################################
## Define Globally shareable Caches    ##
#########################################
# Finding K at startup, maybe put this in the global cache file?
gc = Global_Cache(t_s_base,T_s,params)
K = gc.K

#for pathN in range(0,N_paths):
def process(pathN):
    # Generate the market grid and basic r and lambda
    [t_s,r,lambdas,r_ongrid,lambdas_ongrid] = bg.mkt_base_from_grid(t_s_base,params)
    # Create a pricing object for this specific market
    pricer = pf.PricingFunc(params, gc, t_s, r, r_ongrid,lambdas, lambdas_ongrid)
    
    Q_s = [[pricer.Q(t,T) for t in t_s_base] for T in T_s]   # Here there must be room for performance improvement? These lists could be pre-allocated or something since we know that it's going to be a list of a list of floats, same for below??
    Swaptions = [[pricer.swaption_price(t,T_s_2,K) for t in t_s_base] for T_s_2 in [T_s[i:] for i in range(0,len(T_s)-1)]] # Maybe we could C compile this file? Should be a huge performance increase, but might be a headache since we need to track down and type hint everything
    CVA = [pricer.CVA(t,T_s,K) for t in t_s_base]
    Swaps = [[pricer.swap_price(t,T_s_2,K) for t in t_s_base] for T_s_2 in [T_s[i:] for i in range(0,len(T_s)-1)]]

    return Path(t_s, lambdas, r, CVA, Q_s, Swaps, Swaptions, K)

strt = time.time()
paths = Parallel(n_jobs = 4)(delayed(process)(pathN) for pathN in range(0,N_paths)) 
end_time = time.time()

print("Total Time: %s" %(end_time - strt))
print("Average Time Per Path %s: " %((end_time - strt)/N_paths))

#with open("3kRunDemo.pkl","wb") as fp:
#    pickle.dump(paths,fp)
