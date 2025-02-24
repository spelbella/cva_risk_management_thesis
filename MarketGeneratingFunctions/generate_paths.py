import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
import random
from scipy.stats import norm
import scipy.optimize as optimize
import time
from joblib import Parallel, delayed

# Custom imports
import base_from_gen as bg
import pricing_func as pf

"""
The Goal of this File is to generate a bunch of sample paths for the value of our tracked market through 30 years

I'm setting the preliminary goal of around 1600 paths per hour since this gives us a reasonably fast
rate (We can get a testable set in about an hour, and a set that should be large enough to be useable over 
a day 24*1600 = 38400 paths)

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
params["rho"] = rho = 0.8

# Number of steps and Number of Paths total
params["N"] = N = (T-t0)*128
N_paths = 10

# Global base time grid, pre jump insertion
t_s_base = np.linspace(t0,T,N)
T_s = np.arange(0,11,1)


#########################################
## Define Globally shareable Caches    ##
#########################################
# General functions
def A_naive(t,T):
    A = -(sigma**2)/(4*alpha**3) * (3 + np.e**(-2*alpha*(T-t)) - 4*np.e**(-alpha*(T-t)) - 2*alpha*(T-t)) - (theta/alpha)*((T-t) - (1/alpha)*(1 - np.e**(-alpha*(T-t))))
    return A

def B_naive(t,T):
    B = -(1/alpha)*(1 - np.e**(-alpha*(T-t)))
    return B

minimize = lambda cashflows, dates, Tm, rstar: 1 - sum([c*np.e**(A_naive(Tm,date) + B_naive(Tm,date)*rstar) for c, date in zip(cashflows,dates)])
def rstar_naive(T_s, K):
    cashflows = (K*np.diff(T_s) + np.concatenate((np.zeros((1,len(T_s)-2)), np.asmatrix(1)),1)).A1
    dates = T_s[1:]
    Tm = T_s[0]
    optim = lambda rstarl: minimize(cashflows, dates, Tm, rstarl)
    rstar = optimize.newton(optim, 0)
    return rstar

# A big cache
class Global_Cache:
    def __init__(self, T_s, K):
        self.A_cache = dict()
        self.B_cache = dict()
        self.rstar_cache = dict()
        
        # Fill the Cache 
        for t in t_s_base:
            for T in T_s:
                # Fill the A_cache
                keyA = self.A_key(t,T)
                self.A_cache[keyA] = A_naive(t,T)

                # Fill the B_cache
                keyB = self.B_key(t,T)
                self.B_cache[keyB] = B_naive(t,T)

        for i in range(0,len(T_s)-1):
            # Fill the rstar cache
            keyR = self.rstar_key(T_s[i:],K)
            self.rstar_cache[keyR] = rstar_naive(T_s[i:],K)
           
    def A(self,t,T):
        key = self.A_key(t,T)
        if key in self.A_cache:
            A = self.A_cache[key]
        else:
            A = A_naive(t,T)
        return A
    
    def B(self,t,T):
        key = self.B_key(t,T)
        if key in self.B_cache:
            B = self.B_cache[key]
        else:
            B = B_naive(t,T)
        return B
    
    def rstar(self,T_s,K):
        key = self.rstar_key(T_s,K)
        if key in self.rstar_cache:
            r_star = self.rstar_cache[key]
        else:
            r_star = rstar_naive(T_s,K)
        return r_star

    def A_key(self,t,T):
        key = hash((t,T))#str(np.round(t,15)) + ',' + str(np.round(T,15))
        return key
    
    def B_key(self,t,T):
        key = hash((t,T))#str(np.round(t,15)) + ',' + str(np.round(T,15))
        return key
        
    def rstar_key(self,T_s,K):
        key = hash((str(T_s),K))#str(np.round(T_s,15)) + ',' + str(np.round(K,15))
        return key

P = lambda t,T,rt: np.e**(A_naive(t,T)+B_naive(t,T)*rt)
K = (P(0,T_s[0],r0) - P(0,T_s[-1],r0))/(sum([P(0,T_s[i],r0) for i in range(1,len(T_s))]))
gc = Global_Cache(T_s,K)

"""
# In Cache Time test
strt1 = time.time()
for i in np.arange(0,100,1):
    gc.A(t_s_base[i],30)
end1 = time.time()

# Out of Cache Time test
strt2 = time.time()
for i in np.arange(0,100,1):
    gc.A(t_s_base[i],31)
end2 = time.time()

print("In Cache 100 steps A speed = %s" %(end1-strt1))
print("Out of Cache 100 steps A speed = %s" %(end2-strt2))
## RSTAR is orders of magnitude better as well, once the bigger loop is setup it's worth double
checking that the dict is finding all of the available values since floating points might jiggle into 
not counting as the same hash.
"""


"""
For the sake of simplicity I'll define a path object which just stores the relevant matrices for now, 
this could really be a dict or a dataframe or a np.ndarray where we know what sits at each index
"""
class Path():
    def __init__(self, t_s, lambdas, r, CVA, Q_s, Swaps, Swaptions):
        self.t_s = t_s
        self.lambdas = lambdas
        self.r = r
        self.CVA = CVA
        self.Q_s = Q_s
        self.Swaps = Swaps
        self.Swaptions = Swaptions

strt = time.time()
paths = []
for pathN in range(0,N_paths):
    # Generate the market grid and basic r and lambda
    [t_s,r,lambdas,r_ongrid,lambdas_ongrid] = bg.mkt_base_from_grid(t_s_base,params)
    # Create a pricing object for this specific market
    pricer = pf.PricingFunc(params, gc, t_s, r, r_ongrid,lambdas, lambdas_ongrid)
    
    #Q_s = [[pricer.Q(t,T) for t in t_s] for T in T_s]
    CVA = [pricer.CVA(t,T_s,K) for t in t_s]
    #Swaptions = [[pricer.swaption_price(t,T_s_2,K) for t in t_s] for T_s_2 in [T_s[i:] for i in range(0,len(T_s)-1)]]
    #Swaps = [[pricer.swap_price(t,T_s_2,K) for t in t_s] for T_s_2 in [T_s[i:] for i in range(0,len(T_s)-1)]]
    print("N = %s" %pathN)

    paths.append(Path(t_s, lambdas, r, CVA, None, None, None))


print((time.time()-strt)/N_paths)
