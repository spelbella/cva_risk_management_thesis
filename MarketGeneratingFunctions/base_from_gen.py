import numpy as np
from numpy.random import Generator, PCG64

def mkt_base_from_grid(grid,params):
    t_s_base = grid
    t0 = params["t0"]
    T = params["T"]
    gamma = params["gamma"]
    j_alpha = params["j_alpha"]
    rho = params["rho"]
    lambda0 = params["lambda0"]
    r0 = params["r0"]
    theta = params["theta"]
    alpha = params["alpha"]
    sigma = params["sigma"]
    kappa = params["kappa"]
    mu = params["mu"]
    v = params["v"]

    # Generator
    rand = Generator(PCG64())
    # Jumps
    jump_times = t0 + (T-t0)*rand.random(rand.poisson(j_alpha*(T-t0),))  # The inner random generates the event count and the outer distributes them in time geometrically
    jump_times.sort()
    jump_intensitys = [rand.exponential(gamma) for a in jump_times] # Draws exponential intensities for each of the jumps
    jumps = list(zip(jump_times,jump_intensitys)) # A list of tuples of ordered times and intensities

    # Local Patch to t_s
    t_s = t_s_base.copy()
    t_s = np.concatenate((t_s,jump_times))
    t_s.sort()

    # Generating Noise
    gen_noise = rand.multivariate_normal([0,0],[[1, rho], [rho, 1]],(len(t_s)))
    Z_gen = gen_noise[:,0]
    W_gen = gen_noise[:,1]
    
    # Init Lambdas and r
    lambdas = np.linspace(lambda0, lambda0,len(t_s))
    r = np.linspace(r0, r0,len(t_s))
    r_ongrid = dict()   # A fast way to get r from t if it's on the established time grid
    lambda_ongrid = dict()

    for i in np.arange(1,len(t_s)):
        dt = t_s[i]-t_s[i-1]
        dW = np.sqrt(dt)*W_gen[i]
        dZ = np.sqrt(dt)*Z_gen[i]

        # Generate interest rate
        r[i] = r[i-1] + (theta-alpha*r[i-1])*dt + sigma*dW
        r_ongrid[hash(t_s[i])] = r[i]

        # Generate JCIR 
        lambdas[i] = lambdas[i-1] + kappa*(mu-lambdas[i-1])*dt + v*np.sqrt(lambdas[i-1])*dZ

        # Emergency negative catch
        if lambdas[i] < 0:
            lambdas[i] = 0

        if not not jumps:
            if t_s[i] == jumps[0][0]:
                lambdas[i] = lambdas[i] + jumps[0][1]
                jumps.pop(0)
        lambda_ongrid[hash(t_s[i])] = lambdas[i]
        
    return (t_s,r,lambdas,r_ongrid,lambda_ongrid)