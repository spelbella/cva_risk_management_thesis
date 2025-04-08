import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import warnings
from bisect import bisect

class Global_Cache:
    def __init__(self, t_s, T_s, params):
        # Store Params
        self.t0 = params["t0"]
        self.T = params["T"]
        self.gamma = params["gamma"]
        self.j_alpha = params["j_alpha"]
        self.rho = params["rho"]
        self.lambda0 = params["lambda0"]
        self.r0 = params["r0"]
        self.theta = params["theta"]
        self.alpha = params["alpha"]
        self.sigma = params["sigma"]
        self.kappa = params["kappa"]
        self.mu = params["mu"]
        self.v = params["v"]

        # Initialize Caches
        self.A_cache = dict()
        self.B_cache = dict()
        self.rstar_cache = dict()

        P = lambda t,T,rt: np.e**(self.A_naive(t,T)+self.B_naive(t,T)*rt)
        self.K = (P(0,T_s[0],self.r0) - P(0,T_s[-1],self.r0))/(sum([P(0,T_s[i],self.r0) for i in range(1,len(T_s))]))

        # Fill the Cache 
        for t in t_s.astype(float):
            for T in T_s:
                T = T.astype(float)
                # Fill the A_cache
                keyA = self.A_key(t,T)
                self.A_cache[keyA] = self.A_naive(t,T)

                # Fill the B_cache
                keyB = self.B_key(t,T)
                self.B_cache[keyB] = self.B_naive(t,T)

        for i in range(0,len(T_s)-1):
            # Fill the rstar cache
            keyR = self.rstar_key(T_s[i:],self.K)
            self.rstar_cache[keyR] = self.rstar_naive(T_s[i:],self.K)
           
    def A(self,t,T):
        key = self.A_key(t,T)
        if key in self.A_cache:
            A = self.A_cache[key]
        else:
            A = self.A_naive(t,T)
            #self.A_cache[key] = A
        return A
    
    def B(self,t,T):
        key = self.B_key(t,T)
        if key in self.B_cache:
            B = self.B_cache[key]
        else:
            B = self.B_naive(t,T)
            #self.B_cache[key] = B
        return B
    
    def rstar(self,T_s,K):
        key = self.rstar_key(T_s,K)
        if key in self.rstar_cache:
            r_star = self.rstar_cache[key]
        else:
            r_star = self.rstar_naive(T_s,K)
            #self.rstar_cache[key] = r_star
        return r_star

    def A_key(self,t,T):
        key = hash((t,float(T)))#str(np.round(t,15)) + ',' + str(np.round(T,15))
        return key
    
    def B_key(self,t,T):
        key = hash((t,float(T)))#str(np.round(t,15)) + ',' + str(np.round(T,15))
        return key
        
    def rstar_key(self,T_s,K):
        key = str(T_s) + ',' + str(np.round(K,15)) # hash((str(np.round(T_s,15)),np.round(K,15))) #
        return key
    

    def A_naive(self,t,T):
        A = -(self.sigma**2)/(4*self.alpha**3) * (3 + np.e**(-2*self.alpha*(T-t)) - 4*np.e**(-self.alpha*(T-t)) - 2*self.alpha*(T-t)) - (self.theta/self.alpha)*((T-t) - (1/self.alpha)*(1 - np.e**(-self.alpha*(T-t))))
        return A

    def B_naive(self,t,T):
        B = -(1/self.alpha)*(1 - np.e**(-self.alpha*(T-t)))
        return B

    def rstar_naive(self, T_s, K):
        minimize = lambda cashflows, dates, Tm, rstar: 1 - sum([c*np.e**(self.A_naive(Tm,date) + self.B_naive(Tm,date)*rstar) for c, date in zip(cashflows,dates)])
        cashflows = (K*np.diff(T_s) + np.concatenate((np.zeros((1,len(T_s)-2)), np.asmatrix(1)),1)).A1
        dates = T_s[1:]
        Tm = T_s[0]
        optim = lambda rstarl: minimize(cashflows, dates, Tm, rstarl)
        rstar = optimize.newton(optim, 0)
        return rstar
    

class Global_Cache_HW:
    def __init__(self, t_s, T_s, params, calib_data):
        # Store Params
        self.t0 = params["t0"]
        self.T = params["T"]
        self.gamma = params["gamma"]
        self.j_alpha = params["j_alpha"]
        self.rho = params["rho"]
        self.lambda0 = params["lambda0"]
        self.r0 = params["r0"]
        self.alpha = params["alpha"]
        self.sigma = params["sigma"]
        self.kappa = params["kappa"]
        self.mu = params["mu"]
        self.v = params["v"]
        self.t_s = t_s
        self.T_s = T_s
        self.params = params
        
        self.thetas = []
        self.theta_grid = []

        self.calib_data = calib_data

        ti = calib_data["ti"]
        Pi = calib_data["Pi"]
        self.interpolator =  interpolate.splrep(ti, Pi, s=0.0001)

        self.P0T_cache = dict()
        self.f0T_cache = dict()
        self.theta_cache = dict()
        self.A_cache = dict()
        self.Khat_cache = dict()
        self.rstar_cache = dict()
      
        # Fill the Cache 
        for t in t_s:
            # Fill theta and P0T Cache these should intuitively be done before theta but the order doesn't actually matter
            self.P0T(t,startup = True)
            self.f0T(t,startup = True)

        # Generate combined timesteps and expiry timetable, this ensures that things are found on timestep which is sometimes neesecary 
        ext_time = np.append(t_s,T_s)
        # Fill the theta Cache, in the current implentation this *needs* to be done before A since A naively asks for interpolated thetas, this can be changed by setting startup = True in the theta call duing A startup
        for i in range(len(ext_time)):
            for i in range(10):
                t_subtick = t_s[i] + i*(ext_time[i+1]-ext_time[i])
                self.theta(t_subtick,1e-4,startup = True)

        # Fill the A cache, this should be exhaustive as long as we only price on the standard grid and not jump added grid points
        for t in ext_time:
            for T in T_s:
                # Fill the A_cache
                self.A(t,T, startup = True)

        # Now we can find K, this has to be after theta since it calls A, see the comment on A for how to change that behaivour
        P = lambda t,T,rt: np.e**(self.A(t,T,startup = True)+self.B(t,T)*rt)       
        self.K = (P(0,T_s[0],self.r0) - P(0,T_s[-1],self.r0))/(sum([P(0,T_s[i],self.r0) for i in range(1,len(T_s))]))

        # Fill the rstar cache, this should always be exhaustive
        for i in range(0,len(T_s)-1):
            # Fill the rstar cache
            self.rstar(T_s[i:],self.K, startup = True)

        # Fill Khat Cache, this should be relatively exhaustive but I could have missed some cases
        for t in ext_time:
            for i in range(0,len(T_s)-1):
                for m in T_s:
                    if m >= t:
                        self.Khat(t,m, self.rstar(T_s[i:],self.K, startup = True), startup = True)

    def P0T(self,T, startup = False):
        key = hash(T)
        if key in self.P0T_cache:
            ret = self.P0T_cache[key]
        elif startup:
            ret = interpolate.splev(T, self.interpolator, der=0)
            self.P0T_cache[key] = ret
        else:
            ret = interpolate.splev(T, self.interpolator, der=0)
        return ret   

    def f0T(self,t, dt = 1e-4, startup = False):
        key = np.round(t,15)
        if key in self.f0T_cache:
            ret = self.f0T_cache[key]
        elif startup:
            ret = - (np.log(self.P0T(t+dt, startup = True))-np.log(self.P0T(t-dt, startup = True)))/(2*dt)
            self.f0T_cache[key] = ret
        else:
            ret = - (np.log(self.P0T(t+dt))-np.log(self.P0T(t-dt)))/(2*dt)
        return ret

    def theta(self, t, dt = 1e-4, startup = False):
        key = np.round(t,15) 
        if key in self.theta_cache:
            ret = self.theta_cache[key]
        elif startup:
            ret = (self.f0T(t+dt, startup = True)-self.f0T(t-dt, startup = True))/(2.0*dt) + self.f0T(t, startup = True) + self.sigma**2/(2.0*self.alpha)*(1.0-np.exp(-2.0*self.alpha*t))
            self.theta_cache[key] = ret
            l_idx = bisect(self.theta_grid, t) # Find out index to pass to insert
            self.theta_grid.insert(l_idx,t)
            self.thetas.insert(l_idx,ret)
        else:
            ret = np.interp(t,self.theta_grid,self.thetas)
        return ret
    
    def A(self, t, T, stp_density = 50, startup = False):
        int_steps = min(max(10,round(stp_density*(T-t))),100)
        key = hash((np.round(t,15),float(np.round(T,15)),stp_density))
        if key in self.A_cache:
            ret = self.A_cache[key]
        elif startup:
            pt_simple = -(self.sigma**2)/(4*self.alpha**3) * (3 + np.e**(-2*self.alpha*(T-t)) - 4*np.e**(-self.alpha*(T-t)) - 2*self.alpha*(T-t))
            integrand = [self.theta(z)*self.B(z,T) for z in np.linspace(t, T, int_steps)]
            pt_integral =  integrate.trapezoid(integrand, x=np.linspace(t, T, int_steps))  
            ret = pt_simple + pt_integral
            self.A_cache[key] = ret
        else:
            pt_simple = -(self.sigma**2)/(4*self.alpha**3) * (3 + np.e**(-2*self.alpha*(T-t)) - 4*np.e**(-self.alpha*(T-t)) - 2*self.alpha*(T-t))
            integrand = [self.theta(z)*self.B(z,T) for z in np.linspace(t, T, int_steps)]
            pt_integral =  integrate.trapezoid(integrand, dx = (T-t)/int_steps)  
            ret = pt_simple + pt_integral
            print("Uncached A")
        return ret
    
    def B(self, t, T):
        ret = -(1/self.alpha)*(1 - np.e**(-self.alpha*(T-t)))
        return ret
    
    def Khat(self,t,T, rstar, startup = False):
        key = ((float(np.round(t,15)),float(np.round(T,15)),np.round(rstar,15)))
        if key in self.Khat_cache:
            ret = self.Khat_cache[key]
        elif startup:
            ret = np.e**(self.A(t,T, startup = True) + self.B(t,T)*rstar)
            self.Khat_cache[key] = ret
        else:
            ret = np.e**(self.A(t,T) + self.B(t,T)*rstar)
            print("Uncached Khat")
            print(key)
        return ret
    
    def rstar(self,T_s,K, startup = False):
        key = (T_s[0], T_s[-1],K)
        if key in self.rstar_cache:
            ret = self.rstar_cache[key]
        elif startup:  
            # Solve for rstar
            minimize = lambda cashflows, dates, Tm, rstar: 1 - sum([c*np.e**(self.A(Tm,date, startup = True) + self.B(Tm,date)*rstar) for c, date in zip(cashflows,dates)])
            cashflows = (K*np.diff(T_s) + np.concatenate((np.zeros((1,len(T_s)-2)), np.asmatrix(1)),1)).A1
            Tm = T_s[0]
            dates = T_s[1:]
            optim = lambda rstarl: minimize(cashflows, dates, Tm, rstarl)
            ret = optimize.newton(optim, 0)
            self.rstar_cache[key] = ret
        else:
            # Solve for rstar
            minimize = lambda cashflows, dates, Tm, rstar: 1 - sum([c*np.e**(self.A(Tm,date) + self.B(Tm,date)*rstar) for c, date in zip(cashflows,dates)])
            cashflows = (K*np.diff(T_s) + np.concatenate((np.zeros((1,len(T_s)-2)), np.asmatrix(1)),1)).A1
            Tm = T_s[0]
            dates = T_s[1:]
            optim = lambda rstarl: minimize(cashflows, dates, Tm, rstarl)
            ret = optimize.newton(optim, 0)
            print("Uncached rstar")
        return ret
    