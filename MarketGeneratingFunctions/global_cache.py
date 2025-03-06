import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.integrate as integrate

class Global_Cache:
    def __init__(self, t_s, T_s, params, K):
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
            keyR = self.rstar_key(T_s[i:],K)
            self.rstar_cache[keyR] = self.rstar_naive(T_s[i:],K)
           
    def A(self,t,T):
        key = self.A_key(t,T)
        if key in self.A_cache:
            A = self.A_cache[key]
        else:
            A = self.A_naive(t,T)
        return A
    
    def B(self,t,T):
        key = self.B_key(t,T)
        if key in self.B_cache:
            B = self.B_cache[key]
        else:
            B = self.B_naive(t,T)
        return B
    
    def rstar(self,T_s,K):
        key = self.rstar_key(T_s,K)
        if key in self.rstar_cache:
            r_star = self.rstar_cache[key]
        else:
            r_star = self.rstar_naive(T_s,K)
            print(key)
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
    def __init__(self, t_s, T_s, params, calib_data, K):
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
        self.K = K
        self.params = params

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
            # Fill theta and P0T Cache
            self.theta(t)
            self.P0T(t)
            self.f0T(t)

            for T in T_s:
                # Fill the A_cache
                self.A(t,T)
                # Fill the B_cache
                self.B(t,T)

        for i in range(0,len(T_s)-1):
            # Fill the rstar cache
            self.rstar(T_s[i:],K)

    def P0T(self,T):
        if T in self.P0T_cache:
            ret = self.P0T_cache[T]
        else:
            ret = interpolate.splev(T, self.interpolator, der=0)
            self.P0T_cache[T] = ret
        return ret   

    def f0T(self,t, dt = 1e-8):
        if t in self.f0T_cache:
            ret = self.f0T_cache[t]
        else:
            ret = - (np.log(self.P0T(t+dt))-np.log(self.P0T(t-dt)))/(2*dt)
        return ret

    def theta(self, t, dt = 1e-8):
        if t in self.theta_cache:
            ret = self.theta_cache[t]
        else:
            ret = (self.f0T(t+dt)-self.f0T(t-dt))/(2.0*dt) + self.f0T(t) + self.sigma**2/(2.0*self.alpha)*(1.0-np.exp(-2.0*self.alpha*t))
            self.theta_cache[t] = ret
        return ret
    
    def A(self, t, T, int_steps = 100):
        key = hash((t,T,int_steps))
        if key in self.A_cache:
            ret = self.A_cache[key]
        else:
            pt_simple = -(self.sigma**2)/(4*self.alpha**3) * (3 + np.e**(-2*self.alpha*(T-t)) - 4*np.e**(-self.alpha*(T-t)) - 2*self.alpha*(T-t))
            pt_integral =  integrate.trapezoid([self.theta(z)*self.B(z,T) for z in np.linspace(t, T, int_steps)], x=np.linspace(t, T, int_steps))  
            ret = pt_simple + pt_integral
            self.A_cache[key] = ret
        return ret
    
    def B(self, t, T):
        ret = -(1/self.alpha)*(1 - np.e**(-self.alpha*(T-t)))
        return ret
    
    def Khat(self,t,T, rstar):
        key = hash((t,T))
        if key in self.Khat_cache:
            ret = self.Khat_cache[key]
        else:
            ret = np.e**(self.A(t,T) + self.B(t,T)*rstar)
            self.Khat_cache[key] = ret
        return ret
    
    def rstar(self,T_s,K):
        key = hash(((str(T_s)),K))
        if key in self.rstar_cache:
            ret = self.rstar_cache[key]
        else:  
            # Solve for rstar
            minimize = lambda cashflows, dates, Tm, rstar: 1 - sum([c*np.e**(self.A(Tm,date) + self.B(Tm,date)*rstar) for c, date in zip(cashflows,dates)])
            cashflows = (K*np.diff(T_s) + np.concatenate((np.zeros((1,len(T_s)-2)), np.asmatrix(1)),1)).A1
            Tm = T_s[0]
            dates = T_s[1:]
            optim = lambda rstarl: minimize(cashflows, dates, Tm, rstarl)
            ret = optimize.newton(optim, 0)
            self.rstar_cache[key] = ret
        return ret