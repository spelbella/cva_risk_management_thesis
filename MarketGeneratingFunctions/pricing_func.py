import numpy as np
from numpy.random import Generator, PCG64
import random
from scipy.stats import norm

class PricingFunc():
    def __init__(self, params, meta_cache, t_s, r, r_ongrid, lambdas, lambda_ongrid):
        # Absorb Params, we don't need all of these sometimes but sometimes we do.
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

        self.t_s = t_s
        self.r = r
        self.r_ongrid = r_ongrid
        self.lambdas = lambdas
        self.lambda_ongrid = lambda_ongrid

        # Define Caches for potentially re-used objects
        self.meta_cache = meta_cache
        self.P_cache = dict()
        self.ZCBV_cache = dict()
        self.Swap_cache = dict()
        self.Swaption_cache = dict()
        self.Q_cache = dict()
        self.h = (self.kappa**2 + 2*self.v**2)**(1/2)

        self.i = 0
        return

    """ Define all of the pricing funcs with caching """
    # Basic Functions
    def A(self,t,T):
        return self.meta_cache.A(t,T)
    def B(self,t,T):
        return self.meta_cache.B(t,T)
    def Khat(self,t,T,rstar):
        return np.e**(self.A(t,T) + self.B(t,T)*rstar)
    def P(self,t,T,rt):
        key = hash((round(t,10),round(T,10),round(rt,10)))
        if key in self.P_cache:
            P_ret = self.P_cache[key]
        else:
            P_ret = np.e**(self.A(t,T)+self.B(t,T)*rt)
            self.P_cache[key] = P_ret
        return P_ret
    
    # Put on ZCB functions
    def Fn(self,arg):
        return norm.cdf(arg)
    def mu_v(self,t,T,r0):
        return r0*np.e**(-self.alpha*(T - t)) + (self.theta/self.alpha)*(1 - np.e**(-self.alpha*(T - t))) + (self.sigma**2)*(1/self.alpha)*((1/(2*self.alpha))*(1 - np.e**(-2*self.alpha*(T-t))) - (1/self.alpha)*(1 - np.e**(-self.alpha*(T-t))))
    def v_r2(self,t,T):
        return (self.sigma**2)/(2*self.alpha)*(1 - np.e**(-2*self.alpha*(T-t)))
    def B_bar(self,tau):
        return self.B(0,tau)
    def A_bar(self,tau):
        return self.A(0,tau)
    def Khat2(self,tau,K):
        return K*np.e**(-self.A_bar(tau))
    def a(self,t,T,tau,K,r0):
        return (np.log(self.Khat2(tau,K)) - self.B_bar(tau)*self.mu_v(t,T,r0))/(self.B_bar(tau)*(self.v_r2(t,T))**(1/2))

    def d1(self,t,T,tau,K,r0):
        return self.a(t,T,tau,K,r0) - self.B_bar(tau)*(self.v_r2(t,T)**(1/2))
    def d2(self,t,T,tau,K,r0):
        return self.d1(t,T,tau,K,r0) + self.B_bar(tau)*(self.v_r2(t,T)**(1/2))

    # Value of a put option on a ZCB, option expiry T before or at payout date of ZCB T + tau, for call pass type = 0 for put type = 1
    def VZCB(self,t,T,tau,K,type):
        key = hash((round(t,10),round(T,10),round(tau,10),round(K,10),type))
        rt = self.r_time(t)
        if key in self.ZCBV_cache:
            val = self.ZCBV_cache[key]
        else:
            term1 =  np.exp((0.5*self.B_bar(tau)**2*self.v_r2(t,T) + self.B_bar(tau)*self.mu_v(t,T,rt)))*self.Fn(self.d1(t,T,tau,K,rt)) - self.Khat2(tau,K)*self.Fn(self.d2(t,T,tau,K,rt))
            call_value = self.P(t,T,rt)*np.exp(self.A_bar(tau))*term1 
            val = call_value + type*(-self.P(t,T+tau,rt) + K*self.P(t,T,rt))
            self.ZCBV_cache[key] = val
        return val
    
    # Finding RSTAR
    def cashflows(self, K, Ts):
        return (K*np.diff(Ts) + np.concatenate((np.zeros((1,len(Ts)-2)), np.asmatrix(1)),1)).A1
    def rstar(self,T_s,K):
        return self.meta_cache.rstar(T_s,K)
    
    # Finding interest rate from time
    def r_time(self,t):
        key = hash(round(t,10))
        if key in self.r_ongrid:
            r = self.r_ongrid[key]
        else:
            r = np.interp(t,self.t_s,self.r)
        return r 
    
    # Swap pricing
    def swap_price(self,t,T_s,K):
        # Catch passed swap
        if T_s[-1] <= t:
            return 0
        
        # Check Cache
        key = hash((round(t,10),str(T_s),round(K,10)))
        if key in self.Swap_cache:
            val = self.Swap_cache[key]
        else:
            T_s = T_s.copy()
            rt = self.r_time(t)

            # Strip away all but the first passed time, such that only future payments, and the last reset remain
            T_s = np.append(T_s,np.inf)
            T_s = [T_s[i] for i in np.arange(0,len(T_s)-1) if T_s[i+1] > t]

        
            # Locked in portion, ie. if we are in the first swap bucket after the strip above
            set_payment_value = 0
            interest_rate_locked = 0
            if t > T_s[0] and not len(T_s) <= 1:
                fix_date = T_s[0]
                payment_date = T_s[1]
                tau = payment_date - fix_date
                interest_rate_locked = self.r_time(fix_date) 
                set_payment_value = tau*(interest_rate_locked - K)*self.P(t,payment_date,rt)
                T_s = T_s[1:]

            # Future portion
            future_value = 0
            if not len(T_s) <= 1:
                future_value = self.P(t,T_s[0],rt) - self.P(t,T_s[-1],rt) - K*sum([tau_k*self.P(t,T_k,rt) for tau_k, T_k in zip(np.diff(T_s),T_s[1:])])
            val = (future_value + set_payment_value)
            self.Swap_cache[key] = val
        return val

    # Swaption Pricing
    def swaption_price(self, t, T_s, K):
        rt = self.r_time(t)
        """Sloppy Caching Code Begins"""
        key = hash((round(t,10),round(rt,10),str(T_s),round(K,10)))
        if key in self.Swaption_cache:
            return self.Swaption_cache[key]
        """Sloppy Caching Code Ends"""
        

        T_m = T_s[0] # The entrance date
        if (t>T_m or len(T_s)<2): # If the expiry has passed your swaption should be worthless / Expired
            value = 0 
            return value

        # Otherwise we use the formula as written
        dates = T_s[1:] # The payment dates
        cashs = self.cashflows(K, T_s) # The Cashflows

        rst = self.rstar(T_s, K) # Rstar 
        value = (sum([c*self.VZCB(t, T_m, Tk-T_m, self.Khat(T_m,Tk,rst), 1) for c, Tk in list(zip(cashs,dates))]))

        self.Swaption_cache[key] = value
        return value
    
    # Caplet Pricing
    def caplet_price(self, t, rt,T_m,T_n,K):
        value = 0
        # First we check if the caplet is already locked in 
        if T_m <= t:
            r_0 = self.r_ongrid(T_m)
            valueatexpiry = max((0,(r_0-K)))
            value =  (self.P(t,T_n,rt)*valueatexpiry)
        # Otherwise use caplet formula for HW model see page 76 of Interest rate models by Brigo
        else:
            sigma_p = lambda t,T: self.sigma*np.sqrt((1 - np.exp(-2*self.alpha*(T - t)))/(2*self.alpha))*self.P(t,T,rt)
            h = lambda t,T,S,X: (1/sigma_p(t,T))*np.log((self.P(t,S,rt))/(self.P(t,T,rt)*X)) + sigma_p(t,T)/2
            ZPB = lambda t,T,S,X: X*self.P(t,T,rt)*norm.cdf(-h(t,T,S,X) + sigma_p(t,T)) - self.P(t,S,rt)*norm.cdf(-h(t,T,S,X))
 
            X = 1/(1 + K*(T_n-T_m))
            value = (1 + K*(T_n-T_m))*ZPB(t,T_m,T_n,X)
        return value
    
    # Now we find Q(T > tau | F_t)
    def B_cir(self, t,T):
        return (2*(np.exp((T-t)*self.h)-1))/(2*self.h + (self.kappa + self.h)*(np.exp((T-t)*self.h)-1))
    def A_cir(self,t,T):
        return ((2*self.h*np.exp((self.kappa + self.h)*(T - t)/2))/(2*self.h + (self.kappa + self.h)*(np.exp((T-t)*self.h) -1)))**(2*self.kappa*self.mu/(self.v**2))
    def beta_hat(self,t,T):
        return self.B_cir(t,T)
    def alpha_hat(self,t,T):
        return self.A_cir(t,T)*((2*self.h*np.exp((self.h + self.kappa + 2*self.gamma)/(2) * (T-t)))/(2*self.h + (self.kappa + self.h + 2*self.gamma)*(np.exp(self.h*(T-t))-1)))**((2*self.j_alpha*self.gamma)/(self.v**2 - 2*self.kappa*self.gamma - 2*self.gamma**2))
    def lambda_from_t(self,t):
        if hash(round(t,10)) in self.lambda_ongrid:
            return self.lambda_ongrid[hash(round(t,10))]
        else:
            return np.interp(t,self.t_s,self.lambdas)
    def Q(self,t,T):
        key = hash((round(t,10),round(T,10)))
        if key in self.Q_cache:
            Q_val = self.Q_cache[key]
        else:
            lambd = self.lambda_from_t(t)
            Q_val = self.alpha_hat(t,T)*np.exp(-self.beta_hat(t,T)*lambd) *(T>t) + (t>=T)
            self.Q_cache[key] = Q_val
        return Q_val
    
    # Finally we can find the CVA
    # CVA deserves a full func, since there is some efficiency to be won, and it's both critical and hard to test
    def CVA(self, t, T_s, K, adj = False):
        # Args:
        # t, the current time/the filtration time
        # rt, the current interest rate
        # T_s, the key dates of the swap
        # K, the fixed leg, you pay diff(T_s)*K as a fixed leg
        # lambd, the current probability of default, the rest of the factors needed to find Q come from the model which Q already "knows"
        # Returns: 
        # The CVA estimate, a float
        lambd = self.lambda_from_t(t)

        # First lets rewrite the T_s vector into a T_s vector where all the payments are actually still in the future, and it starts at the last passed adjustment date
        T_s_local = T_s.copy()
        T_s_local = np.append(T_s_local,np.inf)
        T_s_local = [T_s_local[i] for i in range(len(T_s_local)-1) if T_s_local[i+1] > t]

        # Then we find the probability of default before each payment given that we haven't defaulted up to now, allowing the probability to default in the first bucket to extend to t instead of T
        T_s_proxy = T_s_local.copy()
        if t > T_s_local[0]:
            T_s_proxy = T_s_local[1:]
            T_s_proxy[0] = t
        default_after = [self.Q(t,T_s_proxy[k]) for k in range(0,len(T_s_proxy),1)]
        default_buckets = [default_after[k-1] - default_after[k] for k in range(1,len(T_s_proxy))]
        
        # Finding hedging swaptions
        hedging_swaptions = [self.swaption_price(t, T_s_local[k:], K) for k in range(1,len(T_s_local)-1,1)]
        self.i = self.i+1
        if adj:
            hedging_swaptions.append(self.caplet_price(t,T_s_local[0],T_s_local[1],K))
            default_buckets.append(1-self.Q(t,T_s_local[1],lambd))

        return sum([buc*hedg for buc, hedg in zip(default_buckets,hedging_swaptions)])
    
    # And that's all Folks!