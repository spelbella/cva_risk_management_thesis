import gymnasium as gym
from gymnasium import wrappers as wrap
import numpy as np
import random
import torch as th

import path_datatype

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from MarketGeneratingFunctions import DeltaHedge

class tradingEng(gym.Env):
    def __init__(self, paths, action = 'small-More-Trust', obs = 'big', reward = 'L1', rewardscale = 1, huberBeta = 0.1):
        # The paths
        self.paths = paths.copy()
        random.shuffle(self.paths)

        # The currently looked at path indx
        self.pthIDX = 0
        self.currpth =  paths[self.pthIDX]
        self.npths = len(paths)

        # The action and obs space
        self.action = action
        self.obs = obs

        # The Reward settings
        self.rewardf = reward
        self.reward_scale = rewardscale
        self.Huber = huberBeta

        # The time index
        self.tIDX = 0

        # Tracks the last held position
        self._agent_position = dict()

        # The Observation space, 
        scaleo = 50
        scalebig = 1
        scalesmall = 1

        # Encoder 
        self.encoder = None

        match obs:
            case 'big':
                # let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (36 actions), 
                # order = [Actions, Swaptions, Qs, Interest Rate]
                lower = -1*np.ones(18)
                upper = 1*np.ones(18)
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)
            case 'big-nt':
                # let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (36 actions), 
                # order = [Actions, Swaptions, Qs, Interest Rate]
                lower = -1*np.ones(36)
                upper = 1*np.ones(36)
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)
            case 'small':
                # Just the interest rate, default intensity and previous action and time, order = [prev actions, intensity, interest, time]
                lower = np.concatenate([np.zeros(18),[0.00] , [-scaleo], [0.00]])
                upper = np.concatenate([np.ones(18), [scaleo], [scaleo], [scaleo]])
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)
            case 'xs':
                # Just the interest rate and default intensity and time, order = [intensity, interest, time]
                lower = np.asarray([-1, -1, -1])
                upper = np.asarray([1, 1, 1])
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)
            case 'auto':
                # Autoencoder based information, gives 3 dim out
                from AutoEncoder import MarketAutoencoder 
                import pickle
                with open("autoencoderT","rb") as fp:
                    self.encoder = pickle.load(fp)
                    print("Opened File ok")

                lower = np.asarray([-1, -1, -1])
                upper = np.asarray([1, 1, 1])
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)
            case 'none':
                # A dummy input with no information about the env to test best case of uninformed hedging
                lower = np.asarray([-1])
                upper = np.asarray([1])
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)            
            case 'xs-nt':
                # Just the interest rate and default intensity and time, order = [intensity, interest, time]
                lower = np.asarray([-1, -1, -1])
                upper = np.asarray([1, 1, 1])
                self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)
            case _:
                return "No observation space matching input"
        
        match action:
            case 'big':
                # The action space, let's let the action space be to take a new position -> 18 dim
                lowera = -scalebig*np.ones(18)
                uppera = scalebig*np.ones(18)
                self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=float)
            case 'small':
                # Just the swaption and q with last expiry
                lowera = np.asarray([-1, -1])
                uppera = np.asarray([1, 1])
                self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=float)
            case 'small-More-Trust':
                # Just the swaption and q with last expiry
                lowera = np.asarray([-1, -1])
                uppera = np.asarray([1, 1])
                self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=float)
            case _:
                return "No matching action space"

        self.reset()

    # Constructing observations
    def _get_obs(self):
        match self.obs:
            case 'big':
                return np.concat([
                    #self._agent_position["Swaption Position"]+1)) #Hovers around 0.01 a lot, add small offset to avoid log of -, pos def, low variance, so add 1, take log and take tanh,
                    #self._agent_position["Q Position"], #Bounded 0,1 to bounded -1, 1
                    np.tanh(np.log([a+1 for a in self.swaptions_now()])), # An NN output so let's not normalize
                    [(a - 0.5)*2 for a in self.Q_now()] # Also an NN output
                ])
            case 'big-nt':
                return np.concatenate([
                    self._agent_position["Swaption Position"], #Hovers around 0.01 a lot, add small offset to avoid log of -, pos def, low variance, so add 1, take log and take tanh,
                    self._agent_position["Q Position"], #Bounded 0,1 to bounded -1, 1
                    self.swaptions_now(), # An NN output so let's not normalize
                    self.Q_now(), # Also an NN output
                ])
            case 'small':
                return np.concatenate([
                    self._agent_position["Swaption Position"],
                    self._agent_position["Q Position"],
                    [self.currpth.lambdas[self.tIDX]],
                    [self.currpth.r[self.tIDX]],
                    [self.currpth.t_s[self.tIDX]]
                ])
            case 'xs': # Transformed Observations work best!, Need to test not transforming t or doing a different transform since time goes 0 to 10, maybe just divide by 5 - 1??
                return np.asmatrix([
                    np.tanh(np.log((self.currpth.lambdas[self.tIDX] - 0.001)+1)) if self.currpth.lambdas[self.tIDX] != -0.999 else -1,
                    np.tanh(self.currpth.r[self.tIDX]),
                    # Different TIme Norm
                    self.currpth.t_s[self.tIDX]/5 - 1,
                    #np.tanh(np.log(self.currpth.t_s[self.tIDX])) if self.currpth.t_s[self.tIDX] != 0 else -1
                ])
            case 'auto': # Transformed Observations work best!, Need to test not transforming t or doing a different transform since time goes 0 to 10, maybe just divide by 5 - 1??
                sample = np.concat([self.swaptions_now(), self.Q_now()])
                sample = self.encoder.preprocess(sample)
                z = self.encoder.encoder(th.from_numpy(sample))
                return z.detach().numpy()
            case 'none':
                return np.asmatrix([0])

            case 'xs-nt':
                return np.asmatrix([
                    self.currpth.lambdas[self.tIDX],
                    self.currpth.r[self.tIDX],
                    self.currpth.t_s[self.tIDX]
                ])
    
    def swaptions_now(self):
        try:
            return [self.currpth.Swaptions[i][self.tIDX] for i in range(1,10)]
        except:
            print(self.tIDX)
        finally:
            pass
    def Q_now(self):
        return [self.currpth.Q_s[i][self.tIDX] for i in range(1,10)]
    
    def posValue(self):
        swaptions_val = np.inner(self._agent_position["Swaption Position"],self.swaptions_now())
        Q_val = np.inner(self._agent_position["Q Position"],self.Q_now())
        return Q_val + swaptions_val
        
    def AposValue(self, action):
        swaptions_val = np.inner(action["Swaption Position"],self.swaptions_now())
        Q_val = np.inner(action["Q Position"],self.Q_now())
        return Q_val + swaptions_val
    
    def PnL(self):
        return self.posValue() - self.currpth.CVA[self.tIDX]

    def _get_info(self):
        return{
            "SqrdP&L": self.PnL()**2,
            "P&L": self.PnL(),
            "CVA": self.currpth.CVA[self.tIDX]
        }
    
    def reset(self, seed = None, options = None):
        self.tIDX = 0
        self.pthIDX = self.pthIDX + 1
        if self.pthIDX >= 125:
            self.pthIDX = 0
            random.shuffle(self.paths)
        self.currpth = self.paths[self.pthIDX]

        # Initialize the portfolio to some ratio
        init_ratio  = np.ones(18)*(1/18)    # the initial value distribution, should probably be a delta hedge
        action = self.vec_to_dict(init_ratio)
        self._agent_position = action

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    # Convert a non dict action into a dict
    def vec_to_dict(self, action):
        if not len(action) == 18:
            dumm = str(action)
            raise ValueError('You passed an action of incorrect length to the enivironment, it should be 18 long in a normal numeric format like a list or ndarray' + dumm)
        swapts = action[0:9]
        Qs = action[9:18]
        return dict({"Swaption Position" : swapts, "Q Position" : Qs})\

    def reward(self, diff):
        match self.rewardf:
            case 'L1':
                reward = (-1)* np.abs(diff) * self.reward_scale
            case 'L2':
                reward = -(diff**2 * self.reward_scale)
            case 'Huber':
                reward = -(int(diff >= self.Huber)*(1/2)*diff**2 + int(diff < self.Huber)*self.Huber*(np.abs(diff) - 1/2*self.Huber)) * self.reward_scale
            case 'PnL':
                reward = diff
            case '(Pnl)-':
                reward = np.min([diff,0.0]) 
            case _:
                reward = 0.0
        return reward


    # The meat and potatoes
    def step(self, action):
        # Format action and try to avoid sideeffects
        actionl = action.copy()
        
        match self.action:
            case 'big':
                actionl = actionl*0.1
            case 'small':
                swpt = actionl[0]
                Q = actionl[1]
                actionl = np.concatenate([np.zeros(8), [swpt], np.zeros(8), [Q]])
            case 'small-More-Trust':
                swpt = actionl[0]*2
                Q = actionl[1]*2
                actionl = np.concatenate([np.zeros(19), [swpt], np.zeros(19), [Q]])
            case 'small-Magnus':
                Q = [1.0] + self.Q_now()
                Swapts = self.swaptions_now()
                [SwapsHedge,Qhedge] = DeltaHedge.delta_hedge(Swapts,Q,np.arange(0,21),self.currpth.t_s[self.tIDX])
                Qhedge = Qhedge[1:]
                SwapsHedge = SwapsHedge*(actionl[2] + 1)
                SwapsHedge[-1] = SwapsHedge[-1] + actionl[0]
                Qhedge = Qhedge*(actionl[2] + 1)
                Qhedge[-1] = Qhedge[-1] + actionl[1]
                actionl = np.concatenate([SwapsHedge,Qhedge])
            case 'big-Magnus':
                Q = [1.0] + self.Q_now()
                Swapts = self.swaptions_now()
                [SwapsHedge,Qhedge] = DeltaHedge.delta_hedge(Swapts,Q,np.arange(0,21),self.currpth.t_s[self.tIDX])
                Qhedge = Qhedge[1:]
                SwapsHedge = [SwapsHedge[i]*(actionl[40] + 1) + actionl[i]*0.1 for i in range(0,20)]
                Qhedge = [Qhedge[i-20]*(actionl[40] + 1) + actionl[i]*0.1 for i in range(20,40)]
                actionl = np.concatenate([SwapsHedge,Qhedge])
        if not isinstance(actionl, dict):
            actionl = self.vec_to_dict(actionl)
        
        # Set the new state
        self._agent_position = actionl

        # Find Cost of Hedge and old CVA
        cost = self.posValue()
        oCVA = self.currpth.CVA[self.tIDX]

        # Step Time forward
        self.tIDX = self.tIDX + 1     

        # Find Value of Hedge and new CVA
        value = self.posValue()
        nCVA = self.currpth.CVA[self.tIDX]

        # Observe the reward, which comes from the previous action
        dHedge = nCVA - oCVA
        dCVA = value - cost
        diff = dHedge - dCVA
        #print(diff)
        
        # Since the diff is unbounded posdef we can take a log transform and a tanh to map to -1 to 1
        reward = self.reward(diff)

        info = self._get_info()
        observation = self._get_obs()
        #print(reward)

        # End the environment after we reach year 9
        terminated = self.currpth.t_s[self.tIDX] > 5
        truncated = False

        if terminated:
            #print(self.currpth.t_s[self.tIDX])
            self.reset()

        return observation, reward, terminated, truncated, info
