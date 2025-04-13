import gymnasium as gym
from gymnasium import wrappers as wrap
import numpy as np
import scipy as sp

from path_datatype import Path

class tradingEng(gym.Env):
    def __init__(self, paths):
        # The paths
        self.paths = paths
        self.pthIDX = 0
        self.npths = len(self.paths)

        # The Observation space, for now let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (37 actions)
        #self.observation_space = gym.spaces.Box(low = -5*np.ones(37),high = 5*np.ones(37), dtype=np.float32)
        scaleo = 10
        scalea = 10 #0.5

        lower = np.asarray([0, -scaleo , 0])
        upper = np.asarray([scaleo, scaleo, scaleo])
        self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=float)

        # action space
        lowera = np.asarray([-scalea, -scalea])
        uppera = np.asarray([scalea, scalea])
        self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=float)

        self._agent_position = dict()
        self._agent_position_old = dict()
        self._agent_price = dict()
        self._agent_price_old = dict()
  
        self.reset()

    def reset(self, seed = None, options = None):
        self.tIDX = 0
        self.pthIDX = self.pthIDX + 1
        if self.pthIDX >= self.npths:
            self.pthIDX = 0
        self.currpth =  self.paths[self.pthIDX]

        # Initialize the portfolio to some ratio
        init_ratio  = np.ones(18)*(1/18)    # the initial value distribution, should probably be a delta hedge
        action = self.vec_to_dict_obs(init_ratio)
        self._agent_position = action
        self._agent_price = self.prices()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    # Constructing observations
    def _get_obs(self):
        return np.concatenate([
                    np.ones(1)*(np.float64(self.currpth.lambdas[self.tIDX])),
                    np.ones(1)*(np.float64(self.currpth.r[self.tIDX])),
                    np.ones(1)*(np.float64(self.currpth.t_s[self.tIDX]))
                ])

    def prices(self):
        swa = [self.currpth.Swaptions[i][self.tIDX]for i in range(1,10)]
        qs = [self.currpth.Q_s[i][self.tIDX]for i in range(1,10)]
        return dict({"Swaption Price" : swa, "Q Price" : qs})  
    
    def PnL(self):
        if self.tIDX == 0:
            return np.nan
        else:
            h = np.concatenate((self._agent_position["Swaption Position"],self._agent_position["Q Position"]))
            return self.currpth.CVA[self.tIDX]-self.currpth.CVA[self.tIDX-1] - np.dot(h,np.concatenate((self._agent_price["Swaption Price"],self._agent_price["Q Price"]))-np.concatenate((self._agent_price_old["Swaption Price"],self._agent_price_old["Q Price"])))

    def _get_info(self):
        return{
            "SqrdP&L": self.PnL()**2,
            "P&L": self.PnL(),
            "CVA": self.currpth.CVA[self.tIDX]
        }
    
    def vec_to_dict_obs(self, action):
        if not len(action) == 18:
            raise ValueError('You passed an action of incorrect length to the enivironment, it should be 18 long in a normal numeric format like a list or ndarray')
        swapts = action[0:9]
        Qs = action[9:18]
        return dict({"Swaption Position" : swapts, "Q Position" : Qs})
    
    def posValue(self):
        swaptions_val = np.linalg.norm(np.inner(self._agent_position["Swaption Position"],self._agent_price["Swaption Price"]))
        Q_val = np.linalg.norm(np.inner(self._agent_position["Q Position"],self._agent_price["Q Price"]))
        return Q_val + swaptions_val


    # The meat and potatoes
    def step(self, action):

        # Format action and try to avoid sideeffects
        actionl = action.copy()
        if not isinstance(actionl, dict):
            swpt = actionl[0]
            Q = actionl[1]
            actionl = np.concatenate([np.zeros(8), [swpt], np.zeros(8), [Q]])
            actionl = self.vec_to_dict_obs(actionl)
    
        # Step Time forward
        self._agent_position_old = self._agent_position
        self._agent_price_old = self._agent_price
        self._agent_position = actionl


        # Find Cost of Hedge and old CVA
        cost = self.posValue()
        oCVA = self.currpth.CVA[self.tIDX]

        # Step Time forward
        self.tIDX = self.tIDX + 1 

        self._agent_price = self.prices()    

        # Find Value of Hedge and new CVA
        value = self.posValue()
        nCVA = self.currpth.CVA[self.tIDX]
        
        dHedge = nCVA - oCVA
        dCVA = value - cost
        reward = -((dHedge - dCVA)**2  + np.abs(dHedge - dCVA))* 1000
        
        # # Observe the reward, which comes from the previous action
        #reward = -self.PnL()**2# * 1e6
        info = self._get_info()
        observation = self._get_obs()
  
        # End the environment after we reach year 9
        terminated = bool(self.currpth.t_s[self.tIDX] > 0.2)
        truncated = False
        if terminated:
            self.reset()

        return observation, reward, terminated, truncated, info
















class tradingEngOrg(gym.Env):
    def __init__(self, paths):
        # The paths
        self.paths = paths
        self.pthIDX = 0
        self.npths = len(self.paths)

        # The Observation space, for now let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (37 actions)
        #self.observation_space = gym.spaces.Box(low = -5*np.ones(37),high = 5*np.ones(37), dtype=np.float32)

        lower = np.concatenate([np.zeros(36), -np.inf*np.ones(1)])
        upper = np.concatenate([np.ones(18),np.inf*np.ones(9),np.ones(9),np.inf*np.ones(1)])
        self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=np.float32)

        # The action space, let's let the action space be to take a new position -> 18 dim
        lowera = np.concatenate([-1*np.ones(18)])
        ##lowera = np.concatenate([np.zeros(18)])
        uppera = np.concatenate([1*np.ones(18)])
        self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=np.float32)

        self.rewards = []

    def reset(self, seed = None, options = None):
        self.tIDX = 0
        self.pthIDX = self.pthIDX + 1
        if self.pthIDX >= self.npths:
            self.pthIDX = 0
        #print("reset was called", self.pthIDX)
        self.currpth =  self.paths[self.pthIDX]
        # Tracks the last held position
        self._agent_position = dict()
        self._agent_position_old = dict()
        self._agent_price = dict()
        self._agent_price_old = dict()
        CVA_at_t0 = self.currpth.CVA[0]

        # Initialize the portfolio to some ratio
        init_ratio  = np.ones(18)*(1/18)    # the initial value distribution, should probably be a delta hedge
        action = self.vec_to_dict(init_ratio)
        self._agent_position = action
        self._agent_position_old = action
        self._agent_price = self.prices()
        self._agent_price_old = self._agent_price
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    # Constructing observations
    def _get_obs(self):
        return np.concatenate([
                    self._agent_position["Swaption Position"],
                    self._agent_position["Q Position"],
                    self._agent_price["Swaption Price"],
                    self._agent_price["Q Price"],
                    np.ones(1)*(np.float64(self.currpth.r[self.tIDX]))
                ])

    def prices(self):
        swa = [self.currpth.Swaptions[i][self.tIDX] for i in range(0,9)] #ÄNDRA??
        qs = [self.currpth.Q_s[i][self.tIDX] for i in range(0,9)] #ÄNDRA??
        return dict({"Swaption Price" : swa, "Q Price" : qs})  
    
    def PnL(self):
        if self.tIDX == 0:
            return np.nan
        else:
            h = np.concatenate((self._agent_position_old["Swaption Position"],self._agent_position_old["Q Position"]))
            return self.currpth.CVA[self.tIDX]-self.currpth.CVA[self.tIDX-1] - np.dot(h,np.concatenate((self._agent_price["Swaption Price"],self._agent_price["Q Price"]))-np.concatenate((self._agent_price_old["Swaption Price"],self._agent_price_old["Q Price"])))

    def _get_info(self):
        return{
            "SqrdP&L": self.PnL()**2,
            "P&L": self.PnL(),
            "CVA": self.currpth.CVA[self.tIDX]
        }
    
    # Convert a non dict action into a dict
    def vec_to_dict(self, action):
        if not len(action) == 18:
            raise ValueError('You passed an action of incorrect length to the enivironment, it should be 18 long in a normal numeric format like a list or ndarray')
        swapts = action[0:9]
        Qs = action[9:18]
        return dict({"Swaption Position" : swapts, "Q Position" : Qs})
    
    def get_original_reward(self):
        return self.rewards

    # The meat and potatoes
    def step(self, action):
        # Format action and try to avoid sideeffects
        actionl = action.copy()/1000000
        if not isinstance(actionl, dict):
            actionl = self.vec_to_dict(actionl)
    
        # Step Time forward
        self.tIDX = int(self.tIDX + 1)
        self._agent_position_old = self._agent_position
        self._agent_price_old = self._agent_price
        self._agent_position = actionl
        self._agent_price = self.prices()
        

        # Observe the reward, which comes from the previous action
        reward = -self.PnL()**2# * 1e6
        self.rewards.append(reward)
        info = self._get_info()
        observation = self._get_obs()
  
        # End the environment after we reach year 9
        terminated = bool(self.currpth.t_s[self.tIDX] > 9)
        truncated = False
        # if terminated:
        #     self.reset()

        return observation, reward, terminated, truncated, info