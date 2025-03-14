import gymnasium as gym
from gymnasium import wrappers as wrap
import numpy as np
import scipy as sp

from path_datatype import Path

class tradingEng(gym.Env):
    def __init__(self, paths):
        # The paths
        self.paths = paths

        # The Observation space, for now let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (37 actions)
        lower = np.concatenate([np.zeros(36), -np.inf*np.ones(1)])
        upper = np.concatenate([np.ones(18),np.inf*np.ones(9),np.ones(9),np.inf*np.ones(1)])
        self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=np.float32)

        # The action space, let's let the action space be to take a new position -> 18 dim
        lowera = np.concatenate([np.zeros(18)])
        uppera = np.concatenate([np.ones(18)])
        self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=np.float32)

    def reset(self, seed = None, options = None):
        self.pthIDX = 0
        self.tIDX = 0
        self.currpth =  self.paths[self.pthIDX]
        self.npths = len(self.paths)
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
        swa = [self.currpth.Swaptions[i][self.tIDX] for i in range(1,10)]
        qs = [self.currpth.Q_s[i][self.tIDX] for i in range(1,10)]
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

    # Normalize a portfolio to have a consistent value 
    def norm_portfolio(self, action, value = None):
        if value == None:
            value = self.posValue()

        # Convert fraction of portfolio to amount of options with floating scale
        Q_values = self.Q_now()
        swap_values = self.swaptions_now()

        # Buy number of shares to match portfolio fraction of value
        for i in range(len(action["Swaption Position"])):
            if self.currpth.t_s[self.tIDX]-1.005> i:
                action["Swaption Position"][i] = action["Swaption Position"][i]/(max(swap_values[i],1e-16))
                action["Q Position"][i] = action["Q Position"][i]/(max(Q_values[i],1e-16))  
            else:        # Force zero position in expired positions
                action["Swaption Position"][i] = 0.0
                action["Q Position"][i] = 0.0    
        
        # Normalize Value / Enforce value restraints
        value_action = self.AposValue(action)
        #scale = value/value_action
        scale = value / value_action if value_action != 0 else 1.0 #ÄNDRAT

        scaled_action = dict({
             "Swaption Position" : action["Swaption Position"]*scale,
             "Q Position" : action["Q Position"]*scale,
        })

        return scaled_action

    # The meat and potatoes
    def step(self, action):
        # Format action and try to avoid sideeffects
        actionl = action.copy()
        if not isinstance(actionl, dict):
            actionl = self.vec_to_dict(actionl)
        
        # Step Time forward
        self.tIDX = int(self.tIDX + 1)  

        self._agent_position_old = self._agent_position
        self._agent_price_old = self._agent_price
        self._agent_position = actionl
        self._agent_price = self.prices()

        # Observe the reward, which comes from the previous action
        reward = -self.PnL()**2
        info = self._get_info()
        observation = self._get_obs()
  
        # End the environment after we reach year 9
        terminated = bool(self.currpth.t_s[self.tIDX] >= 10)
        truncated = False

        return observation, reward, terminated, truncated, info
    



class tradingEngOrg(gym.Env):
    def __init__(self, paths):
        # The paths
        self.paths = paths

        # The currently looked at path indx
        # self.pthIDX = 0 ÄNDRAT
        self.currpth =  paths[self.pthIDX]
        self.npths = len(paths)

        # The time index
        self.tIDX = 0

        # Tracks the last held position
        self._agent_position = dict()

        # The Observation space, for now let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (37 actions)
        lower = np.concatenate([np.zeros(36), -np.inf*np.ones(1)])
        upper = np.concatenate([np.ones(18),np.inf*np.ones(9),np.ones(9),np.inf*np.ones(1)])
        self.observation_space = gym.spaces.Box(low = lower,high = upper, dtype=np.float64)
        #self.observation_space = gym.spaces.Dict(
        #    {
        #        "Swaption Position" : gym.spaces.Box(low = 0, high = 1, shape=(9,), dtype=float),
        #        "Q Position" : gym.spaces.Box(low = 0, 1, shape=(9,), dtype=float),
        #        "Swaption Value" : gym.spaces.Box(low = 0, high = np.inf, shape=(9,), dtype=float),
        #        "Q Value" : gym.spaces.Box(low = 0, high = 1, shape=(9,), dtype=float),
        #        "r" : gym.spaces.Box(low = -np.inf, high = np.inf, shape=(1,), dtype=float),
        #    }
        #)

        # The action space, let's let the action space be to take a new position -> 18 dim
        lowera = np.concatenate([np.zeros(18)])
        uppera = np.concatenate([np.ones(18)])
        self.action_space = gym.spaces.Box(low = lowera,high = uppera, dtype=np.float64)
        #self.action_space = gym.spaces.Dict(
        #    {
        #        "Swaption Position" : gym.spaces.Box(lower = 0, upper = 1, shape=(9,), dtype=float),
        #        "Q Position" : gym.spaces.Box(lower = 0, upper = 1, shape=(9,), dtype=float),
        #    }
        #)

        #self.reset()

    # Constructing observations
    def _get_obs(self):
        return np.concatenate([
                    self._agent_position["Swaption Position"],
                    self._agent_position["Q Position"],
                    self.swaptions_now(),
                    self.Q_now(),
                    np.ones(1)*(np.float64(self.currpth.r[self.tIDX]))
                ])
        #return {
        #            "Swaption Position": self._agent_position["Swaption Position"],
        #            "Q Position": self._agent_position["Q Position"],
        #            "Swaption Value" : self.swaptions_now(),
        #            "Q Value" : self.Q_now(),
        #            "r" : self.currpth.r[self.tIDX]
        #        }
    
    def swaptions_now(self):
        #try:
        return [self.currpth.Swaptions[i][self.tIDX] for i in range(1,10)]
        #except:
        #    print(self.tIDX)
        #finally:
        #    pass
    def Q_now(self):
        return [self.currpth.Q_s[i][self.tIDX] for i in range(1,10)]
    
    def posValue(self):
        swaptions_val = np.linalg.norm(np.inner(self._agent_position["Swaption Position"],self.swaptions_now()))
        Q_val = np.linalg.norm(np.inner(self._agent_position["Q Position"],self.Q_now()))
        return Q_val + swaptions_val
    
    def AposValue(self, action):
        swaptions_val = np.linalg.norm(np.inner(action["Swaption Position"],self.swaptions_now()))
        Q_val = np.linalg.norm(np.inner(action["Q Position"],self.Q_now()))
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
        if self.pthIDX >= self.npths:
            self.pthIDX = 0
        self.currpth = self.paths[self.pthIDX]
        CVA_at_t0 = self.currpth.CVA[0]

        # Initialize the portfolio to some ratio
        init_ratio  = np.ones(18)*(1/18)    # the initial value distribution, should probably be a delta hedge
        action = self.vec_to_dict(init_ratio)
        self._agent_position = self.norm_portfolio(action, value = CVA_at_t0)
        print(self._agent_position)
        print(self._get_obs)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    # Convert a non dict action into a dict
    def vec_to_dict(self, action):
        if not len(action) == 18:
            raise ValueError('You passed an action of incorrect length to the enivironment, it should be 18 long in a normal numeric format like a list or ndarray')
        swapts = action[0:9]
        Qs = action[9:18]
        return dict({"Swaption Position" : swapts, "Q Position" : Qs})

    # Normalize a portfolio to have a consistent value 
    def norm_portfolio(self, action, value = None):
        if value == None:
            value = self.posValue()

        # Convert fraction of portfolio to amount of options with floating scale
        Q_values = self.Q_now()
        swap_values = self.swaptions_now()

        # Buy number of shares to match portfolio fraction of value
        for i in range(len(action["Swaption Position"])):
            if self.currpth.t_s[self.tIDX]-1.005> i:
                action["Swaption Position"][i] = action["Swaption Position"][i]/(max(swap_values[i],1e-16))
                action["Q Position"][i] = action["Q Position"][i]/(max(Q_values[i],1e-16))  
            else:        # Force zero position in expired positions
                action["Swaption Position"][i] = 0.0
                action["Q Position"][i] = 0.0    
        
        # Normalize Value / Enforce value restraints
        value_action = self.AposValue(action)
        #scale = value/value_action
        scale = value / value_action if value_action != 0 else 1.0 #ÄNDRAT

        scaled_action = dict({
             "Swaption Position" : action["Swaption Position"]*scale,
             "Q Position" : action["Q Position"]*scale,
        })

        return scaled_action

    # The meat and potatoes
    def step(self, action):
        # Format action and try to avoid sideeffects
        actionl = action.copy()
        if not isinstance(actionl, dict):
            actionl = sp.special.softmax(actionl)
            actionl = self.vec_to_dict(actionl)
        
        # Step Time forward
        self.tIDX = int(self.tIDX + 1)  

        # Set the new action
        actionl = self.norm_portfolio(actionl)
        self._agent_position = actionl

        # Observe the reward, which comes from the previous action
        reward = -self.PnL()**2
        info = self._get_info()
        observation = self._get_obs()
  
        # End the environment after we reach year 9
        terminated = self.currpth.t_s[self.tIDX] > 9
        truncated = False

        if terminated:
            self.reset()

        return observation, reward, terminated, truncated, info
    