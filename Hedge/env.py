import gymnasium as gym
from gymnasium import wrappers as wrap
import numpy as np
import scipy as sp

from path_datatype import Path

class tradingEngOrg(gym.Env):
    def __init__(self, paths):
        # The paths
        self.paths = paths

        # The currently looked at path indx
        self.pthIDX = 0
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
    



class tradingEng(gym.Env):
    
    def __init__(self, paths):
        # The paths
        self.paths = paths

        # The currently looked at path indx
        self.pthIDX = 0
        self.currpth =  paths[self.pthIDX]

        # The time index
        self.tIDX = 0

        # Tracks the last held position
        self._agent_position = dict()

        # The Observation space, for now let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and the CVA(strictly shouldn't be nessecary but eh)
        self.observation_space = gym.spaces.Dict(
            {
                "Swaption Position" : gym.spaces.Box(0, 9, shape=(1,), dtype=float),
                "Q Position" : gym.spaces.Box(0, 9, shape=(1,), dtype=float),
                "Swaption Value" : gym.spaces.Box(0, 9, shape=(1,), dtype=float),
                "Q Value" : gym.spaces.Box(0, 9, shape=(1,), dtype=float),
                "r" : gym.spaces.Box(0, 1, shape=(1,), dtype=float),
            }
        )

        # The action space, let's let the action space be to take a new position
        self.action_space = gym.spaces.Dict(
            {
                "Swaption Position" : gym.spaces.Box(0, 9, shape=(1,), dtype=float),
                "Q Position" : gym.spaces.Box(0, 9, shape=(1,), dtype=float),
            }
        )

        self.reset()

    # Constructing observations
    def _get_obs(self):
        return {
                    #"Swaption Position": np.nan_to_num(self._agent_position["Swaption Position"], nan=0.0),
                    "Swaption Position": self._agent_position["Swaption Position"],
                    "Q Position": self._agent_position["Q Position"],
                    "Swaption Value" : self.swaptions_now(),
                    "Q Value" : self.Q_now(),
                    "r" : self.currpth.r[self.tIDX]
                }
    
    def swaptions_now(self):
        try:
            return [self.currpth.Swaptions[i][self.tIDX] for i in range(0,9)]
        except:
            print(self.tIDX)
        finally:
            pass
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
            "P&L": self.PnL()
        }
    
    def reset(self, seed = None, options = None):
        self.tIDX = 0
        self.pthIDX = self.pthIDX + 1
        if self.pthIDX > 1550:
            self.pthIDX = 0
        self.currpth = self.paths[self.pthIDX]
        swaption_value_at0 = self.currpth.Swaptions[9][0]
        CVA_at_t0 = self.currpth.CVA[0]
        self._agent_position = dict({
                "Swaption Position" : [0.0001,0.000001,0.000001, 0.000001, 0.000001, 0.000001,0.00001,0.00001,CVA_at_t0/swaption_value_at0],
                "Q Position" : [0.00001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001,0.000001],
        })

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    # The meat and potatoes
    def step(self, action):
        self.tIDX = self.tIDX + 1
        actionl = action.copy()
        # Force zero position in expired positions
        for i in range(len(actionl["Swaption Position"])):
            if self.currpth.t_s[self.tIDX]-2 > i:
                actionl["Swaption Position"][i] = 0.0
                actionl["Q Position"][i] = 0.0               

        # Convert fraction of portfolio to amount of options
        agent_position = dict()
        agent_position["Swaption Position"] = np.zeros(9)
        agent_position["Q Position"] = np.zeros(9)
        swap_values = self.swaptions_now()
        Q_values = self.Q_now()
        for i in range(len(action["Swaption Position"])):
            agent_position["Swaption Position"][i] = actionl["Swaption Position"][i]/(max(swap_values[i],1e-16))
            agent_position["Q Position"][i] = actionl["Q Position"][i]/(max(Q_values[i],1e-16))  

        # Enforce value restraints
        entry_value = self.posValue()
        value_action = self.AposValue(agent_position)
        scale = entry_value/value_action

        scaled_action = dict({
             "Swaption Position" : agent_position["Swaption Position"]*scale,
             "Q Position" : agent_position["Q Position"]*scale,
        })
        self._agent_position = scaled_action

        # End the environment after we reach year 9
        terminated = self.currpth.t_s[self.tIDX] > 9
        truncated = False
        reward = -self.PnL()**2
        info = self._get_info()
        observation = self._get_obs()

        if terminated:
            self.reset()

        return observation, reward, terminated, truncated, info

