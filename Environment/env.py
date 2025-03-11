import gymnasium as gym
from gymnasium import wrappers as wrap
import numpy as np

from path_datatype import Path

class tradingEng(gym.Env):
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
            if not self.currpth.t_s[self.tIDX]-2 > i:
                action["Swaption Position"][i] = action["Swaption Position"][i]/(max(swap_values[i],1e-16))
                action["Q Position"][i] = action["Q Position"][i]/(max(Q_values[i],1e-16))  
            else:        # Force zero position in expired positions
                action["Swaption Position"][i] = 0.0
                action["Q Position"][i] = 0.0    
        
        # Normalize Value / Enforce value restraints
        value_action = self.AposValue(action)
        scale = value/value_action

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
        self.tIDX = self.tIDX + 1     

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

        return wrap.FlattenObservation(observation), reward, terminated, truncated, info
