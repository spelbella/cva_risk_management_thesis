import gymnasium as gym
from path_datatype import Path
import numpy as np

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
