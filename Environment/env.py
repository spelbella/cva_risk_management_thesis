import gymnasium as gym
import numpy as np

class tradingEng(gym.Env):
    
    def __init__(self, paths):
        # The paths
        self.paths = paths

        # The currently looked at path indx
        self.pthIDX = 0
        self.currpth =  paths[self.pthIDX]

        # The time index
        self.tIDX = -1

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
        return [self.currpth.Swaptions[self.tIDX][i] for i in range(0,9)]
    def Q_now(self):
        return [self.currpth.Q_s[self.tIDX][i] for i in range(1,10)]
    
    def posValue(self):
        swaptions_val = np.linalg.norm(np.inner(self._agent_position["Swaption Position"],self.swaptions_now()))
        Q_val = np.linalg.norm(np.inner(self._agent_position["Q Position"],self.Q_now()))
        return Q_val + swaptions_val
    
    def AposValue(self, action):
        swaptions_val = np.linalg.norm(np.inner(self.action["Swaption Position"],self.currpth.Swaptions[:][self.tIDX]))
        Q_val = np.linalg.norm(np.inner(self.action["Q Position"],self.currpth.Q_s[1:,self.tIDX]))
        return Q_val + swaptions_val
    
    def PnL(self):
        return self.posValue() - self.currpth.CVA[self.tIDX]

    def _get_info(self):
        return{
            "SqrdP&L": self.PnL()**2
        }
    
    def reset(self, seed = None, options = None):
        self.pthIDX = self.pthIDX + 1
        if self.pthIDX > 1550:
            self.pthIDX = 0
        self.currpth = self.paths[self.pthIDX]
        swaption_value_at0 = self.currpth.Swaptions[2][0]
        CVA_at_t0 = self.currpth.CVA[0]
        self._agent_position = dict({
                "Swaption Position" : [0,0,CVA_at_t0/swaption_value_at0, 0, 0, 0,0,0,0],
                "Q Position" : [0,0,0,0,0,0,0,0,0],
        })

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    # The meat and potatoes
    def step(self, action):
        self.tIDX = self.tIDX + 1
        entry_value = self.posValue()
        value_action = self.AposValue(action)

        scaled_action = action*(entry_value/value_action)

        self._agent_position = scaled_action

        # End the environment after we reach year 9
        terminated = self.currpth.t_s[self.tIDX] > 9
        truncated = False
        reward = -(self.PnL())**2
        info = self._get_info()
        observation = self._get_obs()

        return observation, reward, terminated, truncated, info
