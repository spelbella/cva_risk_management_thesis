import gymnasium as gym
from gymnasium import wrappers as wrap
import numpy as np

from MarketGeneratingFunctions import Path

class tradingEng(gym.Env):
    def __init__(self, paths, action = 'big', obs = 'big'):
        # The paths
        self.paths = paths

        # The currently looked at path indx
        self.pthIDX = 0
        self.currpth =  paths[self.pthIDX]
        self.npths = len(paths)

        # The action and obs space
        self.action = action
        self.obs = obs

        # The time index
        self.tIDX = 0

        # Tracks the last held position
        self._agent_position = dict()

        # The Observation space, 
        match obs:
            case 'big':
                # let's let it look at the value of the 9 swaptions, the 9 (non constant) Qs, it's portfolio in each of those, and r (37 actions), 
                # order = [Actions, Swaptions, Qs, Interest Rate]
                lower = np.concatenate([-np.inf*np.ones(18), np.zeros(18), -np.inf])
                upper = np.concatenate([np.inf*np.ones(18), np.inf*np.ones(9), np.ones(9),np.inf])
                self.observation_space = gym.spaces.Box(lower = lower,upper = upper, dtype=float)
            case 'small':
                # Just the interest rate, default intensity and previous action and time, order = [prev actions, intensity, interest, time]
                lower = np.concatenate([np.zeros(18),0 , -np.inf])
                upper = np.concatenate([np.ones(18), np.inf, np.inf])
                self.observation_space = gym.spaces.Box(lower = lower,upper = upper, dtype=float)
            case 'xs':
                # Just the interest rate and default intensity and time, order = [intensity, interest, time]
                lower = np.concatenate([0, -np.inf])
                upper = np.concatenate([np.inf, np.inf])
                self.observation_space = gym.spaces.Box(lower = lower,upper = upper, dtype=float)
            case _:
                return "No observation space matching input"
        
        match action:
            case 'big':
                # The action space, let's let the action space be to take a new position -> 18 dim
                lowera = np.concatenate([-np.inf*np.ones(18)])
                uppera = np.concatenate([np.inf*np.ones(18)])
                self.action_space = gym.spaces.Box(lower = lowera,upper = uppera, dtype=float)
            case 'small':
                # Just the swaption and q with last expiry
                lowera = np.concatenate([-np.inf])
                uppera = np.concatenate([np.inf])
                self.action_space = gym.spaces.Box(lower = lowera,upper = uppera, dtype=float)
            case _:
                return "No matching action space"

        self.reset()

    # Constructing observations
    def _get_obs(self):
        match self.obs:
            case 'big':
                return np.concatenate([
                    self._agent_position["Swaption Position"],
                    self._agent_position["Q Position"],
                    self.swaptions_now(),
                    self.Q_now(),
                    self.currpth.r[self.tIDX]
                ])
            case 'small':
                return np.concatenate([
                    self._agent_position["Swaption Position"],
                    self._agent_position["Q Position"],
                    self.currpth.lambdas[self.tIDX],
                    self.currpth.r[self.tIDX],
                    self.currpth.t_s[self.tIDX]
                ])
            case 'xs':
                return np.concatenate([
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

    # The meat and potatoes
    def step(self, action):
        # Format action and try to avoid sideeffects
        actionl = action.copy()
        match self.action:
            case 'big':
                pass
            case 'small':
                swpt = actionl[0]
                Q = actionl[1]
                actionl = [np.zeros(8), swpt, np.zeros(8), Q]
        if not isinstance(actionl, dict):
            actionl = self.vec_to_dict(actionl)
        
        # Step Time forward
        self.tIDX = self.tIDX + 1     

        # Set the new action
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
