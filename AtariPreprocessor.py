from gym import Wrapper
from gym.spaces.box import Box
import numpy as np

# Implement the preprocessing described in Mnih et al. 2015 (Playing Atari with Deep Reinforcement Learning)
class AtariPreprocessor(Wrapper):
    def __init__(self, env):
        super(AtariPreprocessor, self).__init__(env)
        
        o = env.observation_space
        assert o.shape == (210, 160, 3)
        
        self.history = None
        self.num_frames = 4
        self.observation_space = Box(o.low[0,0,0], o.high[0,0,0], shape=(o.shape[0]/2, o.shape[1]/2, self.num_frames))
    
    def get_state(self):
        frames = []
        for frame in self.history:
            frame = np.dot(frame, [0.299, 0.587, 0.114])
            frame = frame[::2,::2]
            frames.append(frame)
        return np.stack(frames, axis=-1)
    
    def _step(self, action):
        state, reward, done, info = self.env.step(action)
        self.history.pop(0)
        self.history.append(state)
        return self.get_state(), reward, done, info
    
    def _reset(self):
        self.history = [self.env.reset()] * self.num_frames
        return self.get_state()
