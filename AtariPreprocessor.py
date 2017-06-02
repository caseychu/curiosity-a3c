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
        self.num_frames_to_stack = 4
        self.num_actions_to_repeat = 4
        self.observation_space = Box(o.low[0,0,0], o.high[0,0,0], shape=(o.shape[0]/2, o.shape[1]/2, self.num_frames_to_stack))
    
    def get_state(self):
        frames = []
        for frame in self.history[-self.num_frames_to_stack:]:
            frame = np.dot(frame / 255., [0.299, 0.587, 0.114])
            frame = frame[::2,::2]
            frames.append(frame)
        return np.stack(frames, axis=-1)
    
    def _step(self, action):
        total_rewards = 0
        done = False
        for i in range(self.num_actions_to_repeat):
            state, reward, done, info = self.env.step(action)
            total_rewards += reward
            self.history.pop(0)
            self.history.append(state)
            if done:
                break
        return self.get_state(), total_rewards, done, info
    
    def _reset(self):
        self.history = [self.env.reset()] * self.num_frames_to_stack
        return self.get_state()
    
    def _render(self, mode='rgb_array', close=False):
        return np.stack([self.get_state()[:-1,:,0].astype(np.uint8)]*3, axis=-1)
