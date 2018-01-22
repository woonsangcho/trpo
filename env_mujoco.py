from environment import Environment
import gym
import numpy as np

class Mujoco(Environment):

    def __init__(self, **kwargs):
        super().__init__()
        self.env = self.create_environment(**kwargs)
        self.reset_environment()
        self.finished = False

    def create_environment(self, **kwargs):
        env = gym.make(kwargs['environ_string'])
        return env

    def reset_environment(self):
        self.current_state = self.env.reset()
        return self.current_state

    def get_state_shape(self):
        return self.env.observation_space.shape

    def get_num_action(self):
        return self.env.action_space.shape[0]

    def get_current_state(self):
        return self.current_state

    def get_action_bound(self):
        return self.env.action_space.high

    def perform_action(self, action):
        next_state, reward, terminal, info = self.env.step(action)
        return np.reshape(next_state, [-1,]), reward, terminal, info

    def set_seed(self, seed):
        self.env.seed(seed)

