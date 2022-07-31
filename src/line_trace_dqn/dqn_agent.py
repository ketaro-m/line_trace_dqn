#! /usr/bin/env python

import numpy as np
import random
from collections import namedtuple, deque
from src.line_trace_dqn.env import Env
from src.line_trace_dqn.q_net import QNet

import torch
import torch.nn.functional as F
import torch.optim as optim

class DQNAgent():

    """Interacts with and learns form environment."""
    def __init__(self, state_size, action_size, optional_hyperparams: dict = {}, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed


        """ hyperparameters """
        self.hyperparams = ['batch_size', 'gamma', 'tau', 'lr', 'update_every', 'n_episodes', 'max_t', 'eps_start', 'eps_end', 'eps_decay']
        self.buffer_size = int(1e5) # replay buffer size
        self.batch_size = 64        # minibatch size, or how many samples taken the replay buffer for experience replay
        self.gamma = 0.995          # discount factor
        self.tau = 1e-3             # for soft update of target parameters
        self.lr = 0.1               # learning rate
        self.update_every = 4       # how often to update the network
        self.n_episodes = 3000      # maximum number of training epsiodes
        self.max_t = 1000           # maximum number of timesteps per episode
        self.eps_start = 1.0        # starting value of epsilon, for epsilon-greedy action selection
        self.eps_end = 0.01         # minimum value of epsilon
        self.eps_decay = 0.995      # mutiplicative factor (per episode) for decreasing epsilon
        self.set_hyperparams(optional_hyperparams)


    def set_hyperparams(self, param_dict: dict):
        """setter for hyperparamters

        Args:
            param_dict (dict): parameter {param_name: param_value}
        """
        for k, v in param_dict.items():
            assert hasattr(self, k), "no such hyperparamter named '{}'".format(k)
            setattr(self, k, v)


    def get_hyperparams(self):
        """getter for hyperparameters

        Returns:
            dict: param_dict {param_name: param_value}
        """
        param_dict = {}
        for key in self.hyperparams:
            param_dict[key] = getattr(self, key)
        return param_dict


if __name__ == "__main__":
    net = QNet(1,1,1) # tmp
