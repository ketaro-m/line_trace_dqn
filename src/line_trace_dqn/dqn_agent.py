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

    def __init__(self):
        pass


if __name__ == "__main__":
    net = QNet(1,1,1) # tmp