#! /usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.line_trace_dqn.dqn_agent import DQNAgent

import torch
import torch.nn.functional as F
import torch.optim as optim


def train(hyperparams: dict):

    state_size = 26
    action_size = 5

    agent = DQNAgent(state_size, action_size, hyperparams, seed=0)

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## necessary to be executed from launch file: argv=[..., __name:=..., __log:=...]
    if any((s.startswith('__name') for s in sys.argv)):
        del sys.argv[-2:]
    ##

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--every_update', type=int, default=4)

    parser.add_argument('--n_episodes', type=int, default=3000)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.995)

    args = vars(parser.parse_args())

    agent = train(hyperparams=args)

    """ for debug (later delete) """
    print(agent.get_hyperparams())
