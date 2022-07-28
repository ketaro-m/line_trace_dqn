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
    print(args)
    agant = DQNAgent()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## necessary to be executed from launch file: argv=[..., __name:=..., __log:=...]
    if any((s.startswith('__name') for s in sys.argv)):
        del sys.argv[-2:]
    ##

    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--gamma', default=0.995)
    parser.add_argument('--tau', default=1e-3)
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--every_update', default=4)

    parser.add_argument('--n_episodes', default=3000)
    parser.add_argument('--max_t', default=1000)
    parser.add_argument('--eps_start', default=1.0)
    parser.add_argument('--eps_end', default=0.01)
    parser.add_argument('--eps_decay', default=0.995)

    args = vars(parser.parse_args())

    agent = train(hyperparams=args)
