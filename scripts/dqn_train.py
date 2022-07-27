#! /usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.line_trace_dqn.dqn_agent import DQNAgent

import torch
import torch.nn.functional as F
import torch.optim as optim

if __name__ == "__main__":
    agant = DQNAgent()

