#! /usr/bin/env python

import os
import numpy as np
import random
from collections import namedtuple, deque
from src.line_trace_dqn.env import Env
from src.line_trace_dqn.q_net import QNet

import torch
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # TODO

class DQNAgent():

    """Interacts with and learns form environment."""
    def __init__(self, state_size, action_size, optional_hyperparams: dict = {}, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        self.load_model = False # whether load model from some file
        self.load_episode = 0   # episode number

        """ hyperparameters """
        self.hyperparams = ['batch_size', 'gamma', 'tau', 'lr', 'update_every', 'n_episodes', 'max_t', 'eps_start', 'eps_end', 'eps_decay']
        self.buffer_size = int(1e5) # replay buffer size
        self.batch_size = 64        # minibatch size, or how many samples taken the replay buffer for experience replay
        self.gamma = 0.995          # discount factor
        self.tau = 1e-3             # for soft update of target parameters
        self.lr = 0.001             # learning rate
        self.update_every = 4       # how often to update the network
        self.n_episodes = 1000      # maximum number of training epsiodes
        self.max_t = 6000           # maximum number of timesteps per episode (control_freq * max_s)
        self.eps_start = 1.0        # starting value of epsilon, for epsilon-greedy action selection
        self.eps_end = 0.01         # minimum value of epsilon
        self.eps_decay = 0.995      # mutiplicative factor (per episode) for decreasing epsilon
        self.set_hyperparams(optional_hyperparams)

        # Q-Network
        # TODO: to(device)
        # self.qnetwork_local = QNet(action_size, seed).to(device)
        # self.qnetwork_target = QNet(action_size, seed).to(device)
        self.qnetwork_local = QNet(action_size, seed)
        self.qnetwork_target = QNet(action_size, seed)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed) # Replay memory
        self.t_step = 0 # Initialize time step (for updating every UPDATE_EVERY steps)
        self.epsilon = self.eps_start


    def set_model_param(self, fpath):
        """load saved model parameters

        Args:
            fpath (str): model .pth file name
        """
        self.load_model = True
        # self.qnetwork_local.load_state_dict(torch.load(fpath, map_location=device)) # TODO
        self.qnetwork_local.load_state_dict(torch.load(fpath)) # TODO


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


    def update_epsilon(self):
        """decrease the epsilon
        """
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_end)


    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn
            if len(self.memory) > self.batch_size:
                experience = self.memory.sample()
                self.learn(experience)


    def act(self, state):
        """action

        Args:
            state (np.array_like): current state image

        Returns:
            int, np.ndarray: action number, Q value
        """
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device) # TODO
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = torch.permute(state, (0, 3, 1, 2)) # [batch_num, RGB channel=3, x, y]
        self.qnetwork_local.eval() # change to the eval mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # change to the train mode

        # Epsilon-greedy action selction
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().data.numpy()[0]
        else:
            return random.choice(np.arange(self.action_size)), action_values.cpu().data.numpy()[0]


    def learn(self, experiences):
        """_summary_

        Args:
            experiences (Tuble[torch.Variable]): tuple of (s, a, r, s', done) tuples
        """

        states, actions, rewards, next_states, dones = experiences
        ## compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to qnetwork_local
        # size(predicted_targets) = (batch_size, 1)
        states = torch.permute(states, (0, 3, 1, 2)) # [batch_num, RGB channel=3, x, y]
        next_states = torch.permute(next_states, (0, 3, 1, 2)) # [batch_num, RGB channel=3, x, y]
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma * labels_next * (1-dones))

        # loss = criterion(predicted_targets,labels).to(device) #TODO
        loss = criterion(predicted_targets, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)




class ReplayBuffer():
    """Fixed-size buffe to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state",
                                                                "done"])
        self.seed = seed


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        """
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)


    def sample(self):
        """Randomly sample a batch of experiences from memory
        """
        experiences = random.sample(self.memory,k=self.batch_size)

        # TODO: to(device)
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=0)).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory.
        """
        return len(self.memory)






if __name__ == "__main__":
    agent = DQNAgent((36, 64), 5)
