import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4              # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """ Returns actions for given state as per current policy """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state) 

        self.qnetwork_local.train()

        if random.random() > eps:
            a = np.argmax(action_values.cpu().data.numpy())
        else:
            a = random.choice(np.arange(self.action_size))

        return a

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max Q values for next_states from target network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from Local Model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute Loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Target Network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ θ_target = τ*θ_local + (1 - τ)*θ_target, where τ is interpolation parameter """
        for target_p, local_p in zip(target_model.parameters(), local_model.parameters()):
            tt = tau * local_p.data + (1 - tau) * target_p.data
            target_p.data.copy_(tt)

class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        fields = ['state', 'action', 'reward', 'next_state', 'done']
        self.experience = namedtuple('Experience', field_names=fields)
        self.seed = seed
    
    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [e.state for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
