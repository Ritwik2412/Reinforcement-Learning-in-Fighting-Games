import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize a deque with maximum capacity to store experiences
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store a transition (s, a, r, s', done) in the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)

# --- DQN Neural Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Neural network with two hidden layers (128 neurons each) for Q-value approximation
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        # Forward pass to compute Q-values for all actions
        return self.fc(x)