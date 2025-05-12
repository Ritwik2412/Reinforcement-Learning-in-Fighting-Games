import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# --- PPO Neural Network ---
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Compute policy (actor) and value (critic)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

# --- PPO Agent ---
class PPO:
    def __init__(self, state_dim, action_dim, lr=1e-3, clip_epsilon=0.2, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def select_action(self, state):
        # Select action using the policy
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, _ = self.policy(state)
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, states, actions, log_probs, rewards, dones, next_states):
        # Update policy and critic using PPO
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device).detach()  # Detach to avoid graph tracking
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        _, values = self.policy(states)
        _, next_values = self.policy(next_states)
        values = values.squeeze()
        next_values = next_values.squeeze()

        # Compute advantages
        returns = []
        gae = 0
        for reward, done, next_value in zip(reversed(rewards), reversed(dones), reversed(next_values)):
            delta = reward + self.gamma * next_value * (1 - done) - values[len(returns)]
            gae = delta + self.gamma * 0.95 * (1 - done) * gae
            returns.insert(0, gae + values[len(returns)])
        returns = torch.tensor(returns).to(self.device)
        advantages = (returns - values).detach()  # Detach to avoid graph tracking

        # PPO update
        for _ in range(10):
            # Recompute policy and values to create a fresh computation graph
            policy, values = self.policy(states)
            dist = Categorical(policy)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()  # No need for retain_graph=True
            self.optimizer.step()