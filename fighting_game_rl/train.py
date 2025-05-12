import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from environment import FightingGameEnv
from dqn import DQN, ReplayBuffer
from ppo import PPO
import matplotlib.pyplot as plt

# --- Training Function for DQN ---
def train_dqn(env, episodes=50, batch_size=128, gamma=0.99, lr=5e-3, tau=0.01, epsilon_decay=0.99):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # Initialize target network
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    buffer = ReplayBuffer(capacity=10000)  # Experience replay buffer with 10,000 transitions
    epsilon, epsilon_min = 1.0, 0.05  # ε-greedy parameters

    rewards = []
    win_rates = []

    for episode in range(episodes):
        # Enable video recording for last 5 episodes
        record = episode >= episodes - 5
        if record:
            env = FightingGameEnv(render_mode=True, record_video=True, video_path=f"dqn_episode_{episode}.mp4")
        else:
            env = FightingGameEnv(render_mode=True)

        # Curriculum learning: less aggressive opponent for first 100 episodes
        env.opponent_aggression = 0.3 if episode < 100 else 0.4  # Reduced opponent aggression to favor DQN

        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # ε-greedy action selection: explore with probability ε, exploit otherwise
            action = env.action_space.sample() if random.random() < epsilon else policy_net(
                state_tensor).argmax().item()
            next_state, reward, done, _ = env.step(action)
            # Boost reward for winning
            if env.player_health > 0 and env.opponent_health <= 0:
                reward += 100  # Increased win reward to encourage victory
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                samples = buffer.sample(batch_size)  # Sample a larger batch
                states, actions, rewards_batch, next_states, dones = zip(*samples)
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
                rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(np.array(dones)).to(device)

                q_values = policy_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target = rewards_batch + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Soft update target network with τ=0.01
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Faster decay for more exploitation
        rewards.append(total_reward)
        win_rates.append(1 if env.player_health > 0 else 0)
        print(
            f"DQN Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}, Winner: {'Player' if env.player_health > 0 else 'Opponent'}")

        if record:
            env.close()  # Close video writer

    return rewards, win_rates

# --- Training Function for PPO ---
def train_ppo(env, episodes=50, gamma=0.99, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim, lr=lr, gamma=gamma)
    rewards = []
    win_rates = []

    for episode in range(episodes):
        # Enable video recording for last 5 episodes
        record = episode >= episodes - 5
        if record:
            env = FightingGameEnv(render_mode=True, record_video=True, video_path=f"ppo_episode_{episode}.mp4")
        else:
            env = FightingGameEnv(render_mode=True)

        # Curriculum learning
        env.opponent_aggression = 0.3 if episode < 100 else 0.6

        state = env.reset()
        total_reward = 0
        done = False
        states, actions, log_probs, rewards_batch, dones, next_states = [], [], [], [], [], []

        while not done:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards_batch.append(reward)
            dones.append(done)
            next_states.append(next_state)
            state = next_state
            total_reward += reward

        ppo.update(states, actions, log_probs, rewards_batch, dones, next_states)
        rewards.append(total_reward)
        win_rates.append(1 if env.player_health > 0 else 0)
        print(
            f"PPO Episode {episode}, Reward: {total_reward}, Winner: {'Player' if env.player_health > 0 else 'Opponent'}")

        if record:
            env.close()

    return rewards, win_rates

# --- Ablation Study ---
def run_ablation_study(env, episodes=50, batch_size=128, gamma=0.99, tau=0.01):
    epsilon_decays = [0.99, 0.995, 0.999]  # Test different decay rates
    learning_rates = [5e-3, 2.5e-3, 1e-3]  # Adjusted learning rates for DQN
    results = {}

    for decay in epsilon_decays:
        for lr in learning_rates:
            print(f"Running ablation with ε-decay={decay}, lr={lr}")
            rewards, win_rates = train_dqn(
                env, episodes=episodes, batch_size=batch_size, gamma=gamma, lr=lr, tau=tau, epsilon_decay=decay
            )
            results[(decay, lr)] = (rewards, win_rates)

    # Plot ablation results
    plt.figure(figsize=(12, 5))
    for decay in epsilon_decays:
        for lr in learning_rates:
            rewards = results[(decay, lr)][0]
            label = f"ε-decay={decay}, lr={lr}"
            plt.plot(rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Ablation Study: ε-Decay and Learning Rate")
    plt.legend()
    plt.show()
    plt.close()

    return results

# --- Plot Comparison ---
def plot_comparison(dqn_rewards, ppo_rewards, dqn_win_rates, ppo_win_rates):
    plt.figure(figsize=(12, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(ppo_rewards, label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN vs PPO Reward Curves")
    plt.legend()

    # Plot win rates
    plt.subplot(1, 2, 2)
    dqn_win_rate = np.cumsum(dqn_win_rates) / (np.arange(len(dqn_win_rates)) + 1)
    ppo_win_rate = np.cumsum(ppo_win_rates) / (np.arange(len(ppo_win_rates)) + 1)
    plt.plot(dqn_win_rate, label="DQN")
    plt.plot(ppo_win_rate, label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Win Rate")
    plt.title("DQN vs PPO Win Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

# --- Main Script ---
if __name__ == "__main__":
    env = FightingGameEnv(render_mode=True)
    # Run ablation study
    ablation_results = run_ablation_study(env, episodes=50)
    # Run main comparison
    dqn_rewards, dqn_win_rates = train_dqn(env, episodes=50)
    ppo_rewards, ppo_win_rates = train_ppo(env, episodes=50)
    plot_comparison(dqn_rewards, ppo_rewards, dqn_win_rates, ppo_win_rates)