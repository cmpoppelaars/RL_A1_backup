import gymnasium as gym
import numpy as np
import time

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from NDQN import NDQN

def smooth(y, window, poly=2):
    return savgol_filter(y,window,poly)


def average_over_repetitions(
    n_repetitions,
    agent,
):
    returns_over_repetitions = []

    for rep in range(n_repetitions):  # Loop over repetitions
        print(f"Starting iteration {rep+1}")
        now = time.time()

        # Reset agent
        agent.reset_weights()
        
        # Train using reset weights
        agent.train_agent()

        # Obtain returns and timesteps
        returns, timesteps = agent.eval_returns, agent.eval_timesteps
        returns_over_repetitions.append(returns)

        print("Running one rep takes {} minutes".format((time.time() - now) / 60))

    learning_curve = np.mean(
        np.array(returns_over_repetitions), axis=0
    )  # average over repetitions
    
    return learning_curve, returns_over_repetitions, timesteps


n_eval_episodes = 50
envs = gym.make_vec("CartPole-v1", 5000)
eval_envs = gym.make_vec("CartPole-v1", n_eval_episodes // 2)

NDQN = NDQN(
    env=envs,
    eval_env=eval_envs,
    eval_time=5000,
    epsilon=0.1,
    gamma=0.9,
    learning_rate=1e-4, 
    network_size=64,
)

# Start learning
NDQN_learning_curve, NDQN_returns, NDQN_timesteps = average_over_repetitions(5, NDQN)

# Plot results using matplotlib
plt.figure(figsize=(10, 6))
for it in range(len(NDQN_returns)):
    plt.plot(NDQN_timesteps, NDQN_returns[it], label=f'Naive DQN itr {it+1}', alpha=0.5)
plt.plot(NDQN_timesteps, smooth(NDQN_learning_curve, window=9), label='Naive DQN smoothed', linewidth=2, color='black')
plt.xlabel('Timesteps')
plt.ylabel('Returns')
plt.title('Learning Curve Naive DQN')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("baseline_DQN_results.pdf", dpi=400)