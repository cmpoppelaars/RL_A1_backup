import gymnasium as gym
import numpy as np
import os
import time

from NDQN import VEC_NDQN

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


def hyperparameter_study(
    learning_rates=[1e-5, 1e-3, 0.1],
    epsilons=[0.05, 0.1, 0.5],
    network_sizes=[32, 64, 128],
    update_to_data_ratio=[0.1, 0.5, 1.0],
):
    n_eval_episodes = 50
    envs = gym.make_vec("CartPole-v1", 5000)
    eval_envs = gym.make_vec("CartPole-v1", n_eval_episodes // 2)

    for update_ratio in update_to_data_ratio:
        for epsilon in epsilons:
            for network_size in network_sizes:
                for learning_rate in learning_rates:
                
                
                    # Check if this set of hyperparameters has already been stored to a file, if yes, skip it
                    filename = f"NDQN_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz"
                    if os.path.exists(filename):
                        print(f"Skipping {filename}, already exists.")
                        continue

                    print(f"Testing update ratio {update_ratio} - lr {learning_rate} - eps {epsilon} - network size {network_size}")
                    NDQN = VEC_NDQN(
                        env=envs,
                        eval_env=eval_envs,
                        eval_time=5000,
                        epsilon=epsilon,
                        gamma=0.9,
                        learning_rate=learning_rate,
                        network_size=network_size,
                        update_to_data_ratio=update_ratio,
                    )

                    # Start learning
                    NDQN_learning_curve, NDQN_returns, NDQN_timesteps = (
                        average_over_repetitions(5, NDQN)
                    )

                    # Store results
                    np.savez(
                        filename,
                        learning_curve=NDQN_learning_curve,
                        returns_over_repetitions=NDQN_returns,
                        timesteps=NDQN_timesteps,
                    )


if __name__ == "__main__":
    hyperparameter_study()