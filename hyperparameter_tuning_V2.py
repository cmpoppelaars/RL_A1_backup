import gymnasium as gym
import numpy as np
import os
import time
import sys
import importlib

def average_over_repetitions(n_repetitions, agent):
    returns_over_repetitions = []

    for rep in range(n_repetitions):
        print(f"Starting iteration {rep+1}")
        now = time.time()

        agent.reset_weights()
        agent.train_agent()

        returns, timesteps = agent.eval_returns, agent.eval_timesteps
        returns_over_repetitions.append(returns)

        print("Running one rep takes {} minutes".format((time.time() - now) / 60))

    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)
    return learning_curve, returns_over_repetitions, timesteps

def hyperparameter_study(agent_class_name):
    nenvs = 5000
    n_eval_episodes = 50
    envs = gym.make_vec("CartPole-v1", nenvs)
    eval_envs = gym.make_vec("CartPole-v1", n_eval_episodes // 2)

    # Default hyperparamters
    learning_rates = [1e-5, 1e-3, 0.1]
    epsilons = [0.05, 0.25, 0.5]
    network_sizes = [32, 64, 128]
    update_to_data_ratio = [0.1, 0.5, 1.0]

    # ER related
    buffer_sizes = [25*nenvs, 15*nenvs, 5*nenvs]

    # TN related
    update_frequencies = [10*nenvs, 5*nenvs, nenvs]
    
    # Import the specified agent class dynamically
    module_name = f"{agent_class_name}"
    class_name = f"{agent_class_name}"
    try:
        module = importlib.import_module(module_name)
        AgentClass = getattr(module, class_name)
    except (ImportError, AttributeError):
        print(f"Error: Could not import {agent_class_name}. Make sure the module and class exist.")
        sys.exit(1)
    
    # Ensure the correct directory exists
    results_dir = f"{agent_class_name}_data"
    os.makedirs(results_dir, exist_ok=True)

    if agent_class_name == 'NDQN':

        for update_ratio in update_to_data_ratio:
            for epsilon in epsilons:
                for network_size in network_sizes:
                    for learning_rate in learning_rates:
                        filename = os.path.join(results_dir, f"NDQN_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                        if os.path.exists(filename):
                            print(f"Skipping {filename}, already exists.")
                            continue

                        print(f"Testing {agent_class_name} with update ratio {update_ratio} - lr {learning_rate} - eps {epsilon} - network size {network_size}")
                        agent = AgentClass(
                            env=envs,
                            eval_env=eval_envs,
                            eval_time=nenvs,
                            epsilon=epsilon,
                            gamma=0.9,
                            learning_rate=learning_rate,
                            network_size=network_size,
                            update_to_data_ratio=update_ratio,
                            n_eval_episodes=n_eval_episodes,
                        )

                        learning_curve, returns, timesteps = average_over_repetitions(5, agent)

                        np.savez(
                            filename,
                            learning_curve=learning_curve,
                            returns_over_repetitions=returns,
                            timesteps=timesteps,
                        )

    elif agent_class_name == "DQN_ER":

        for update_ratio in update_to_data_ratio:
            for epsilon in epsilons:
                for network_size in network_sizes:
                    for learning_rate in learning_rates:
                        for buffer_size in buffer_sizes:


                            filename = os.path.join(results_dir, f"DQN_ER_buffersize{buffer_size}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                            if os.path.exists(filename):
                                print(f"Skipping {filename}, already exists.")
                                continue

                            print(f"Testing {agent_class_name} with buffer size {buffer_size} - update ratio {update_ratio} - lr {learning_rate} - eps {epsilon} - network size {network_size}")
                            agent = AgentClass(
                                env=envs,
                                eval_env=eval_envs,
                                eval_time=nenvs,
                                epsilon=epsilon,
                                gamma=0.9,
                                learning_rate=learning_rate,
                                network_size=network_size,
                                update_to_data_ratio=update_ratio,
                                batch_size = nenvs,
                                replay_buffer_size = buffer_size,
                                n_eval_episodes=n_eval_episodes, 
                            )

                            learning_curve, returns, timesteps = average_over_repetitions(5, agent)

                            np.savez(
                                filename,
                                learning_curve=learning_curve,
                                returns_over_repetitions=returns,
                                timesteps=timesteps,
                            )

    elif agent_class_name == "DQN_TN":
        for update_ratio in update_to_data_ratio:
            for epsilon in epsilons:
                for network_size in network_sizes:
                    for learning_rate in learning_rates:
                        for update_freq in update_frequencies:


                            filename = os.path.join(results_dir, f"DQN_TN_update_freq{update_freq}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                            if os.path.exists(filename):
                                print(f"Skipping {filename}, already exists.")
                                continue

                            print(f"Testing {agent_class_name} with update freq {update_freq} - update ratio {update_ratio} - lr {learning_rate} - eps {epsilon} - network size {network_size}")
                            agent = AgentClass(
                                env=envs,
                                eval_env=eval_envs,
                                eval_time=nenvs,
                                epsilon=epsilon,
                                gamma=0.9,
                                learning_rate=learning_rate,
                                network_size=network_size,
                                update_to_data_ratio=update_ratio,
                                TargetNetworkUpdateFq = update_freq,
                                n_eval_episodes=n_eval_episodes, 
                            )

                            learning_curve, returns, timesteps = average_over_repetitions(5, agent)

                            np.savez(
                                filename,
                                learning_curve=learning_curve,
                                returns_over_repetitions=returns,
                                timesteps=timesteps,
                            )

    elif agent_class_name == "DQN_ER_TN":
        for update_ratio in update_to_data_ratio:
            for epsilon in epsilons:
                for network_size in network_sizes:
                    for learning_rate in learning_rates:
                        for update_freq in update_frequencies:
                            for buffer_size in buffer_sizes:
                                filename = os.path.join(results_dir, f"DQN_ER_TN_buffer_size{buffer_size}_update_freq{update_freq}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                                if os.path.exists(filename):
                                    print(f"Skipping {filename}, already exists.")
                                    continue

                                print(f"Testing {agent_class_name} with buffer size {buffer_size} - update freq {update_freq} - update ratio {update_ratio} - lr {learning_rate} - eps {epsilon} - network size {network_size}")
                                agent = AgentClass(
                                    env=envs,
                                    eval_env=eval_envs,
                                    eval_time=nenvs,
                                    epsilon=epsilon,
                                    gamma=0.9,
                                    learning_rate=learning_rate,
                                    network_size=network_size,
                                    update_to_data_ratio=update_ratio,
                                    TargetNetworkUpdateFq = update_freq,
                                    batch_size = nenvs,
                                    replay_buffer_size = buffer_size,
                                    n_eval_episodes=n_eval_episodes, 
                                )

                                learning_curve, returns, timesteps = average_over_repetitions(5, agent)

                                np.savez(
                                    filename,
                                    learning_curve=learning_curve,
                                    returns_over_repetitions=returns,
                                    timesteps=timesteps,
                                )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hyperparameter_tuning.py <AgentClass>")
        sys.exit(1)
    agent_class_name = sys.argv[1]
    hyperparameter_study(agent_class_name)
