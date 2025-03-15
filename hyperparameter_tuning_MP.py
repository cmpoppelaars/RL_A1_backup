import gymnasium as gym
import numpy as np
import os
import time
import sys
import importlib
import itertools
import multiprocessing

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

def run_experiment(params):
    """ Runs a single experiment with given hyperparameters. """
    agent_class_name, param_dict, results_dir = params

    filename = os.path.join(results_dir, param_dict["filename"])
    if os.path.exists(filename):
        print(f"Skipping {filename}, already exists.")
        return  # Skip if already computed

    print(f"Testing {agent_class_name} with {param_dict}")

    # Dynamically import agent class
    try:
        module = importlib.import_module(agent_class_name)
        AgentClass = getattr(module, agent_class_name)
    except (ImportError, AttributeError):
        print(f"Error: Could not import {agent_class_name}. Ensure the module and class exist.")
        return

    # Create agent with current hyperparameters
    agent = AgentClass(**param_dict["agent_kwargs"])

    # Run experiment
    learning_curve, returns, timesteps = average_over_repetitions(5, agent)

    # Save results
    np.savez(
        filename,
        learning_curve=learning_curve,
        returns_over_repetitions=returns,
        timesteps=timesteps,
    )

def hyperparameter_study(agent_class_name, num_workers=None):
    """ Manages hyperparameter testing with multiprocessing. """
    nenvs = 5000
    n_eval_episodes = 50
    envs = gym.make_vec("CartPole-v1", nenvs)
    eval_envs = gym.make_vec("CartPole-v1", n_eval_episodes // 2)

    # Default hyperparameters
    learning_rates = [1e-5, 1e-3, 0.1] # [0.1, 1e-3, 1e-5]
    epsilons = [0.05, 0.25, 0.5]
    network_sizes = [32, 64, 128]
    update_to_data_ratio = [0.1, 0.5, 1.0]
    buffer_sizes = [25*nenvs, 15*nenvs, 5*nenvs]
    update_frequencies = [10*nenvs, 5*nenvs, nenvs]

    # Ensure the correct directory exists
    results_dir = f"{agent_class_name}_data"
    os.makedirs(results_dir, exist_ok=True)

    # Generate all parameter combinations
    param_list = []
    
    if agent_class_name == 'NDQN':
        for update_ratio, epsilon, network_size, learning_rate in itertools.product(update_to_data_ratio, epsilons, network_sizes, learning_rates):
            filename = f"NDQN_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz"
            param_list.append({
                "filename": filename,
                "agent_kwargs": {
                    "env": envs,
                    "eval_env": eval_envs,
                    "eval_time": nenvs,
                    "epsilon": epsilon,
                    "gamma": 0.9,
                    "learning_rate": learning_rate,
                    "network_size": network_size,
                    "update_to_data_ratio": update_ratio,
                    "n_eval_episodes": n_eval_episodes,
                }
            })

    elif agent_class_name == "DQN_ER":
        for update_ratio, epsilon, network_size, learning_rate, buffer_size in itertools.product(update_to_data_ratio, epsilons, network_sizes, learning_rates, buffer_sizes):
            filename = f"DQN_ER_buffersize{buffer_size}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz"
            param_list.append({
                "filename": filename,
                "agent_kwargs": {
                    "env": envs,
                    "eval_env": eval_envs,
                    "eval_time": nenvs,
                    "epsilon": epsilon,
                    "gamma": 0.9,
                    "learning_rate": learning_rate,
                    "network_size": network_size,
                    "update_to_data_ratio": update_ratio,
                    "batch_size": nenvs,
                    "replay_buffer_size": buffer_size,
                    "n_eval_episodes": n_eval_episodes,
                }
            })

    elif agent_class_name == "DQN_TN":
        for update_freq, learning_rate, network_size, epsilon, update_ratio in itertools.product(update_frequencies, learning_rates, network_sizes, epsilons, update_to_data_ratio):
            filename = f"DQN_TN_update_freq{update_freq}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz"
            param_list.append({
                "filename": filename,
                "agent_kwargs": {
                    "env": envs,
                    "eval_env": eval_envs,
                    "eval_time": nenvs,
                    "epsilon": epsilon,
                    "gamma": 0.9,
                    "learning_rate": learning_rate,
                    "network_size": network_size,
                    "update_to_data_ratio": update_ratio,
                    "TargetNetworkUpdateFq": update_freq,
                    "n_eval_episodes": n_eval_episodes,
                }
            })

    elif agent_class_name == "DQN_ER_TN":
        for buffer_size, update_freq, learning_rate, network_size, epsilon, update_ratio in itertools.product(buffer_sizes, update_frequencies, learning_rates, network_sizes, epsilons, update_to_data_ratio):
            filename = f"DQN_ER_TN_buffer_size{buffer_size}_update_freq{update_freq}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz"
            param_list.append({
                "filename": filename,
                "agent_kwargs": {
                    "env": envs,
                    "eval_env": eval_envs,
                    "eval_time": nenvs,
                    "epsilon": epsilon,
                    "gamma": 0.9,
                    "learning_rate": learning_rate,
                    "network_size": network_size,
                    "update_to_data_ratio": update_ratio,
                    "TargetNetworkUpdateFq": update_freq,
                    "batch_size": nenvs,
                    "replay_buffer_size": buffer_size,
                    "n_eval_episodes": n_eval_episodes,
                }
            })

    # Run experiments in parallel
    if num_workers is not None:
        try:
            num_workers = int(num_workers)
        except ValueError:
            print("Invalid num_workers argument. Must be an integer.")
            sys.exit(1)

        # Ensure it doesn't exceed tasks
        num_workers = min(num_workers, len(param_list))

    else:
        num_workers = min(multiprocessing.cpu_count(), len(param_list))  # Use all available CPUs

    print(f"Using {num_workers} processes.")


    with multiprocessing.Pool(num_workers) as pool:
        pool.map(run_experiment, [(agent_class_name, p, results_dir) for p in param_list])

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python hyperparameter_tuning.py <AgentClass> [num_workers]")
        sys.exit(1)

    agent_class_name = sys.argv[1]
    num_workers = sys.argv[2] if len(sys.argv) == 3 else None
    hyperparameter_study(agent_class_name, num_workers)
