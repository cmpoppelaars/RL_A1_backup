# Reinforcement Learning Agents and Utilities

This repository contains a collection of Python scripts for training and evaluating different Deep Q-Network (DQN) agents, performing hyperparameter tuning, and visualizing learning curves.

## Table of Contents

- [Agents](#agents)
- [Baseline Experiment](#baseline-experiment)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Helper Scripts](#helper-scripts)
- [Usage](#usage)

## Agents

The repository includes four reinforcement learning agents implemented in Python:

1. **NDQN** - A naive Deep Q-Network agent.
2. **DQN\_ER** - A DQN agent with experience replay, which utilizes a replay buffer.
3. **DQN\_TN** - A DQN agent with a target network that is periodically updated.
4. **DQN\_ER\_TN** - A DQN agent that combines both experience replay and a target network.

## Baseline Experiment

The script `baseline_NDQN_result.py` contains the first experimental result for the project. It runs the NDQN agent on the `CartPole-v1` environment from Gymnasium for **5 repetitions of 1 million environment steps**. The output includes:

- Individual learning curves for each repetition.
- The mean learning curve over all 5 repetitions.

## Hyperparameter Tuning

The script `hyperparameter_tuning_MP.py` performs hyperparameter tuning using **multiprocessing** to utilize all available CPU resources efficiently. It conducts a hyperparameter grid search for all agents.

**Usage:**

```sh
python hyperparameter_tuning_MP.py <Agent> <Number of processors>
```

- `<Agent>`: The agent to be tuned (e.g., `NDQN`, `DQN_ER`, `DQN_TN`, `DQN_ER_TN`).
- `<Number of processors>`: (Optional) Number of CPU cores to use. If not specified, all available cores are used.

## Helper Scripts

1. `plot_learning_curves.py`

   - Plots learning curves from the hyperparameter tuning process.
   - Aggregates and visualizes the average learning curve for each hyperparameter setting.
   - Saves the results as a **PDF file** with subplots for each hyperparameter.

2. `ReplayBuffer.py`

   - Implements the replay buffer used in experience replay agents.
   - Stores past experiences and enables efficient sampling for training.

## Usage

For hyperparameter tuning:

```sh
python hyperparameter_tuning_MP.py DQN_ER_TN 4
```

(This runs the tuning for `DQN_ER_TN` using 4 processors.)

For plotting learning curves:

```sh
python plot_learning_curves.py
```

---
