#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(
    backup,
    n_repetitions,
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy="egreedy",
    epsilon=None,
    temp=None,
    smoothing_window=None,
    plot=False,
    n=5,
    eval_interval=500,
):
    returns_over_repetitions = []
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        if backup == "q":
            returns, timesteps = q_learning(
                n_timesteps,
                learning_rate,
                gamma,
                policy,
                epsilon,
                temp,
                plot,
                eval_interval,
            )

        returns_over_repetitions.append(returns)

    print("Running one setting takes {} minutes".format((time.time() - now) / 60))
    learning_curve = np.mean(
        np.array(returns_over_repetitions), axis=0
    )  # average over repetitions
    if smoothing_window is not None:
        learning_curve = smooth(
            learning_curve, smoothing_window
        )  # additional smoothing
    return learning_curve, timesteps


def experiment():
    ####### Settings
    # Experiment
    n_repetitions = 5 # number of times an experiment is repeated
    smoothing_window = 9  # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False  # Plotting is very slow, switch it off when we run repetitions

    # MDP
    n_timesteps = 1000001  # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 1000
    max_episode_length = 100
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values:
    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.05

    # Back-up & update
    backup = "q"  # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.1
    n = 5  # only used when backup = 'nstep'


    ####### Experiments
    optimal_episode_return = 280  # Set the reward that considers the problem of cartpole solved

    #### Assignment 2: Effect of exploration
    policy = "egreedy"
    epsilons = [0.03, 0.1, 0.3]
    learning_rate = 0.1
    backup = "q"
    Plot = LearningCurvePlot(
        title="Exploration: $\epsilon$-greedy versus softmax exploration"
    )
    Plot.set_ylim(-100, 100)
    for epsilon in epsilons:
        learning_curve, timesteps = average_over_repetitions(
            backup=backup,
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            learning_rate=learning_rate,
            gamma=gamma,
            policy=policy,
            epsilon=epsilon,
            smoothing_window=smoothing_window,
            plot=plot,
            n=n,
            eval_interval=eval_interval,
        )

        Plot.add_curve(timesteps, learning_curve, label=r"$\epsilon$-greedy, $\epsilon $ = {}".format(epsilon),)
    
    Plot.add_hline(optimal_episode_return, label="Considered solved")
    Plot.save("exploration.png")


if __name__ == "__main__":
    experiment()
