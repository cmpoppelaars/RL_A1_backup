#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax


class BaseAgent:
    def __init__(
        self, n_states: int, n_actions: int, learning_rate: float, gamma: float
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(
        self, s, policy: str = "egreedy", epsilon: float = None, temp: float = None
    ):
        if policy == "greedy":
            # Select best action based on Q-values of state
            a = argmax(self.Q_sa[s, :])

        elif policy == "egreedy":
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            # Select a random action - explore
            # if a random number drawn between 0 and 1 is below epsilon, we take a random action
            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.n_actions)

            # Select greedy policy
            else:
                a = argmax(self.Q_sa[s, :])

        elif policy == "softmax":  # AKA: Boltzmann
            if temp is None:
                raise KeyError("Provide a temperature")

            # Return the softmax vector, then select the action with the highest probability
            a = argmax(softmax(self.Q_sa[s, :], temp))
        return a

    def update(self):
        raise NotImplementedError(
            "For each agent you need to implement its specific back-up method"
        )  # Leave this and overwrite in subclasses in other files

    def evaluate(
        self, eval_env, n_eval_episodes: int = 30, max_episode_length: int = 100
    ):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, "greedy")
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
