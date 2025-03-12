#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


class QLearningAgent(BaseAgent):  # inherit variables from BaseAgent
    def update(self, s, a, r, s_next, done: bool):
        if done:
            # If epsiode is done, there is no next state, so G = r
            Gt = r

        else:
            # Collect the q-values of next state
            next_q_values = self.Q_sa[s_next, :]

            # Inherit gamma (discount factor) from BaseAgent
            Gt = r + self.gamma * max(next_q_values)

        # Update Q-values
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (Gt - self.Q_sa[s, a])

    # For easy storing and loading after having trained a network
    def store_q_table(self, filename: str):
        np.save(f"{filename}.npy", self.Q_sa)

    def load_q_table(self, filename: str):
        try:
            self.Q_sa = np.load(filename)
            print(f"Successfully loaded Q-table from {filename}")
        except FileNotFoundError:
            print(f"Could not find file {filename}, proceeding with fresh Q-table.")
        except Exception as e:
            print(f"Error loading Q-table: {e}")


def q_learning(
    n_timesteps: int,
    learning_rate: float,
    gamma: float,
    policy="egreedy",
    epsilon: float = None,
    temp: float = None,
    plot: bool = True,
    eval_interval: int = 500,
    store_q_values: bool = False,
    filename: str = "Q_table_values",
):
    """runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep"""

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TODO: Write your Q-learning algorithm here!

    # Reset env to start from state 0
    s = env.reset()
    for timestep in range(n_timesteps):
        # Obtain action from agent based on current state, the policy, epsilon and temp
        a = agent.select_action(s, policy, epsilon, temp)

        # Simulate the environment
        s_next, r, done = env.step(a)

        # Update Q-value
        agent.update(s, a, r, s_next, done)

        # Check if termination was reached
        if done:
            # Restart the env
            s = env.reset()
        else:
            # Move to the next state
            s = s_next

        # Evaluate an episode at timesteps eval_interval
        if (timestep % eval_interval) == 0:
            # Use the evaluate metod from the agent
            eval_return = agent.evaluate(eval_env)

            # Add returns and timestep to lists
            eval_returns.append(eval_return)
            eval_timesteps.append(timestep)

        if plot:
            env.render(
                Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1
            )  # Plot the Q-value estimates during Q-learning execution

    if store_q_values:
        agent.store_q_table(filename)

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    eval_interval = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(
        n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval
    )
    print(eval_returns, eval_timesteps)


if __name__ == "__main__":
    test()
