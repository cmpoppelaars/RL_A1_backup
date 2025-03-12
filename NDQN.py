#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class NDQN:
    def __init__(
        self,
        env,
        eval_env,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        n_steps: int = 1000001,  # number of steps the agent can take, marking how long it can train
        eval_time: int = 1000,  # number of steps before we evaluate the agent
        n_eval_episodes: int = 30,
        network_size: int = 32,
        update_to_data_ratio: float = 1.0 # ratio of data from vectorized env that is used for updating/training the agent, between 0 and 1
    ):
        
        # Environment related parameters
        self.env = env
        self.eval_env = eval_env
        self.n_envs = env.num_envs  # Number of environments to run
        self.n_eval_envs = eval_env.num_envs
        self.state_size = self.env.single_observation_space.shape[0]
        self.action_size = self.env.single_action_space.n

        # Training related parameters
        self.n_steps = n_steps
        self.eval_time = eval_time
        self.n_eval_episodes = n_eval_episodes

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.network_size = network_size
        self.update_to_data_ratio = max(0.0, min(update_to_data_ratio, 1.0))
        
        # Construct naive DQN model
        self.agent = self.build_model()

    def build_model(self):
        """Constructs a simple neural network for Q-learning."""
        model = Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(2 * self.network_size, activation="relu"),
                Dense(self.network_size, activation="relu"),
                Dense(self.action_size, activation="linear")
            ]
        )
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics = ['accuracy'])
        return model

    def select_action(self, state: tf.Tensor):
        """Returns an action based on an epsilon-greedy strategy for batch input."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size, size=self.n_envs)
        else:
            Q_values = self.agent(state)  # Shape (n_envs, action_size)
            return np.argmax(Q_values, axis=1)
        
    def train_agent(self):
        self.eval_timesteps = []
        self.eval_returns = []

        # Train the agent until we have completed the max number of training steps
        step = 0
        step_eval = 0

        # Reset the environment for all n_envs in batch
        states, _ = self.env.reset() # Shape (n_envs, state_size)
        old_dones = np.zeros(self.n_envs, dtype=bool)
    
        # Episode length is defined to be from the start until termination/truncation, however, when using a vectorized environment, the dones are reset through this
        # vectorized environment, so we no longer need an inner loop.
        while step < self.n_steps:

            # Update step using the number of envs that take steps
            step += self.n_envs

            # Select epsilon greedy action for batch of states
            actions = self.select_action(states)

            # Give action to env, obtain next state, reward and 'done' (terminate or truncate)
            n_states, rewards, terminated, truncated, _ = self.env.step(actions)
            done = terminated | truncated  # Update done flags for all environments

            # Compute Q-targets for batch
            targets = rewards 

            # Get next Q-values for batch of next states, but first filter out states that were reset in the previous itr, i.e. those are the reset state from which we shouldn't train
            # then, filter out the terminal states
            next_q_values = self.agent(n_states[~old_dones * ~done]).numpy()
            targets[~old_dones * ~done] += self.gamma * np.max(next_q_values, axis=1)

            # Update the Q-values of the selected actions in batch
            target_q_values = self.agent(states).numpy()
            target_q_values[range(len(actions)), actions] = targets
            
            # Perform training step for batch of states -> select the ratio of data-to-update
            slice_up_to = int(states[~old_dones].shape[0] * self.update_to_data_ratio)
            self.agent.fit(
                states[~old_dones][:slice_up_to, :],
                target_q_values[~old_dones][:slice_up_to, :],
                verbose=0,
            )

            # Update state and old_dones
            states = n_states  # Update states to next states
            old_dones = done # Update old dones 

            # Evaluate an episode at timesteps eval_interval
            if step_eval <= step:
                step_eval += self.eval_time
                # Use the evaluate method from the agent
                eval_return, eval_std = self.evaluate()

                # print(f"Step {step}/{self.n_steps}, average reward of evaluated episodes: {eval_return} +- {eval_std}")

                # Add returns and timestep to lists
                self.eval_returns.append(eval_return)
                self.eval_timesteps.append(step)
                

    def evaluate(self):
        returns = []
        ep_rewards = np.zeros(self.n_eval_envs)
        states, _ = self.eval_env.reset() 
        
        # Track the number of evaluated episodes
        eps_done = 0
        while eps_done < self.n_eval_episodes:
            Q_values = self.agent(states)  # Shape (n_envs, action_size)
            actions =  np.argmax(Q_values, axis=1)

            # Obtain next state, rewards and done flags
            n_states, rewards, terminated, truncated, _ = self.eval_env.step(actions)
            done = terminated | truncated
            ep_rewards += rewards
            n_done = sum(done)

            # Check if any episode/env was done
            if n_done > 0: 
                # Check if we have evaluated enough episodes
                if len(returns) + n_done > self.n_eval_episodes:
                    # Add done episode rewards to returns up to only 'n_eval_episodes - len(returns)' such that we only have exactly at most n_eval_episodes of returns
                    returns.extend(ep_rewards[done][:self.n_eval_episodes - len(returns)])
                    
                    # Now break the loop, we have enough evaluated episodes
                    break
                # Add all done episodes as we do not exceed the max allowed
                else:
                    returns.extend(ep_rewards[done])
                    
                # If we are not done yet, increase the number of dones reached
                eps_done += n_done

                # Reset the terminated envs
                ep_rewards[done] = 0
            
            # Move to the nexxt state
            states = n_states
        return np.mean(returns), np.std(returns)

    def save_weights(self, filename="dqn.weights.h5"):
        """Saves the model weights."""
        self.agent.save_weights(filename)

    def load_weights(self, filename="dqn.weights.h5"):
        """Loads previously saved model weights."""
        self.agent.load_weights(filename)

    def reset_weights(self):
        """Resets the agent by reinitializing the neural network weights."""
        self.agent = self.build_model()