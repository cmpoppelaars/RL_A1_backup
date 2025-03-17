import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam 
from ReplayBuffer import ReplayBuffer

class DQN_ER_TN:
    def __init__(
        self,
        env,
        eval_env,
        TargetNetworkUpdateFq : int, # frequency of updating the target network weights
        batch_size: int,  # size of batch used in replay training
        seed: int = None,  # for debug
        replay_buffer_size: int = 100000,  # size of the replay buffer for training
        learning_rate: float = 0.001,
        gamma: float = 0.9, # discount factor
        epsilon: float = 0.1, # exploration factor
        n_steps: int = 1000000,  # number of steps the agent can take, marking how long it can train
        eval_time: int = 1000,  # number of steps before we evaluate the agent
        n_eval_episodes: int = 30,
        network_size: int = 32,
        update_to_data_ratio: float = 1.0,  # ratio of data from vectorized env that is used for updating/training the agent, between 0 and 1
    ):
        # Environment related parameters
        self.env = env
        self.eval_env = eval_env
        self.n_envs = env.num_envs  # Number of environments to run
        self.n_eval_envs = eval_env.num_envs
        self.state_size = self.env.single_observation_space.shape[0]
        self.action_size = self.env.single_action_space.n
        self.seed = seed

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

        # Initialize replay buffer
        self.ReplayBuffer = ReplayBuffer(replay_buffer_size, self.state_size)
        self.batch_size = batch_size

        # Construct DQN model
        self.MainNetwork = self.build_model()

        # Create target network by copying agent's weights
        self.TargetNetwork = self.build_model()
        self.TargetNetwork.set_weights(self.MainNetwork.get_weights())
        self.TargetNetworkUpdateFq = TargetNetworkUpdateFq

    def build_model(self):
        """Constructs a simple neural network for Q-learning."""
        model = Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(2 * self.network_size, activation="relu"),
                Dense(self.network_size, activation="relu"),
                Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model

    def select_action(self, state):
        """Returns an action based on an epsilon-greedy strategy for batch input."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size, size=self.n_envs)
        else:
            Q_values = self.MainNetwork(state)  # Shape (n_envs, action_size)
            return np.argmax(Q_values, axis=1)

    def experience_replay(self, batch_size):
        # Sample a mini-batch
        (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_n_states,
            sampled_dones,
            sampled_old_dones,
        ) = self.ReplayBuffer.sample(batch_size)

        # Set yj
        yj = sampled_rewards.copy()

        # Update only the non-terminated steps and states that are not the initial state using the TN
        next_q_values = self.TargetNetwork(sampled_n_states[~sampled_old_dones * ~sampled_dones]).numpy()
        yj[~sampled_old_dones * ~sampled_dones] += self.gamma * np.max(next_q_values, axis=1)

        # Update the Q-values of the selected actions in batch from Q values of MN
        target_q_values = self.MainNetwork(sampled_states).numpy()
        target_q_values[range(len(sampled_actions)), sampled_actions] = yj

        # Perform training step for batch of states -> select the ratio of data-to-update
        slice_up_to = int(sampled_states[~sampled_old_dones].shape[0] * self.update_to_data_ratio)

        self.MainNetwork.fit(
            sampled_states[~sampled_old_dones][:slice_up_to, :],
            target_q_values[~sampled_old_dones][:slice_up_to, :],
            verbose=0,
        )

    def train_agent(self):
        self.eval_timesteps = []
        self.eval_returns = []

        # Train the agent until we have completed the max number of training steps
        step = 0
        step_eval = 0
        step_update = 0
        old_dones = np.zeros(self.n_envs, dtype=bool)

        # Reset the environment for all n_envs in batch
        if self.seed:
            states, _ = self.env.reset(seed=self.seed)
        else:
            states, _ = self.env.reset()

        # Episode length is defined to be from the start until termination/truncation, however, when using a vectorized environment, the dones are reset through this
        # vectorized environment, so we no longer need an inner loop.
        while step < self.n_steps:
            # Update step using the number of envs that take steps
            step += self.n_envs

            # NOTE: SAMPLING
            # Select epsilon greedy action for batch of states
            actions = self.select_action(states)

            # Give action to env, obtain next state, reward and 'done' (terminate or truncate)
            n_states, rewards, terminated, truncated, _ = self.env.step(actions)
            done = terminated | truncated  # Update done flags for all environments

            # Add current observation, action, reward, next observation and termination status to the replay buffer.
            self.ReplayBuffer.add_experience(states, actions, rewards, n_states, done, old_dones)

            # NOTE: TRAINING -> apply experience replay
            if self.ReplayBuffer.size() >= self.batch_size:
                self.experience_replay(self.batch_size)

            # Update the target network every C steps -> C the update frequency
            if step_update <= step:
                step_update += self.TargetNetworkUpdateFq
                self.TargetNetwork.set_weights(self.MainNetwork.get_weights())

            # Update states and dones for each environment
            states = n_states  
            old_dones = done

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
        # Keep list to store the returns/rewards on an env, these will be used to returned to the user
        evaluated_episode_returns = []

        # Create empty rewards for active tracking
        episode_rewards_thus_far = np.zeros(self.n_eval_envs)

        # Track the number of evaluated episodes
        number_of_done_epsiodes = 0

        # Set the initial states of all environments
        states, _ = self.eval_env.reset() 

        # Start evaluation
        while number_of_done_epsiodes < self.n_eval_episodes:
            # Obtain the policy greedy actions
            Q_values = self.MainNetwork(states)  # Shape (n_envs, action_size)
            actions = np.argmax(Q_values, axis=1)

            # Obtain next state, rewards and done flags
            n_states, rewards, terminated, truncated, _ = self.eval_env.step(actions)
            done = terminated | truncated
            
            # Update the reward of each episode thus far
            episode_rewards_thus_far += rewards
            
            # Check how many environments were terminated or truncated
            number_of_dones = sum(done)

            # Check if any episode/env was done
            if number_of_dones > 0:
                # Check if we have evaluated enough episodes
                if len(evaluated_episode_returns) + number_of_dones > self.n_eval_episodes:
                    # Add done episode rewards to returns up to only 'n_eval_episodes - len(evaluated_episode_returns)' such that we only have exactly at most n_eval_episodes of returns
                    evaluated_episode_returns.extend(
                        episode_rewards_thus_far[done][: self.n_eval_episodes - len(evaluated_episode_returns)]
                    )

                    # Now break the loop, we have enough evaluated episodes
                    break

                # Add all done episodes as we do not exceed the max allowed
                else:
                    evaluated_episode_returns.extend(episode_rewards_thus_far[done])

                # If we are not done yet, increase the number of dones reached
                number_of_done_epsiodes += number_of_dones

                # Reset the terminated envs
                episode_rewards_thus_far[done] = 0

            # Move to the next state
            states = n_states

        # Return the mean and the std, the latter is for debugging helpful.
        return np.mean(evaluated_episode_returns), np.std(evaluated_episode_returns)

    def save_weights(self, filename="dqn.weights.h5"):
        """Saves the model weights."""
        self.MainNetwork.save_weights(filename)

    def load_weights(self, filename="dqn.weights.h5"):
        """Loads previously saved model weights."""
        self.MainNetwork.load_weights(filename)

    def reset_weights(self):
        """Resets the agent by reinitializing the neural network weights."""
        # Reset the main network
        self.MainNetwork = self.build_model()
        # Copy the weights to effectively reset the target network
        self.TargetNetwork.set_weights(self.MainNetwork.get_weights())
        # Empty the replay buffer
        self.ReplayBuffer.reset()