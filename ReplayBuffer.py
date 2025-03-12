import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.ptr = 0  # Position pointer
        self.full = False  # Track if buffer has filled

        # Pre-allocate memory
        self.states = np.empty((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.empty((buffer_size,), dtype=np.int32)
        self.rewards = np.empty((buffer_size,), dtype=np.float32)
        self.next_states = np.empty((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.empty((buffer_size,), dtype=np.bool_)
        self.old_dones = np.empty((buffer_size,), dtype=np.bool_)

    def add_experience(self, state, action, reward, next_state, done, old_dones):
        batch_size = len(state)

        if self.ptr + batch_size > self.buffer_size:
            # Roll (shift) the old experiences up to make room
            shift_size = self.ptr + batch_size - self.buffer_size
            self.states[:-shift_size] = self.states[shift_size:]
            self.actions[:-shift_size] = self.actions[shift_size:]
            self.rewards[:-shift_size] = self.rewards[shift_size:]
            self.next_states[:-shift_size] = self.next_states[shift_size:]
            self.dones[:-shift_size] = self.dones[shift_size:]
            self.old_dones[:-shift_size] = self.old_dones[shift_size:]
            self.ptr -= shift_size  # Adjust pointer

        # Insert new experiences (vectors)
        self.states[self.ptr : self.ptr + batch_size] = state
        self.actions[self.ptr : self.ptr + batch_size] = action
        self.rewards[self.ptr : self.ptr + batch_size] = reward
        self.next_states[self.ptr : self.ptr + batch_size] = next_state
        self.dones[self.ptr : self.ptr + batch_size] = done
        self.old_dones[self.ptr: self.ptr + batch_size] = old_dones

        self.ptr += batch_size
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = self.buffer_size  # Keep it at max capacity

    def sample(self, batch_size):
        max_samples = self.buffer_size if self.full else self.ptr
        indices = np.random.choice(max_samples, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.old_dones[indices]
        )
    
    def size(self):
        return self.ptr if not self.full else self.buffer_size
    
    def reset(self):
        self.states = np.empty((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.empty((self.buffer_size,), dtype=np.int32)
        self.rewards = np.empty((self.buffer_size,), dtype=np.float32)
        self.next_states = np.empty((self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.empty((self.buffer_size,), dtype=np.bool_)
        self.old_dones = np.empty((self.buffer_size,), dtype=np.bool_)
        self.ptr = 0  # Position pointer
        self.full = False  # Track if buffer has filled
    

"""Example code

test = ReplayBuffer(100, 2)

states, actions, reward, next_state, done, old_dones = (
    np.array([np.arange(10), np.arange(10)]).T,
    np.zeros(10),
    np.ones(10),
    np.array([np.arange(10), np.arange(10)]).T,
    np.zeros(10, dtype=bool),
    np.zeros(10, dtype=bool),
)

test.add_experience(states, actions, reward, next_state, done,old_dones)
test.sample(10)

"""