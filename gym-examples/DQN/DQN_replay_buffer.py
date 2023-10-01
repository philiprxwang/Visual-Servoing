import numpy as np

class ReplayBuffer:

    def __init__(self, size):

        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, state, action, reward, done, next_state):
        
        data = (state, action, reward, next_state, int(done))

        if self._next_idx >= len(self._storage): # should it be <=??
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy = False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy = False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self._storage)-1, size = batch_size)
        return self._encode_sample(indices)