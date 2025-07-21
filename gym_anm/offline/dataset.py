import numpy as np


class DataBuffer:
    """Simple container storing offline transitions."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(np.asarray(state))
        self.actions.append(np.asarray(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state))
        self.dones.append(bool(done))

    def arrays(self):
        return (
            np.asarray(self.states),
            np.asarray(self.actions),
            np.asarray(self.rewards),
            np.asarray(self.next_states),
            np.asarray(self.dones),
        )

    def __len__(self):
        return len(self.states)
