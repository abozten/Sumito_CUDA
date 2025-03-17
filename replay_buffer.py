# replay_buffer.py
import numpy as np

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.0001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta  # Importance sampling correction (0 = no correction, 1 = full)
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.next_idx = 0

    def add(self, state, action_idx, reward, next_state, done):
        """Add experience to buffer with maximum priority to ensure it's sampled."""
        experience = (state, action_idx, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience

        # New experiences get max priority to ensure they're sampled
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities."""
        if len(self.buffer) == 0:
            return [], [], []

        # Increase beta for more accurate bias correction over time
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities)

        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, action_indices, rewards, next_states, dones = zip(*samples)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        return (
            np.array(states),
            np.array(action_indices),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Small constant to prevent zero priority
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)