"""
Experience replay buffers for DQN agent training.

This module provides both uniform and prioritized replay buffers for storing
and sampling past experiences during RL training. Experience replay is crucial
for:
1. Breaking temporal correlations between consecutive samples
2. Reusing past experiences efficiently
3. Stabilizing training dynamics
4. Enabling batch-based gradient updates

The prioritized replay buffer samples experiences with high TD-error more frequently,
which accelerates learning by focusing on surprising or informative transitions.
"""

import numpy as np
import pickle
from collections import namedtuple, deque
from typing import Tuple, Optional, List
import random


# Experience tuple to store transitions
Experience = namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'next_state', 'done']
)


class ReplayBuffer:
    """
    Uniform random sampling replay buffer with fixed capacity.

    Stores experiences (s, a, r, s', done) in a FIFO queue. When full,
    oldest experiences are automatically removed. Sampling is uniform random.

    Key features:
    - Fixed maximum size (memory-bounded)
    - O(1) insertion and O(batch_size) sampling
    - FIFO eviction when full
    - Reproducible sampling with optional seed
    - Save/load for checkpointing

    Attributes:
        capacity: Maximum number of experiences to store
        buffer: Internal deque storing experiences
        seed: Random seed for reproducible sampling
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducible sampling (optional)
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.

        When buffer is full, oldest experience is automatically removed (FIFO).
        States are stored as numpy arrays for memory efficiency.

        Args:
            state: Current state (numpy array)
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next state after action (numpy array)
            done: Whether episode ended (bool)
        """
        # Ensure states are numpy arrays with correct dtype
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        # Ensure consistent float32 dtype for memory efficiency
        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)

        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of experiences.

        Samples uniformly at random without replacement. Returns arrays
        ready for PyTorch (correct dtypes and shapes).

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays:
            - states: shape (batch_size, state_dim), dtype float32
            - actions: shape (batch_size,), dtype int64
            - rewards: shape (batch_size,), dtype float32
            - next_states: shape (batch_size, state_dim), dtype float32
            - dones: shape (batch_size,), dtype float32 (0.0 or 1.0)

        Raises:
            ValueError: If batch_size exceeds current buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer with only {len(self.buffer)} items"
            )

        # Random sample without replacement
        experiences = random.sample(self.buffer, batch_size)

        # Convert to numpy arrays with correct dtypes
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Remove all experiences from buffer."""
        self.buffer.clear()

    def save(self, path: str) -> None:
        """
        Save buffer state to disk.

        Useful for checkpointing during long training runs.

        Args:
            path: File path to save buffer (will use pickle)
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'capacity': self.capacity,
                'buffer': list(self.buffer),
                'seed': self.seed
            }, f)

    def load(self, path: str) -> None:
        """
        Load buffer state from disk.

        Args:
            path: File path to load buffer from

        Raises:
            ValueError: If loaded capacity doesn't match current capacity
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if data['capacity'] != self.capacity:
            raise ValueError(
                f"Loaded buffer capacity {data['capacity']} doesn't match "
                f"current capacity {self.capacity}"
            )

        self.buffer = deque(data['buffer'], maxlen=self.capacity)
        self.seed = data['seed']


class SumTree:
    """
    Binary sum tree for efficient prioritized sampling.

    A sum tree is a binary tree where each node stores the sum of its children's
    values. This allows:
    - O(log n) sampling proportional to priorities
    - O(log n) priority updates
    - O(1) total priority query

    The leaf nodes store the actual priorities, and internal nodes store sums.

    Structure:
        - data_pointer: Current position for adding new data
        - capacity: Maximum number of leaf nodes (experiences)
        - tree: Array storing the tree (size = 2 * capacity - 1)
        - data: Array storing experience data (size = capacity)

    Tree layout for capacity=4:
               0 (root, total sum)
              / \
            1      2
           / \    / \
          3   4  5   6  (leaf nodes with priorities)

    Array indices: parent = i, children = 2*i+1, 2*i+2
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.data_pointer = 0

        # Tree array: internal nodes + leaf nodes
        # For capacity n, we need 2*n - 1 nodes total
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

        # Data array stores actual experiences
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority: float, data: Experience) -> None:
        """
        Add new experience with given priority.

        Args:
            priority: Priority value (typically |TD-error| + epsilon)
            data: Experience tuple to store
        """
        # Calculate tree index for this data position
        tree_idx = self.data_pointer + self.capacity - 1

        # Store data
        self.data[self.data_pointer] = data

        # Update tree with new priority
        self.update(tree_idx, priority)

        # Move pointer (circular buffer)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update priority of a node and propagate changes up the tree.

        Args:
            tree_idx: Index in tree array
            priority: New priority value
        """
        # Calculate change in priority
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up to root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # Parent index
            self.tree[tree_idx] += change

    def get(self, priority_sum: float) -> Tuple[int, float, Experience]:
        """
        Sample an experience by cumulative priority.

        Traverse tree from root to leaf, choosing left or right child based
        on cumulative priorities, until reaching a leaf node.

        Args:
            priority_sum: Random value in [0, total_priority]

        Returns:
            Tuple of (tree_idx, priority, data):
            - tree_idx: Index in tree array
            - priority: Priority of sampled experience
            - data: Experience tuple
        """
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we reach leaf node
            if left_child_idx >= len(self.tree):
                tree_idx = parent_idx
                break

            # Decide left or right based on cumulative priority
            if priority_sum <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                priority_sum -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        # Get data index from tree index
        data_idx = tree_idx - self.capacity + 1

        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Return sum of all priorities (stored at root)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Return maximum priority in tree."""
        # Leaf nodes are in second half of tree
        leaf_start = self.capacity - 1
        leaf_end = len(self.tree)
        return np.max(self.tree[leaf_start:leaf_end])

    @property
    def min_priority(self) -> float:
        """Return minimum non-zero priority in tree."""
        leaf_start = self.capacity - 1
        leaf_end = len(self.tree)
        non_zero = self.tree[leaf_start:leaf_end][self.tree[leaf_start:leaf_end] > 0]
        return np.min(non_zero) if len(non_zero) > 0 else 0.0


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples experiences proportional to their TD-error (how surprising they are).
    This accelerates learning by focusing on the most informative transitions.

    Key concepts:
    - Priority: p_i = |TD-error| + epsilon (small constant to ensure non-zero)
    - Sampling probability: P(i) = p_i^alpha / sum(p_j^alpha)
    - Importance sampling weight: w_i = (N * P(i))^(-beta)
    - Alpha controls how much prioritization (0 = uniform, 1 = full prioritization)
    - Beta controls bias correction (anneals from beta_start to 1.0)

    The importance sampling weights correct for bias introduced by non-uniform
    sampling, which is crucial for convergence guarantees.

    Attributes:
        capacity: Maximum number of experiences
        alpha: Prioritization exponent
        beta: Importance sampling exponent (annealed during training)
        beta_start, beta_end: Beta annealing schedule
        beta_frames: Number of frames to anneal beta over
        frame_count: Current frame for beta annealing
        tree: SumTree for efficient prioritized sampling
        epsilon: Small constant to ensure non-zero priorities
        max_priority: Maximum priority seen (for initializing new experiences)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
        seed: Optional[int] = None
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_frames: Number of frames to anneal beta over
            epsilon: Small constant added to priorities to ensure non-zero
            seed: Random seed for reproducible sampling (optional)
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta_start <= 1:
            raise ValueError(f"Beta_start must be in [0, 1], got {beta_start}")
        if not 0 <= beta_end <= 1:
            raise ValueError(f"Beta_end must be in [0, 1], got {beta_end}")

        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame_count = 0
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        if seed is not None:
            np.random.seed(seed)

    @property
    def beta(self) -> float:
        """
        Current beta value (annealed from beta_start to beta_end).

        Beta annealing ensures that importance sampling correction starts small
        (when priorities are noisy) and increases to full correction by end of training.
        """
        progress = min(1.0, self.frame_count / self.beta_frames)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add a new experience to the buffer.

        New experiences are added with maximum priority seen so far,
        ensuring they are sampled at least once for priority calculation.

        Args:
            state: Current state (numpy array)
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next state after action (numpy array)
            done: Whether episode ended (bool)
            priority: Optional explicit priority (if None, uses max_priority)
        """
        # Ensure states are numpy arrays with correct dtype
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)

        experience = Experience(state, action, reward, next_state, done)

        # Use provided priority or max priority
        if priority is None:
            priority = self.max_priority

        self.tree.add(priority, experience)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Sample a batch of experiences with priorities.

        Samples proportional to priorities and computes importance sampling weights
        to correct for bias. Returns indices for later priority updates.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices):
            - states: shape (batch_size, state_dim), dtype float32
            - actions: shape (batch_size,), dtype int64
            - rewards: shape (batch_size,), dtype float32
            - next_states: shape (batch_size, state_dim), dtype float32
            - dones: shape (batch_size,), dtype float32
            - weights: shape (batch_size,), dtype float32 (importance sampling weights)
            - indices: list of tree indices (for updating priorities later)

        Raises:
            ValueError: If batch_size exceeds current buffer size
        """
        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer with only {len(self)} items"
            )

        # Divide priority range into batch_size segments
        segment_size = self.tree.total_priority / batch_size

        # Storage for results
        indices = []
        priorities = []
        experiences = []

        # Sample one experience from each segment
        for i in range(batch_size):
            # Random value within segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = np.random.uniform(a, b)

            # Get experience from tree
            tree_idx, priority, data = self.tree.get(value)

            indices.append(tree_idx)
            priorities.append(priority)
            experiences.append(data)

        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-beta) / max(w_j)
        # P(i) = p_i^alpha / sum(p_j^alpha)
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total_priority

        # Importance sampling weights
        weights = np.power(len(self) * sampling_probabilities, -self.beta)
        # Normalize by max weight for numerical stability
        weights /= weights.max()
        weights = weights.astype(np.float32)

        # Convert experiences to arrays
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.float32)

        # Increment frame count for beta annealing
        self.frame_count += 1

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities of sampled experiences based on TD-errors.

        Called after computing TD-errors during training. Higher TD-error means
        the experience was more surprising and should be sampled more frequently.

        Args:
            indices: Tree indices from sample() call
            priorities: New priority values (typically |TD-error| + epsilon)
        """
        for idx, priority in zip(indices, priorities):
            # Apply alpha exponent and add epsilon
            priority = (priority + self.epsilon) ** self.alpha

            # Update max priority
            self.max_priority = max(self.max_priority, priority)

            # Update tree
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        # Count non-None entries in data array
        return np.sum(self.tree.data != 0)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer size >= batch_size
        """
        return len(self) >= batch_size

    def save(self, path: str) -> None:
        """
        Save buffer state to disk.

        Args:
            path: File path to save buffer
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'capacity': self.capacity,
                'alpha': self.alpha,
                'beta_start': self.beta_start,
                'beta_end': self.beta_end,
                'beta_frames': self.beta_frames,
                'frame_count': self.frame_count,
                'epsilon': self.epsilon,
                'max_priority': self.max_priority,
                'tree_array': self.tree.tree,
                'tree_data': self.tree.data,
                'data_pointer': self.tree.data_pointer
            }, f)

    def load(self, path: str) -> None:
        """
        Load buffer state from disk.

        Args:
            path: File path to load buffer from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if data['capacity'] != self.capacity:
            raise ValueError(
                f"Loaded buffer capacity {data['capacity']} doesn't match "
                f"current capacity {self.capacity}"
            )

        self.alpha = data['alpha']
        self.beta_start = data['beta_start']
        self.beta_end = data['beta_end']
        self.beta_frames = data['beta_frames']
        self.frame_count = data['frame_count']
        self.epsilon = data['epsilon']
        self.max_priority = data['max_priority']

        # Restore tree state
        self.tree.tree = data['tree_array']
        self.tree.data = data['tree_data']
        self.tree.data_pointer = data['data_pointer']

