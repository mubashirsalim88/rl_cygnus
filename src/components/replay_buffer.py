import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Experience Replay Buffer for TD3 agent.

    This buffer stores experience tuples (state, action, next_state, reward, done)
    and provides efficient random sampling for training. The buffer implements
    a circular storage mechanism to maintain a fixed maximum size.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1000000):
        """
        Initialize the replay buffer.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.ptr = 0  # Pointer to current position in buffer
        self.size = 0  # Current number of transitions stored

        # Initialize storage arrays
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool
    ) -> None:
        """
        Add a transition to the replay buffer.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state after taking action
            reward: Reward received
            done: Whether the episode terminated
        """
        # Store transition at current pointer position
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to

        Returns:
            Tuple of (states, actions, next_states, rewards, dones) as torch tensors

        Raises:
            ValueError: If batch_size is larger than current buffer size
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer with {self.size} transitions")

        # Randomly sample indices
        indices = np.random.randint(0, self.size, size=batch_size)

        # Convert to torch tensors and move to device
        device_tensor = torch.device(device)

        states = torch.FloatTensor(self.states[indices]).to(device_tensor)
        actions = torch.FloatTensor(self.actions[indices]).to(device_tensor)
        next_states = torch.FloatTensor(self.next_states[indices]).to(device_tensor)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device_tensor)
        dones = torch.FloatTensor(self.dones[indices]).to(device_tensor)

        return states, actions, next_states, rewards, dones

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough transitions for sampling.

        Args:
            batch_size: Required batch size for sampling

        Returns:
            True if buffer contains at least batch_size transitions
        """
        return self.size >= batch_size

    def clear(self) -> None:
        """Clear the replay buffer."""
        self.ptr = 0
        self.size = 0

    def get_buffer_info(self) -> dict:
        """
        Get information about the current buffer state.

        Returns:
            Dictionary containing buffer statistics
        """
        return {
            "current_size": self.size,
            "max_size": self.max_size,
            "fill_percentage": (self.size / self.max_size) * 100,
            "current_pointer": self.ptr
        }