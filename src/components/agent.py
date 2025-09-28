import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .replay_buffer import ReplayBuffer

class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network Architecture.

    This network separates the Q-value estimation into two streams:
    1. A Value stream that estimates the state value, V(s).
    2. An Advantage stream that estimates the advantage for each action, A(s, a).

    These streams are then combined to produce the final Q-values.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the Dueling Q-Network.

        Args:
            state_dim: Dimension of the state space.
            action_dim: Number of discrete actions (e.g., 3 for BUY/HOLD/SELL).
            hidden_dim: Number of neurons in hidden layers.
        """
        super(DuelingQNetwork, self).__init__()
        self.action_dim = action_dim

        # Shared feature learning base
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream head
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Outputs a single scalar V(s)
        )

        # Advantage stream head
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)  # Outputs a vector of A(s, a)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor.

        Returns:
            Q-value tensor for each action.
        """
        features = self.feature_layer(state)

        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage streams to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values

class DuelingDDQNAgent:
    """
    Dueling Double Deep Q-Network Agent.

    Combines Dueling DQN architecture with the Double DQN update rule for stable and efficient learning.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim

        # Initialize online and target Q-networks
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        print(f"Dueling DDQN Agent initialized on device: {self.device}")

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state: The current state observation.
            epsilon: The probability of choosing a random action for exploration.

        Returns:
            A discrete action (integer).
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)  # Explore: random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()  # Exploit: greedy action

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """
        Train the agent using a batch from the replay buffer with the DDQN update rule.
        """
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size, str(self.device))

        # Actions from the buffer are float, convert to long for indexing
        actions = actions.long()

        with torch.no_grad():
            # 1. Select best action for the next state using the online network
            best_next_actions = self.q_network(next_states).argmax(1, keepdim=True)

            # 2. Evaluate that action using the target network to get Q(s', a*)
            target_q_values = self.target_q_network(next_states).gather(1, best_next_actions)

            # 3. Compute the Bellman target: y = r + (1-done) * gamma * Q_target(s', a*)
            y = rewards + (1 - dones) * self.discount * target_q_values

        # Get current Q-values for the actions that were actually taken: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute loss (Mean Squared Error between current and target Q-values)
        loss = F.mse_loss(current_q_values, y)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Soft update the target network
        self._soft_update_target()

        return {"loss": loss.item()}

    def _soft_update_target(self):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, online_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save the model state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load the model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_q_network.eval()
        print(f"Agent loaded from {filepath}")