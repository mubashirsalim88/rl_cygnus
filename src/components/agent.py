import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
import copy
from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """
    Actor network for TD3 agent.

    This network represents the policy that maps states to actions.
    Uses a multi-layer perceptron architecture with tanh output activation
    to ensure actions are bounded between -1 and 1.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the Actor network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Number of neurons in hidden layers
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Actor network.

        Args:
            state: Input state tensor

        Returns:
            Action tensor bounded between -1 and 1
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))

        return action


class Critic(nn.Module):
    """
    Critic network for TD3 agent.

    This network estimates the Q-value (expected cumulative reward) for
    state-action pairs. Takes both state and action as input and outputs
    a single Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the Critic network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Number of neurons in hidden layers
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # First hidden layer processes state only
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # Second hidden layer processes concatenated state and action
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)

        # Output layer produces Q-value
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Critic network.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            Q-value tensor
        """
        # Process state through first hidden layer
        x = F.relu(self.fc1(state))

        # Concatenate processed state with action
        x = torch.cat([x, action], dim=1)

        # Process through remaining layers
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.

    TD3 is an actor-critic algorithm that addresses the overestimation bias
    in deep Q-learning by using twin critics and delayed policy updates.

    Key features:
    - Twin critics to reduce overestimation bias
    - Target networks for stable learning
    - Delayed policy updates
    - Target policy smoothing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the TD3 agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Number of neurons in hidden layers
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic networks
            discount: Discount factor (gamma) for future rewards
            tau: Soft update parameter for target networks
            policy_noise: Noise scale for target policy smoothing
            noise_clip: Noise clipping range for target policy smoothing
            policy_delay: Frequency of delayed policy updates
            device: Device to run computations on (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = torch.device(device)

        # Initialize Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Initialize twin Critic networks
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Ensure target networks start with same weights
        self._hard_update_targets()

        # Training step counter for delayed policy updates
        self.total_it = 0

    def _hard_update_targets(self) -> None:
        """Hard update target networks (copy weights exactly)."""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def _soft_update_targets(self) -> None:
        """
        Soft update target networks using exponential moving average.

        �_target = � * �_current + (1 - �) * �_target
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state: np.ndarray, add_noise: bool = False, noise_scale: float = 0.1) -> np.ndarray:
        """
        Select action using the actor network.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            noise_scale: Scale of exploration noise

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor)

            if add_noise:
                noise = torch.normal(0, noise_scale, size=action.shape).to(self.device)
                action = action + noise
                action = torch.clamp(action, -1, 1)

        return action.cpu().numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """
        Train the TD3 agent using a batch of transitions from the replay buffer.

        Args:
            replay_buffer: Replay buffer containing experience transitions
            batch_size: Number of transitions to sample for training

        Returns:
            Dictionary containing training metrics

        Raises:
            ValueError: If replay buffer doesn't have enough transitions
        """
        # Check if buffer has enough transitions
        if not replay_buffer.is_ready(batch_size):
            raise ValueError(f"Replay buffer has {len(replay_buffer)} transitions, need {batch_size}")

        self.total_it += 1

        # Sample batch of transitions from replay buffer
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size, str(self.device))

        with torch.no_grad():
            # Target policy smoothing: Add clipped noise to target actions
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)

            # Compute target Q-values using Clipped Double Q-Learning
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Calculate target Q-value: r + γ * (1 - done) * min(Q1, Q2)
            target_q = rewards + (1 - dones) * self.discount * target_q

        # Current Q-value estimates
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Critic losses (Mean Squared Error)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update Critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update Critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Prepare training metrics
        training_metrics = {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "mean_q1": current_q1.mean().item(),
            "mean_q2": current_q2.mean().item(),
            "mean_target_q": target_q.mean().item(),
            "total_iterations": self.total_it
        }

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Actor loss: -mean(Q1(s, actor(s)))
            # We want to maximize Q-value, so we minimize the negative Q-value
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            # Update Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update_targets()

            # Add actor metrics
            training_metrics.update({
                "actor_loss": actor_loss.item(),
                "policy_updated": True
            })
        else:
            training_metrics["policy_updated"] = False

        return training_metrics

    def get_network_info(self) -> dict:
        """
        Get information about the network architecture.

        Returns:
            Dictionary containing network information
        """
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic1.parameters())

        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "actor_parameters": actor_params,
            "critic_parameters": critic_params,
            "total_parameters": actor_params + 2 * critic_params,
            "device": str(self.device)
        }

    def save(self, filepath: str) -> None:
        """
        Save the agent's networks to disk.

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'total_it': self.total_it,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'discount': self.discount,
                'tau': self.tau,
                'policy_noise': self.policy_noise,
                'noise_clip': self.noise_clip,
                'policy_delay': self.policy_delay
            }
        }, filepath)

    def load(self, filepath: str) -> None:
        """
        Load the agent's networks from disk.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        self.total_it = checkpoint['total_it']

    def set_eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.actor_target.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

    def set_train_mode(self) -> None:
        """Set all networks to training mode."""
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.actor_target.train()
        self.critic1_target.train()
        self.critic2_target.train()