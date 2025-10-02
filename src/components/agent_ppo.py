import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np
import os

class Actor(nn.Module):
    """
    The Actor network learns the policy, mapping states to a probability
    distribution over actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim) # No Tanh activation here
        )

    def forward(self, state):
        action_logits = self.network(state)
        dist = Categorical(logits=action_logits)
        return dist

class Critic(nn.Module):
    """
    The Critic network evaluates the state, providing a value V(s) that helps
    the Actor judge its actions.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Outputs a single scalar state value
        )

    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, K_epochs: int = 4, eps_clip: float = 0.2, device: str = "cuda"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # Initialize Actor and Critic networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # We need an "old" actor network to calculate the policy ratio
        self.actor_old = Actor(state_dim, action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist = self.actor_old(state)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        # Return the integer action and its log probability
        return action.item(), action_log_prob.cpu().numpy()

    def train(self, memory):
        # 1. Get old trajectories from memory
        old_states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        old_actions = torch.tensor(memory['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs'])).to(self.device)
        rewards = torch.FloatTensor(memory['rewards']).to(self.device)
        dones = torch.BoolTensor(memory['dones']).to(self.device)

        # 2. Calculate advantages using GAE
        advantages = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            advantages.insert(0, discounted_reward)

        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            dist = self.actor(old_states)
            action_log_probs = dist.log_prob(old_actions)
            state_values = self.critic(old_states).squeeze()

            # Find the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(action_log_probs - old_log_probs.detach())

            # Find the surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = self.loss_fn(state_values, advantages)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 4. Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_old.load_state_dict(self.actor.state_dict())