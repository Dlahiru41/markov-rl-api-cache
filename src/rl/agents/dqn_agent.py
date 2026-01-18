import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.rl.networks.q_network import QNetworkConfig, create_network
from src.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

@dataclass
class DQNConfig:
    """Hyperparameters for the DQN agent."""
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dueling: bool = True
    
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    
    # RL
    gamma: float = 0.99
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    
    # Buffer
    buffer_size: int = 100000
    batch_size: int = 64
    prioritized_replay: bool = False
    
    # Stability
    target_update_freq: int = 1000
    max_grad_norm: float = 10.0
    
    # Device
    device: str = 'auto' # 'auto', 'cpu', or 'cuda'
    seed: Optional[int] = None

class DQNAgent:
    """Deep Q-Network (DQN) agent for learning caching policies."""
    
    def __init__(self, config: DQNConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed or config.seed
        
        if self.seed is not None:
            self._set_seed(self.seed)
            
        # Device setup
        if config.device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            
        # Networks
        nw_config = QNetworkConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            dueling=config.dueling
        )
        
        self.online_net = create_network(nw_config).to(self.device)
        self.target_net = create_network(nw_config).to(self.device)
        self._update_target_network()
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        # Buffer
        if config.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(capacity=config.buffer_size, seed=self.seed)
        else:
            self.buffer = ReplayBuffer(capacity=config.buffer_size, seed=self.seed)
            
        # State
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        self.last_loss = 0.0
        
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy strategy."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)
            
        # Set network to eval mode for deterministic action selection
        self.online_net.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.online_net.get_action(state_tensor).item()
        self.online_net.train()
        return action

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add experience to replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        
    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one gradient descent step."""
        if not self.buffer.is_ready(self.config.batch_size):
            return None
            
        # Sample batch
        if self.config.prioritized_replay:
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(self.config.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
            weights = None
            indices = None

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute TD errors and loss
        if self.config.prioritized_replay:
            # For prioritized replay, we need element-wise loss to apply weights
            current_q = self.online_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + (self.config.gamma * next_q * (1 - dones))

            # Element-wise TD errors
            td_errors = current_q - target_q

            # Apply importance sampling weights
            weighted_loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            loss = weighted_loss

            # Store TD errors for priority update
            td_errors_abs = torch.abs(td_errors).detach().cpu().numpy().flatten()
        else:
            # Standard loss computation
            loss = self._compute_loss(states, actions, rewards, next_states, dones)

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
            
        self.optimizer.step()
        
        # Update priorities if using PER
        if self.config.prioritized_replay:
            self.buffer.update_priorities(indices, td_errors_abs)

        self.steps_done += 1
        self.last_loss = loss.item()
        
        # Target update
        if self.steps_done % self.config.target_update_freq == 0:
            self._update_target_network()
            
        # Epsilon decay
        self._decay_epsilon()
        
        return {
            'loss': self.last_loss,
            'epsilon': self.epsilon,
            'q_mean': self._get_q_mean(states)
        }
        
    def _compute_loss(self, states, actions, rewards, next_states, dones) -> torch.Tensor:
        """Standard DQN loss: MSE(Q_online(s, a), r + gamma * max(Q_target(s', a')))"""
        # Current Q values
        current_q = self.online_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.config.gamma * next_q * (1 - dones))
            
        return nn.MSELoss()(current_q, target_q)
        
    def _update_target_network(self):
        """Hard update: copy weights from online to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    def _decay_epsilon(self):
        """Exponential decay of exploration rate."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
    def _get_q_mean(self, states) -> float:
        """Helper to track average Q value for debugging."""
        with torch.no_grad():
            return self.online_net(states).mean().item()
            
    def save(self, path: str):
        """Save agent state to file."""
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """Load agent state from file."""
        # torch 2.6+ defaults to weights_only=True, which blocks custom classes like DQNConfig
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older torch versions
            checkpoint = torch.load(path, map_location=self.device)
            
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        # Note: We don't overwrite self.config to allow loading into different compatible configs if needed,
        # but usually they should match.
        
    def get_metrics(self) -> Dict[str, Any]:
        """Return current agent statistics."""
        return {
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            'last_loss': self.last_loss,
            'buffer_size': len(self.buffer),
            'device': str(self.device)
        }

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent to reduce overestimation bias.
    Target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
    """
    
    def _compute_loss(self, states, actions, rewards, next_states, dones) -> torch.Tensor:
        """Double DQN loss calculation."""
        # Current Q values
        current_q = self.online_net(states).gather(1, actions)
        
        # Next Q values
        with torch.no_grad():
            # Use online network to select the best action for next state
            next_actions = self.online_net(next_states).argmax(1).unsqueeze(1)
            # Use target network to evaluate that action
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.config.gamma * next_q * (1 - dones))
            
        return nn.MSELoss()(current_q, target_q)
