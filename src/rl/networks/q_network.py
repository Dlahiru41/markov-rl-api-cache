"""
Q-Network architectures for Deep Q-Learning.

This module provides neural network architectures for approximating Q-values:
1. QNetwork: Standard DQN architecture
2. DuelingQNetwork: Dueling DQN with separate value and advantage streams

Both architectures support configurable hidden layers, activations, dropout,
and layer normalization for flexible experimentation.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List


@dataclass
class QNetworkConfig:
    """
    Configuration for Q-Network architecture.

    Attributes:
        state_dim: Input dimension (state vector size)
        action_dim: Output dimension (number of actions)
        hidden_dims: List of hidden layer sizes
        activation: Activation function ('relu', 'leaky_relu', 'elu', 'tanh')
        dropout: Dropout rate for regularization (0.0 = no dropout)
        use_layer_norm: Whether to use layer normalization
        dueling: Whether to use dueling architecture
    """
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = 'relu'
    dropout: float = 0.1
    use_layer_norm: bool = False
    dueling: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in ['relu', 'leaky_relu', 'elu', 'tanh']:
            raise ValueError(f"activation must be one of ['relu', 'leaky_relu', 'elu', 'tanh'], got {self.activation}")


class QNetwork(nn.Module):
    """
    Standard Deep Q-Network architecture.

    Architecture:
        Input (state_dim)
        → Hidden layers with activation + optional dropout + optional layer norm
        → Output (action_dim) with no activation

    The network approximates Q(s, a) for all actions simultaneously.
    """

    def __init__(self, config: QNetworkConfig):
        """
        Initialize Q-Network.

        Args:
            config: Network configuration
        """
        super(QNetwork, self).__init__()

        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim

        # Build network layers
        layers = []
        input_dim = config.state_dim

        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Layer normalization (before activation)
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation function
            layers.append(self._get_activation(config.activation))

            # Dropout
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))

            input_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(input_dim, config.action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, method='xavier'))

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values for all actions, shape (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get greedy action (argmax of Q-values).

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action indices, shape (batch_size,) or scalar
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=-1)

    def __repr__(self) -> str:
        """String representation of the network."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"QNetwork(\n"
            f"  state_dim={self.state_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  hidden_dims={self.config.hidden_dims},\n"
            f"  activation='{self.config.activation}',\n"
            f"  dropout={self.config.dropout},\n"
            f"  layer_norm={self.config.use_layer_norm},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture.

    Key insight: Decompose Q(s, a) = V(s) + A(s, a)
    - V(s): Value of being in state s
    - A(s, a): Advantage of taking action a over the average

    Architecture:
        Input (state_dim)
        → Shared feature extraction layers
        → Split into two streams:
          1. Value stream → V(s) (single output)
          2. Advantage stream → A(s, a) (action_dim outputs)
        → Combine: Q(s, a) = V(s) + A(s, a) - mean(A(s, :))

    The mean subtraction ensures identifiability and stabilizes learning.
    """

    def __init__(self, config: QNetworkConfig):
        """
        Initialize Dueling Q-Network.

        Args:
            config: Network configuration
        """
        super(DuelingQNetwork, self).__init__()

        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim

        # Shared feature extraction layers
        shared_layers = []
        input_dim = config.state_dim

        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            shared_layers.append(nn.Linear(input_dim, hidden_dim))

            # Layer normalization
            if config.use_layer_norm:
                shared_layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            shared_layers.append(self._get_activation(config.activation))

            # Dropout
            if config.dropout > 0.0:
                shared_layers.append(nn.Dropout(config.dropout))

            input_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Last hidden dimension is input to value and advantage streams
        last_hidden_dim = config.hidden_dims[-1]

        # Value stream: outputs single value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(last_hidden_dim, last_hidden_dim // 2),
            self._get_activation(config.activation),
            nn.Linear(last_hidden_dim // 2, 1)
        )

        # Advantage stream: outputs advantage for each action A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_hidden_dim, last_hidden_dim // 2),
            self._get_activation(config.activation),
            nn.Linear(last_hidden_dim // 2, config.action_dim)
        )

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, method='xavier'))

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling network.

        Computes Q(s, a) = V(s) + A(s, a) - mean(A(s, :))

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values for all actions, shape (batch_size, action_dim) or (action_dim,)
        """
        # Shared feature extraction
        features = self.shared_network(state)

        # Value and advantage streams
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Combine with mean subtraction for identifiability
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value V(s).

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            State values, shape (batch_size, 1) or (1,)
        """
        features = self.shared_network(state)
        return self.value_stream(features)

    def get_advantage(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action advantages A(s, a).

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Advantages for all actions, shape (batch_size, action_dim) or (action_dim,)
        """
        features = self.shared_network(state)
        return self.advantage_stream(features)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get greedy action (argmax of Q-values).

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action indices, shape (batch_size,) or scalar
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=-1)

    def __repr__(self) -> str:
        """String representation of the network."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"DuelingQNetwork(\n"
            f"  state_dim={self.state_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  hidden_dims={self.config.hidden_dims},\n"
            f"  activation='{self.config.activation}',\n"
            f"  dropout={self.config.dropout},\n"
            f"  layer_norm={self.config.use_layer_norm},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )


def create_network(config: QNetworkConfig) -> nn.Module:
    """
    Factory function to create appropriate network type based on configuration.

    Args:
        config: Network configuration

    Returns:
        QNetwork or DuelingQNetwork instance
    """
    if config.dueling:
        return DuelingQNetwork(config)
    else:
        return QNetwork(config)


def initialize_weights(module: nn.Module, method: str = 'xavier') -> None:
    """
    Initialize network weights for better training stability.

    Args:
        module: PyTorch module to initialize
        method: Initialization method ('xavier', 'he', 'orthogonal')
    """
    if isinstance(module, nn.Linear):
        if method == 'xavier':
            # Xavier/Glorot initialization - good for tanh/sigmoid
            nn.init.xavier_uniform_(module.weight)
        elif method == 'he':
            # He initialization - good for ReLU
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif method == 'orthogonal':
            # Orthogonal initialization - good for RNNs
            nn.init.orthogonal_(module.weight)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        # Initialize bias to small constant
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, nn.LayerNorm):
        # Layer norm: weight=1, bias=0
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module) -> str:
    """
    Get a detailed summary of model architecture and parameters.

    Args:
        model: PyTorch model

    Returns:
        String summary of the model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    summary = []
    summary.append("=" * 70)
    summary.append("Model Summary")
    summary.append("=" * 70)
    summary.append(str(model))
    summary.append("-" * 70)
    summary.append(f"Total parameters: {total_params:,}")
    summary.append(f"Trainable parameters: {trainable_params:,}")
    summary.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
    summary.append("=" * 70)

    return "\n".join(summary)

