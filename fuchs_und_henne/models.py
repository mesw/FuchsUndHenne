"""Actor-Critic Neural Network with Invalid Action Masking for Fuchs und Henne.

Shared Conv2D backbone with separate Actor and Critic heads.
Supports action masking with large negative bias (-1e9) to enforce legal moves.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from fuchs_und_henne.board import NUM_ACTIONS


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic network for Fuchs und Henne."""

    def __init__(self, in_channels: int = 4, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.num_actions = num_actions

        # Feature Extractor: 4x7x7 -> 32x7x7 -> 64x7x7 -> Flatten(3136)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Actor Head: 3136 -> 256 -> 528
        self.actor_head = nn.Sequential(
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        # Critic Head: 3136 -> 128 -> 1 -> Tanh (bounded [-1, 1] for zero-sum outcomes)
        self.critic_head = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute masked action logits and state value.

        Args:
            obs: (B, 4, 7, 7) float tensor
            mask: (B, 528) float/int binary tensor where 1 = legal, 0 = illegal

        Returns:
            (masked_logits, value)
        """
        features = self.conv(obs)
        raw_logits = self.actor_head(features)

        # Apply large negative bias to illegal actions (-1e9)
        masked_logits = raw_logits + (mask.float() - 1.0) * 1e9
        value = self.critic_head(features)
        return masked_logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (or evaluate given action), returning (action, log_prob, entropy, value)."""
        masked_logits, value = self.forward(obs, mask)
        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value only."""
        features = self.conv(obs)
        return self.critic_head(features).squeeze(-1)


class ExportableActor(nn.Module):
    """Stripped-down Actor head for ONNX export and client-side WASM inference."""

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.conv = model.conv
        self.actor_head = model.actor_head

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inference forward pass directly returning action probabilities."""
        features = self.conv(obs)
        logits = self.actor_head(features)
        masked_logits = logits + (mask.float() - 1.0) * 1e9
        return torch.softmax(masked_logits, dim=-1)
