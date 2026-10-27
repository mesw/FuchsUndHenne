"""League Checkpoint Buffer for Fox and Hen policies.

Maintains pools of historical checkpoints to prevent cyclic policy forgetting.
Samples opponents with 70% probability from the latest frontier checkpoint
and 30% uniformly from historical checkpoints.
"""

from pathlib import Path
from typing import List, Optional
import copy
import random
import torch

from fuchs_und_henne.models import ActorCritic


class LeaguePool:
    """Manages frozen policy checkpoints for an agent role ('foxes' or 'hens')."""

    def __init__(self, role: str, checkpoint_dir: Optional[Path] = None):
        self.role = role
        self.checkpoint_dir = checkpoint_dir or Path(f"checkpoints/{role}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.in_memory_models: List[ActorCritic] = []
        self.checkpoint_paths: List[Path] = []
        self.load_existing_checkpoints()

    def load_existing_checkpoints(self) -> None:
        """Load any existing .pt checkpoints from disk into the pool."""
        pts = sorted(self.checkpoint_dir.glob(f"{self.role}_step_*.pt"))
        for p in pts:
            model = ActorCritic()
            model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.in_memory_models.append(model)
            self.checkpoint_paths.append(p)
        if self.in_memory_models:
            print(f"[*] LeaguePool [{self.role}]: Loaded {len(self.in_memory_models)} existing checkpoints from disk.")

    def add_checkpoint(self, model: ActorCritic, step: int) -> Path:
        """Save a frozen snapshot of the policy to memory and disk."""
        # 1. In-memory clone (evaluation mode, frozen gradients)
        snapshot = copy.deepcopy(model).cpu()
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad = False
        self.in_memory_models.append(snapshot)

        # 2. Disk checkpoint
        file_path = self.checkpoint_dir / f"{self.role}_step_{step:07d}.pt"
        torch.save(snapshot.state_dict(), file_path)
        self.checkpoint_paths.append(file_path)
        return file_path

    def sample_opponent(self, fallback_model: ActorCritic, device: torch.device) -> ActorCritic:
        """Sample an opponent model using the 70/30 frontier vs. historical policy.

        Returns:
            An ActorCritic model in eval mode on target device.
        """
        if not self.in_memory_models:
            clone = copy.deepcopy(fallback_model).to(device)
            clone.eval()
            for p in clone.parameters():
                p.requires_grad = False
            return clone

        # 70% chance to select the latest checkpoint, 30% uniform random from pool
        if len(self.in_memory_models) == 1 or random.random() < 0.70:
            selected = self.in_memory_models[-1]
        else:
            selected = random.choice(self.in_memory_models)

        return selected.to(device)

    def __len__(self) -> int:
        return len(self.in_memory_models)
