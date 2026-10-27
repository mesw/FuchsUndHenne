"""PPO Rollout Buffer, GAE Advantage Computation, and Clipped Loss Updates."""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fuchs_und_henne.models import ActorCritic


class PPORolloutBuffer:
    """Stores trajectory data for a single agent during self-play rollouts."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.actions)

    def compute_gae_and_returns(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimator (GAE) and target returns."""
        length = len(self.rewards)
        advantages = np.zeros(length, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(length)):
            if t == length - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t + 1])
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return adv_tensor, ret_tensor


def update_ppo(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: PPORolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Execute mini-batch PPO optimization step."""
    device = buffer.device
    obs_t = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.array(buffer.masks), dtype=torch.float32, device=device)
    actions_t = torch.tensor(buffer.actions, dtype=torch.int64, device=device)
    old_log_probs_t = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = len(buffer)
    indices = np.arange(dataset_size)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            b_obs = obs_t[batch_idx]
            b_mask = mask_t[batch_idx]
            b_act = actions_t[batch_idx]
            b_old_log_prob = old_log_probs_t[batch_idx]
            b_adv = advantages[batch_idx]
            b_ret = returns[batch_idx]

            _, log_prob, entropy, value = model.get_action_and_value(
                b_obs, b_mask, action=b_act
            )

            # Ratio & Clipped surrogate objective
            log_ratio = log_prob - b_old_log_prob
            ratio = torch.exp(log_ratio)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = 0.5 * ((value - b_ret) ** 2).mean()

            # Total loss
            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

    return {
        "actor_loss": total_actor_loss / max(1, num_updates),
        "critic_loss": total_critic_loss / max(1, num_updates),
        "entropy": total_entropy / max(1, num_updates),
    }
