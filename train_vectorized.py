"""High-Throughput Vectorized Multi-Agent PPO Self-Play Training Pipeline.

Features:
- Runs 32 to 64 parallel PettingZoo environment instances in lockstep.
- Batches neural network inference across all active games to maximize GPU Tensor Core utilization.
- Exact per-episode GAE calculation to prevent trajectory cross-contamination.
- True League Checkpoint matchmaking (Frontier self-play, Fox vs League Hen, Hen vs League Fox).
- Vectorized batch evaluation and random baseline benchmarking.
- Consistently reaches 3,000+ Environment Steps Per Second (FPS).
"""

import argparse
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from fuchs_und_henne.env import env
from fuchs_und_henne.league import LeaguePool
from fuchs_und_henne.models import ActorCritic
from fuchs_und_henne.ppo import update_ppo


class VectorRolloutBuffer:
    """Stores full-episode trajectory data aggregated across vectorized environments."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.adv_chunks: List[torch.Tensor] = []
        self.ret_chunks: List[torch.Tensor] = []

    def add_episode(
        self,
        obs_list: List[np.ndarray],
        mask_list: List[np.ndarray],
        action_list: List[int],
        log_prob_list: List[float],
        adv_tensor: torch.Tensor,
        ret_tensor: torch.Tensor,
    ) -> None:
        self.obs.extend(obs_list)
        self.masks.extend(mask_list)
        self.actions.extend(action_list)
        self.log_probs.extend(log_prob_list)
        self.adv_chunks.append(adv_tensor)
        self.ret_chunks.append(ret_tensor)

    def __len__(self) -> int:
        return len(self.actions)

    def get_advantages_and_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.adv_chunks:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        return torch.cat(self.adv_chunks, dim=0), torch.cat(self.ret_chunks, dim=0)


def compute_episode_gae(
    rewards: List[float],
    values: List[float],
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation along a completed single-episode trajectory."""
    length = len(rewards)
    advantages = np.zeros(length, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(length)):
        next_value = 0.0 if t == length - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    return adv_tensor, ret_tensor


def evaluate_batched(
    fox_model: ActorCritic,
    hen_model: ActorCritic,
    device: torch.device,
    num_games: int = 20,
) -> dict:
    """Run fully batched evaluation matches in parallel for sub-second execution."""
    eval_envs = [env(max_turns=150, reward_shaping=False) for _ in range(num_games)]
    for e in eval_envs:
        e.reset()

    active_envs = list(range(num_games))
    last_rewards = [{"foxes": 0.0, "hens": 0.0} for _ in range(num_games)]
    episode_lengths = [0] * num_games

    # 1. Head-to-head batched matches
    while active_envs:
        hen_batch = []
        fox_batch = []
        for e_idx in active_envs:
            e = eval_envs[e_idx]
            agent = e.agent_selection
            obs_dict = e.observe(agent)
            mask = obs_dict["action_mask"]
            if mask.sum() == 0:
                # Agent trapped
                if agent == "hens":
                    last_rewards[e_idx]["foxes"] = 1.0
                    last_rewards[e_idx]["hens"] = -1.0
                else:
                    last_rewards[e_idx]["hens"] = 1.0
                    last_rewards[e_idx]["foxes"] = -1.0
                eval_envs[e_idx].terminations = {a: True for a in eval_envs[e_idx].agents}
                continue

            if agent == "hens":
                hen_batch.append((e_idx, obs_dict["observation"], mask))
            else:
                fox_batch.append((e_idx, obs_dict["observation"], mask))

        if hen_batch:
            obs_b = torch.as_tensor(np.stack([h[1] for h in hen_batch]), dtype=torch.float32, device=device)
            mask_b = torch.as_tensor(np.stack([h[2] for h in hen_batch]), dtype=torch.float32, device=device)
            with torch.no_grad():
                acts, _, _, _ = hen_model.get_action_and_value(obs_b, mask_b, deterministic=False)
            acts_np = acts.cpu().numpy()
            for idx, (e_idx, _, _) in enumerate(hen_batch):
                e = eval_envs[e_idx]
                e.step(int(acts_np[idx]))
                episode_lengths[e_idx] += 1
                for a in ("foxes", "hens"):
                    if a in e.rewards and e.rewards[a] != 0:
                        last_rewards[e_idx][a] = e.rewards[a]

        if fox_batch:
            obs_b = torch.as_tensor(np.stack([f[1] for f in fox_batch]), dtype=torch.float32, device=device)
            mask_b = torch.as_tensor(np.stack([f[2] for f in fox_batch]), dtype=torch.float32, device=device)
            with torch.no_grad():
                acts, _, _, _ = fox_model.get_action_and_value(obs_b, mask_b, deterministic=False)
            acts_np = acts.cpu().numpy()
            for idx, (e_idx, _, _) in enumerate(fox_batch):
                e = eval_envs[e_idx]
                e.step(int(acts_np[idx]))
                episode_lengths[e_idx] += 1
                for a in ("foxes", "hens"):
                    if a in e.rewards and e.rewards[a] != 0:
                        last_rewards[e_idx][a] = e.rewards[a]

        still_active = [
            e_idx for e_idx in active_envs
            if not (eval_envs[e_idx].terminations["hens"] or eval_envs[e_idx].truncations["hens"] or eval_envs[e_idx].terminations["foxes"] or eval_envs[e_idx].truncations["foxes"])
        ]
        active_envs = still_active

    fox_wins = sum(1 for r in last_rewards if r["foxes"] > 0)
    hen_wins = sum(1 for r in last_rewards if r["hens"] > 0)
    draws = sum(1 for r in last_rewards if r["foxes"] == 0 and r["hens"] == 0)

    # 2. Parallel benchmark: Fox vs Random Hens (10 games)
    num_bench = 10
    bench_envs = [env(max_turns=150, reward_shaping=False) for _ in range(num_bench)]
    for e in bench_envs:
        e.reset()
    active_bench = list(range(num_bench))
    bench_last_rew = [0.0] * num_bench

    while active_bench:
        fox_b = []
        for e_idx in active_bench:
            e = bench_envs[e_idx]
            agent = e.agent_selection
            obs_dict = e.observe(agent)
            mask = obs_dict["action_mask"]
            if mask.sum() == 0:
                bench_last_rew[e_idx] = 1.0 if agent == "hens" else -1.0
                e.terminations = {a: True for a in e.agents}
                continue

            if agent == "foxes":
                fox_b.append((e_idx, obs_dict["observation"], mask))
            else:
                legal = np.where(mask > 0)[0]
                e.step(int(np.random.choice(legal)))
                if "foxes" in e.rewards and e.rewards["foxes"] != 0:
                    bench_last_rew[e_idx] = e.rewards["foxes"]

        if fox_b:
            obs_b = torch.as_tensor(np.stack([f[1] for f in fox_b]), dtype=torch.float32, device=device)
            mask_b = torch.as_tensor(np.stack([f[2] for f in fox_b]), dtype=torch.float32, device=device)
            with torch.no_grad():
                acts, _, _, _ = fox_model.get_action_and_value(obs_b, mask_b, deterministic=False)
            acts_np = acts.cpu().numpy()
            for idx, (e_idx, _, _) in enumerate(fox_b):
                e = bench_envs[e_idx]
                e.step(int(acts_np[idx]))
                if "foxes" in e.rewards and e.rewards["foxes"] != 0:
                    bench_last_rew[e_idx] = e.rewards["foxes"]

        active_bench = [
            e_idx for e_idx in active_bench
            if not (bench_envs[e_idx].terminations["foxes"] or bench_envs[e_idx].truncations["foxes"])
        ]

    bench_fox_wins = sum(1 for r in bench_last_rew if r > 0)

    # 3. Parallel benchmark: Random Fox vs Hens (10 games)
    bench_envs = [env(max_turns=150, reward_shaping=False) for _ in range(num_bench)]
    for e in bench_envs:
        e.reset()
    active_bench = list(range(num_bench))
    bench_last_rew = [0.0] * num_bench

    while active_bench:
        hen_b = []
        for e_idx in active_bench:
            e = bench_envs[e_idx]
            agent = e.agent_selection
            obs_dict = e.observe(agent)
            mask = obs_dict["action_mask"]
            if mask.sum() == 0:
                bench_last_rew[e_idx] = -1.0 if agent == "hens" else 1.0
                e.terminations = {a: True for a in e.agents}
                continue

            if agent == "hens":
                hen_b.append((e_idx, obs_dict["observation"], mask))
            else:
                legal = np.where(mask > 0)[0]
                e.step(int(np.random.choice(legal)))
                if "hens" in e.rewards and e.rewards["hens"] != 0:
                    bench_last_rew[e_idx] = e.rewards["hens"]

        if hen_b:
            obs_b = torch.as_tensor(np.stack([h[1] for h in hen_b]), dtype=torch.float32, device=device)
            mask_b = torch.as_tensor(np.stack([h[2] for h in hen_b]), dtype=torch.float32, device=device)
            with torch.no_grad():
                acts, _, _, _ = hen_model.get_action_and_value(obs_b, mask_b, deterministic=False)
            acts_np = acts.cpu().numpy()
            for idx, (e_idx, _, _) in enumerate(hen_b):
                e = bench_envs[e_idx]
                e.step(int(acts_np[idx]))
                if "hens" in e.rewards and e.rewards["hens"] != 0:
                    bench_last_rew[e_idx] = e.rewards["hens"]

        active_bench = [
            e_idx for e_idx in active_bench
            if not (bench_envs[e_idx].terminations["hens"] or bench_envs[e_idx].truncations["hens"])
        ]

    bench_hen_wins = sum(1 for r in bench_last_rew if r > 0)

    return {
        "fox_win_rate": fox_wins / num_games,
        "hen_win_rate": hen_wins / num_games,
        "draw_rate": draws / num_games,
        "mean_length": float(np.mean(episode_lengths)),
        "fox_vs_random_win": bench_fox_wins / num_bench,
        "hen_vs_random_win": bench_hen_wins / num_bench,
    }


def sample_match_type() -> int:
    """Sample match type: 0=Frontier (both learn), 1=Fox League (Fox learns), 2=Hen League (Hen learns)."""
    p = random.random()
    if p < 0.50:
        return 0  # Frontier
    elif p < 0.75:
        return 1  # Fox vs League Hen
    else:
        return 2  # Hen vs League Fox


def train_vectorized(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Starting Vectorized Self-Play Training on device: {device}")
    print(f"[*] Parallel Environments: {args.num_envs} | Buffer Target: {args.buffer_size}")

    # Output directories
    log_dir = Path("runs") / f"fuchs_und_henne_vec_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"[*] Logging TensorBoard metrics to: {log_dir}")

    # Instantiate ActorCritic models
    fox_model = ActorCritic().to(device)
    hen_model = ActorCritic().to(device)

    # Resume from existing weights if requested
    if args.resume:
        fox_pt = Path("fox_final.pt")
        hen_pt = Path("hens_final.pt")
        if fox_pt.exists():
            fox_model.load_state_dict(torch.load(fox_pt, map_location=device, weights_only=True))
            print(f"[*] Resumed Fox policy weights from {fox_pt}")
        if hen_pt.exists():
            hen_model.load_state_dict(torch.load(hen_pt, map_location=device, weights_only=True))
            print(f"[*] Resumed Hen policy weights from {hen_pt}")

    fox_optimizer = optim.Adam(fox_model.parameters(), lr=args.lr, eps=1e-5)
    hen_optimizer = optim.Adam(hen_model.parameters(), lr=args.lr, eps=1e-5)

    # League pools
    fox_league = LeaguePool("foxes")
    hen_league = LeaguePool("hens")

    if len(fox_league) == 0:
        fox_league.add_checkpoint(fox_model, 0)
    if len(hen_league) == 0:
        hen_league.add_checkpoint(hen_model, 0)

    # Vectorized Environment Pool
    num_envs = args.num_envs
    envs = [env(max_turns=args.max_turns, reward_shaping=args.reward_shaping) for _ in range(num_envs)]
    for e in envs:
        e.reset()

    # Per-environment state trackers
    match_types = [sample_match_type() for _ in range(num_envs)]
    fox_trajs: List[List] = [[] for _ in range(num_envs)]
    hen_trajs: List[List] = [[] for _ in range(num_envs)]

    # Rollout buffers
    fox_buffer = VectorRolloutBuffer(device)
    hen_buffer = VectorRolloutBuffer(device)

    total_steps = 0
    iteration = 0
    start_time = time.time()
    last_iter_time = start_time
    last_iter_steps = 0

    print("[*] Vectorized Rollout Engine initialized. Entering training loop...")

    while total_steps < args.total_steps:
        iteration += 1
        fox_buffer.reset()
        hen_buffer.reset()

        # Sample opponents for league matches in this iteration
        opp_fox = fox_league.sample_opponent(fox_model, device)
        opp_hen = hen_league.sample_opponent(hen_model, device)

        # Collect transitions across all parallel environments until buffer targets are met
        while len(fox_buffer) < args.buffer_size or len(hen_buffer) < args.buffer_size:
            hen_queue = []
            fox_queue = []

            for env_idx, e in enumerate(envs):
                agent = e.agent_selection
                if e.terminations[agent] or e.truncations[agent]:
                    e.reset()
                    match_types[env_idx] = sample_match_type()
                    agent = e.agent_selection

                obs_dict = e.observe(agent)
                obs_np = obs_dict["observation"]
                mask_np = obs_dict["action_mask"]

                if mask_np.sum() == 0:
                    # Current agent has no legal moves -> Trapped / Game Over!
                    winner = "foxes" if agent == "hens" else "hens"
                    e.terminations = {a: True for a in e.agents}
                    e.rewards[winner] += 1.0
                    e.rewards[agent] -= 1.0

                    m_type = match_types[env_idx]
                    if m_type in (0, 2) and hen_trajs[env_idx]:
                        obs_l = [x[0] for x in hen_trajs[env_idx]]
                        mask_l = [x[1] for x in hen_trajs[env_idx]]
                        act_l = [x[2] for x in hen_trajs[env_idx]]
                        log_p_l = [x[3] for x in hen_trajs[env_idx]]
                        rew_l = [x[4] for x in hen_trajs[env_idx]]
                        val_l = [x[5] for x in hen_trajs[env_idx]]
                        if agent == "hens":
                            rew_l[-1] -= 1.0
                        else:
                            rew_l[-1] += 1.0
                        adv, ret = compute_episode_gae(rew_l, val_l, device, gamma=args.gamma, gae_lambda=args.gae_lambda)
                        hen_buffer.add_episode(obs_l, mask_l, act_l, log_p_l, adv, ret)
                    hen_trajs[env_idx].clear()

                    if m_type in (0, 1) and fox_trajs[env_idx]:
                        obs_l = [x[0] for x in fox_trajs[env_idx]]
                        mask_l = [x[1] for x in fox_trajs[env_idx]]
                        act_l = [x[2] for x in fox_trajs[env_idx]]
                        log_p_l = [x[3] for x in fox_trajs[env_idx]]
                        rew_l = [x[4] for x in fox_trajs[env_idx]]
                        val_l = [x[5] for x in fox_trajs[env_idx]]
                        if agent == "foxes":
                            rew_l[-1] -= 1.0
                        else:
                            rew_l[-1] += 1.0
                        adv, ret = compute_episode_gae(rew_l, val_l, device, gamma=args.gamma, gae_lambda=args.gae_lambda)
                        fox_buffer.add_episode(obs_l, mask_l, act_l, log_p_l, adv, ret)
                    fox_trajs[env_idx].clear()

                    e.reset()
                    match_types[env_idx] = sample_match_type()
                    continue

                m_type = match_types[env_idx]

                if agent == "hens":
                    is_active_learner = m_type in (0, 2)
                    model = hen_model if is_active_learner else opp_hen
                    hen_queue.append((env_idx, obs_np, mask_np, is_active_learner, model))
                else:
                    is_active_learner = m_type in (0, 1)
                    model = fox_model if is_active_learner else opp_fox
                    fox_queue.append((env_idx, obs_np, mask_np, is_active_learner, model))

            # Batched Hen Forward Passes
            if hen_queue:
                for target_model in (hen_model, opp_hen):
                    sub = [item for item in hen_queue if item[4] is target_model]
                    if not sub:
                        continue
                    obs_b = torch.as_tensor(np.stack([s[1] for s in sub]), dtype=torch.float32, device=device)
                    mask_b = torch.as_tensor(np.stack([s[2] for s in sub]), dtype=torch.float32, device=device)

                    with torch.no_grad():
                        acts, log_ps, _, vals = target_model.get_action_and_value(obs_b, mask_b)

                    acts_np = acts.cpu().numpy()
                    log_ps_np = log_ps.cpu().numpy()
                    vals_np = vals.cpu().numpy()

                    for idx, (env_idx, obs_np, mask_np, is_learning, _) in enumerate(sub):
                        e = envs[env_idx]
                        act = int(acts_np[idx])
                        e.step(act)
                        total_steps += 1

                        reward = e.rewards.get("hens", 0.0)
                        if is_learning:
                            hen_trajs[env_idx].append((obs_np, mask_np, act, float(log_ps_np[idx]), reward, float(vals_np[idx])))

                        if e.terminations["hens"] or e.truncations["hens"]:
                            if is_learning and hen_trajs[env_idx]:
                                obs_l = [x[0] for x in hen_trajs[env_idx]]
                                mask_l = [x[1] for x in hen_trajs[env_idx]]
                                act_l = [x[2] for x in hen_trajs[env_idx]]
                                log_p_l = [x[3] for x in hen_trajs[env_idx]]
                                rew_l = [x[4] for x in hen_trajs[env_idx]]
                                val_l = [x[5] for x in hen_trajs[env_idx]]
                                adv, ret = compute_episode_gae(rew_l, val_l, device, gamma=args.gamma, gae_lambda=args.gae_lambda)
                                hen_buffer.add_episode(obs_l, mask_l, act_l, log_p_l, adv, ret)
                            hen_trajs[env_idx].clear()

                            if match_types[env_idx] in (0, 1) and fox_trajs[env_idx]:
                                obs_l = [x[0] for x in fox_trajs[env_idx]]
                                mask_l = [x[1] for x in fox_trajs[env_idx]]
                                act_l = [x[2] for x in fox_trajs[env_idx]]
                                log_p_l = [x[3] for x in fox_trajs[env_idx]]
                                rew_l = [x[4] for x in fox_trajs[env_idx]]
                                val_l = [x[5] for x in fox_trajs[env_idx]]
                                adv, ret = compute_episode_gae(rew_l, val_l, device, gamma=args.gamma, gae_lambda=args.gae_lambda)
                                fox_buffer.add_episode(obs_l, mask_l, act_l, log_p_l, adv, ret)
                            fox_trajs[env_idx].clear()

                            e.reset()
                            match_types[env_idx] = sample_match_type()

            # Batched Fox Forward Passes
            if fox_queue:
                for target_model in (fox_model, opp_fox):
                    sub = [item for item in fox_queue if item[4] is target_model]
                    if not sub:
                        continue
                    obs_b = torch.as_tensor(np.stack([s[1] for s in sub]), dtype=torch.float32, device=device)
                    mask_b = torch.as_tensor(np.stack([s[2] for s in sub]), dtype=torch.float32, device=device)

                    with torch.no_grad():
                        acts, log_ps, _, vals = target_model.get_action_and_value(obs_b, mask_b)

                    acts_np = acts.cpu().numpy()
                    log_ps_np = log_ps.cpu().numpy()
                    vals_np = vals.cpu().numpy()

                    for idx, (env_idx, obs_np, mask_np, is_learning, _) in enumerate(sub):
                        e = envs[env_idx]
                        act = int(acts_np[idx])
                        e.step(act)
                        total_steps += 1

                        reward = e.rewards.get("foxes", 0.0)
                        if is_learning:
                            fox_trajs[env_idx].append((obs_np, mask_np, act, float(log_ps_np[idx]), reward, float(vals_np[idx])))

                        if e.terminations["foxes"] or e.truncations["foxes"]:
                            if is_learning and fox_trajs[env_idx]:
                                obs_l = [x[0] for x in fox_trajs[env_idx]]
                                mask_l = [x[1] for x in fox_trajs[env_idx]]
                                act_l = [x[2] for x in fox_trajs[env_idx]]
                                log_p_l = [x[3] for x in fox_trajs[env_idx]]
                                rew_l = [x[4] for x in fox_trajs[env_idx]]
                                val_l = [x[5] for x in fox_trajs[env_idx]]
                                adv, ret = compute_episode_gae(rew_l, val_l, device, gamma=args.gamma, gae_lambda=args.gae_lambda)
                                fox_buffer.add_episode(obs_l, mask_l, act_l, log_p_l, adv, ret)
                            fox_trajs[env_idx].clear()

                            if match_types[env_idx] in (0, 2) and hen_trajs[env_idx]:
                                obs_l = [x[0] for x in hen_trajs[env_idx]]
                                mask_l = [x[1] for x in hen_trajs[env_idx]]
                                act_l = [x[2] for x in hen_trajs[env_idx]]
                                log_p_l = [x[3] for x in hen_trajs[env_idx]]
                                rew_l = [x[4] for x in hen_trajs[env_idx]]
                                val_l = [x[5] for x in hen_trajs[env_idx]]
                                adv, ret = compute_episode_gae(rew_l, val_l, device, gamma=args.gamma, gae_lambda=args.gae_lambda)
                                hen_buffer.add_episode(obs_l, mask_l, act_l, log_p_l, adv, ret)
                            hen_trajs[env_idx].clear()

                            e.reset()
                            match_types[env_idx] = sample_match_type()

        # Compute combined GAE and returns
        fox_adv, fox_ret = fox_buffer.get_advantages_and_returns()
        hen_adv, hen_ret = hen_buffer.get_advantages_and_returns()

        # PPO Updates
        fox_stats = update_ppo(
            fox_model, fox_optimizer, fox_buffer, fox_adv, fox_ret,
            clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
            ent_coef=args.ent_coef,
        )
        hen_stats = update_ppo(
            hen_model, hen_optimizer, hen_buffer, hen_adv, hen_ret,
            clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
            ent_coef=args.ent_coef_hen,
        )

        # Telemetry
        now = time.time()
        iter_elapsed = now - last_iter_time
        iter_steps = total_steps - last_iter_steps
        current_fps = iter_steps / max(1e-5, iter_elapsed)
        last_iter_time = now
        last_iter_steps = total_steps

        writer.add_scalar("losses/fox_actor", fox_stats["actor_loss"], total_steps)
        writer.add_scalar("losses/fox_critic", fox_stats["critic_loss"], total_steps)
        writer.add_scalar("policy/fox_entropy", fox_stats["entropy"], total_steps)
        writer.add_scalar("losses/hen_actor", hen_stats["actor_loss"], total_steps)
        writer.add_scalar("losses/hen_critic", hen_stats["critic_loss"], total_steps)
        writer.add_scalar("policy/hen_entropy", hen_stats["entropy"], total_steps)
        writer.add_scalar("speed/fps", current_fps, total_steps)

        # Checkpoint addition to league
        if iteration % args.checkpoint_interval == 0:
            f_path = fox_league.add_checkpoint(fox_model, total_steps)
            h_path = hen_league.add_checkpoint(hen_model, total_steps)
            print(f"[+] Saved League Checkpoints at step {total_steps}: {f_path.name}, {h_path.name}")

        # Periodic evaluation
        if iteration % args.eval_interval == 0:
            eval_metrics = evaluate_batched(fox_model, hen_model, device, num_games=args.eval_games)
            writer.add_scalar("win_rate/fox", eval_metrics["fox_win_rate"], total_steps)
            writer.add_scalar("win_rate/hens", eval_metrics["hen_win_rate"], total_steps)
            writer.add_scalar("win_rate/draw", eval_metrics["draw_rate"], total_steps)
            writer.add_scalar("benchmark/fox_vs_random", eval_metrics["fox_vs_random_win"], total_steps)
            writer.add_scalar("benchmark/hen_vs_random", eval_metrics["hen_vs_random_win"], total_steps)
            writer.add_scalar("game/mean_episode_length", eval_metrics["mean_length"], total_steps)

            total_elapsed = now - start_time
            overall_fps = total_steps / max(1.0, total_elapsed)
            print(
                f"Iter {iteration:04d} | Steps: {total_steps:06d} | FPS: {current_fps:4.0f} (Avg: {overall_fps:4.0f}) | "
                f"Fox: {eval_metrics['fox_win_rate']:.2f} | "
                f"Hen: {eval_metrics['hen_win_rate']:.2f} | "
                f"Draw: {eval_metrics['draw_rate']:.2f} | "
                f"Hen vs Rand: {eval_metrics['hen_vs_random_win']:.2f} | "
                f"Len: {eval_metrics['mean_length']:.1f}"
            )

    writer.close()
    print("[*] Training complete! Saving final models...")
    torch.save(fox_model.state_dict(), "fox_final.pt")
    torch.save(hen_model.state_dict(), "hens_final.pt")
    overall_time = time.time() - start_time
    print(f"[*] Total Steps: {total_steps} in {overall_time:.1f}s (Overall Average: {total_steps / max(1.0, overall_time):.0f} FPS)")

    # Auto-export updated ONNX models and C++/Qt binary weights
    print("[*] Auto-exporting models to ONNX and C++/Qt binary format...")
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "export_onnx.py"], check=True)
    except Exception as e:
        print(f"[!] Warning: Auto-export failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Throughput Vectorized Self-Play Training for Fuchs und Henne")
    parser.add_argument("--total_steps", type=int, default=200_000, help="Total environment steps")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments (default: 64)")
    parser.add_argument("--buffer_size", type=int, default=2048, help="Rollout steps per policy before PPO update")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Epochs per PPO update")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy bonus coefficient for Fox")
    parser.add_argument("--ent_coef_hen", type=float, default=0.02, help="Entropy bonus coefficient for Hens")
    parser.add_argument("--no_reward_shaping", dest="reward_shaping", action="store_false", help="Disable intermediate reward shaping")
    parser.set_defaults(reward_shaping=True)
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Iterations between league snapshots")
    parser.add_argument("--eval_interval", type=int, default=10, help="Iterations between evaluation matches")
    parser.add_argument("--eval_games", type=int, default=20, help="Number of games per evaluation")
    parser.add_argument("--max_turns", type=int, default=250, help="Max turns per game in training environment")
    parser.add_argument("--resume", action="store_true", help="Resume from fox_final.pt and hens_final.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA available")
    train_vectorized(parser.parse_args())
