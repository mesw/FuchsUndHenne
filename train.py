"""Main PPO Self-Play Training Pipeline with League Checkpointing and TensorBoard Telemetry."""

import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from fuchs_und_henne.env import env
from fuchs_und_henne.league import LeaguePool
from fuchs_und_henne.models import ActorCritic
from fuchs_und_henne.ppo import PPORolloutBuffer, update_ppo


def evaluate(
    fox_model: ActorCritic,
    hen_model: ActorCritic,
    device: torch.device,
    num_games: int = 20,
) -> dict:
    """Run evaluation matches: head-to-head (stochastic & deterministic) and vs Random baseline."""
    eval_env = env(max_turns=150, reward_shaping=False)
    fox_wins = 0
    hen_wins = 0
    draws = 0
    episode_lengths = []

    # 1. Head-to-head matches (mix of stochastic and deterministic)
    for g in range(num_games):
        deterministic = (g == 0)  # first game deterministic to monitor greedy loop behavior
        eval_env.reset()
        steps = 0
        last_rewards = {"foxes": 0.0, "hens": 0.0}

        for agent in eval_env.agent_iter():
            obs_dict = eval_env.observe(agent)
            obs = torch.tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(obs_dict["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)

            if eval_env.terminations[agent] or eval_env.truncations[agent] or mask.sum() == 0:
                action = None
            else:
                model = fox_model if agent == "foxes" else hen_model
                with torch.no_grad():
                    action_t, _, _, _ = model.get_action_and_value(obs, mask, deterministic=deterministic)
                    action = action_t.item()

            eval_env.step(action)
            steps += 1

            for a in ("foxes", "hens"):
                if a in eval_env.rewards and eval_env.rewards[a] != 0:
                    last_rewards[a] = eval_env.rewards[a]

        episode_lengths.append(steps)
        if last_rewards["foxes"] > 0:
            fox_wins += 1
        elif last_rewards["hens"] > 0:
            hen_wins += 1
        else:
            draws += 1

    # 2. Benchmark vs Random Opponents (10 games each)
    num_bench = 10
    fox_vs_rand_wins = 0
    for _ in range(num_bench):
        eval_env.reset()
        last_rew = 0.0
        for agent in eval_env.agent_iter():
            obs_dict = eval_env.observe(agent)
            mask = obs_dict["action_mask"]
            if eval_env.terminations[agent] or eval_env.truncations[agent] or mask.sum() == 0:
                action = None
            elif agent == "foxes":
                obs_t = torch.tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    act_t, _, _, _ = fox_model.get_action_and_value(obs_t, mask_t, deterministic=False)
                    action = act_t.item()
            else:
                legal = np.where(mask > 0)[0]
                action = int(np.random.choice(legal))
            eval_env.step(action)
            if "foxes" in eval_env.rewards and eval_env.rewards["foxes"] != 0:
                last_rew = eval_env.rewards["foxes"]
        if last_rew > 0:
            fox_vs_rand_wins += 1

    hen_vs_rand_wins = 0
    for _ in range(num_bench):
        eval_env.reset()
        last_rew = 0.0
        for agent in eval_env.agent_iter():
            obs_dict = eval_env.observe(agent)
            mask = obs_dict["action_mask"]
            if eval_env.terminations[agent] or eval_env.truncations[agent] or mask.sum() == 0:
                action = None
            elif agent == "hens":
                obs_t = torch.tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    act_t, _, _, _ = hen_model.get_action_and_value(obs_t, mask_t, deterministic=False)
                    action = act_t.item()
            else:
                legal = np.where(mask > 0)[0]
                action = int(np.random.choice(legal))
            eval_env.step(action)
            if "hens" in eval_env.rewards and eval_env.rewards["hens"] != 0:
                last_rew = eval_env.rewards["hens"]
        if last_rew > 0:
            hen_vs_rand_wins += 1

    return {
        "fox_win_rate": fox_wins / num_games,
        "hen_win_rate": hen_wins / num_games,
        "draw_rate": draws / num_games,
        "mean_length": float(np.mean(episode_lengths)),
        "fox_vs_random_win": fox_vs_rand_wins / num_bench,
        "hen_vs_random_win": hen_vs_rand_wins / num_bench,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Training on device: {device}")

    # Output directories
    log_dir = Path("runs") / f"fuchs_und_henne_{int(time.time())}"
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

    # League pools (loads existing disk checkpoints automatically)
    fox_league = LeaguePool("foxes")
    hen_league = LeaguePool("hens")

    # Initial checkpoint save if empty
    if len(fox_league) == 0:
        fox_league.add_checkpoint(fox_model, 0)
    if len(hen_league) == 0:
        hen_league.add_checkpoint(hen_model, 0)

    # Rollout buffers
    fox_buffer = PPORolloutBuffer(args.buffer_size, device)
    hen_buffer = PPORolloutBuffer(args.buffer_size, device)

    training_env = env(max_turns=args.max_turns, reward_shaping=args.reward_shaping)
    total_steps = 0
    iteration = 0

    print("[*] Starting PPO Self-Play Training Loop...")
    start_time = time.time()

    while total_steps < args.total_steps:
        iteration += 1
        fox_buffer.reset()
        hen_buffer.reset()

        # Collect trajectories until both buffers are full
        while len(fox_buffer) < args.buffer_size or len(hen_buffer) < args.buffer_size:
            fox_needs = len(fox_buffer) < args.buffer_size
            hen_needs = len(hen_buffer) < args.buffer_size

            # Matchup selection for this episode
            if fox_needs and hen_needs:
                p = random.random()
                if p < 0.50:
                    # Active frontier self-play: both policies train
                    curr_fox, curr_hen = fox_model, hen_model
                    fox_learning, hen_learning = True, True
                elif p < 0.75:
                    # Fox trains against sampled league Hen opponent
                    curr_fox = fox_model
                    curr_hen = hen_league.sample_opponent(hen_model, device)
                    fox_learning, hen_learning = True, False
                else:
                    # Hen trains against sampled league Fox opponent
                    curr_fox = fox_league.sample_opponent(fox_model, device)
                    curr_hen = hen_model
                    fox_learning, hen_learning = False, True
            elif fox_needs:
                curr_fox = fox_model
                curr_hen = hen_league.sample_opponent(hen_model, device)
                fox_learning, hen_learning = True, False
            else:
                curr_fox = fox_league.sample_opponent(fox_model, device)
                curr_hen = hen_model
                fox_learning, hen_learning = False, True

            training_env.reset()

            for agent in training_env.agent_iter():
                obs_dict = training_env.observe(agent)
                obs_np = obs_dict["observation"]
                mask_np = obs_dict["action_mask"]

                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)

                if (
                    training_env.terminations[agent]
                    or training_env.truncations[agent]
                    or mask_np.sum() == 0
                ):
                    action = None
                    training_env.step(action)
                    continue

                is_active_fox = agent == "foxes"
                model = curr_fox if is_active_fox else curr_hen
                is_learning = fox_learning if is_active_fox else hen_learning
                buffer = fox_buffer if is_active_fox else hen_buffer

                with torch.no_grad():
                    action_t, log_prob_t, _, value_t = model.get_action_and_value(obs_t, mask_t)

                action = action_t.item()
                training_env.step(action)
                total_steps += 1

                reward = training_env.rewards.get(agent, 0.0)
                done = training_env.terminations.get(agent, False) or training_env.truncations.get(agent, False)

                if is_learning and len(buffer) < args.buffer_size:
                    buffer.add(
                        obs=obs_np,
                        mask=mask_np,
                        action=action,
                        log_prob=log_prob_t.item(),
                        reward=reward,
                        value=value_t.item(),
                        done=done,
                    )

        # Compute GAE and returns
        fox_adv, fox_ret = fox_buffer.compute_gae_and_returns(
            last_value=0.0, last_done=True, gamma=args.gamma, gae_lambda=args.gae_lambda
        )
        hen_adv, hen_ret = hen_buffer.compute_gae_and_returns(
            last_value=0.0, last_done=True, gamma=args.gamma, gae_lambda=args.gae_lambda
        )

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

        # TensorBoard logging
        writer.add_scalar("losses/fox_actor", fox_stats["actor_loss"], total_steps)
        writer.add_scalar("losses/fox_critic", fox_stats["critic_loss"], total_steps)
        writer.add_scalar("policy/fox_entropy", fox_stats["entropy"], total_steps)
        writer.add_scalar("losses/hen_actor", hen_stats["actor_loss"], total_steps)
        writer.add_scalar("losses/hen_critic", hen_stats["critic_loss"], total_steps)
        writer.add_scalar("policy/hen_entropy", hen_stats["entropy"], total_steps)

        # Checkpoint addition to league
        if iteration % args.checkpoint_interval == 0:
            f_path = fox_league.add_checkpoint(fox_model, total_steps)
            h_path = hen_league.add_checkpoint(hen_model, total_steps)
            print(f"[+] Saved League Checkpoints at step {total_steps}: {f_path.name}, {h_path.name}")

        # Periodic evaluation
        if iteration % args.eval_interval == 0:
            eval_metrics = evaluate(fox_model, hen_model, device, num_games=args.eval_games)
            writer.add_scalar("win_rate/fox", eval_metrics["fox_win_rate"], total_steps)
            writer.add_scalar("win_rate/hens", eval_metrics["hen_win_rate"], total_steps)
            writer.add_scalar("win_rate/draw", eval_metrics["draw_rate"], total_steps)
            writer.add_scalar("benchmark/fox_vs_random", eval_metrics["fox_vs_random_win"], total_steps)
            writer.add_scalar("benchmark/hen_vs_random", eval_metrics["hen_vs_random_win"], total_steps)
            writer.add_scalar("game/mean_episode_length", eval_metrics["mean_length"], total_steps)

            elapsed = time.time() - start_time
            fps = total_steps / max(1, elapsed)
            print(
                f"Iter {iteration:04d} | Steps: {total_steps:06d} | FPS: {fps:.0f} | "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Self-Play Training for Fuchs und Henne")
    parser.add_argument("--total_steps", type=int, default=100_000, help="Total environment steps")
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
    parser.add_argument("--eval_interval", type=int, default=5, help="Iterations between evaluation matches")
    parser.add_argument("--eval_games", type=int, default=20, help="Number of games per evaluation")
    parser.add_argument("--max_turns", type=int, default=250, help="Max turns per game in training environment")
    parser.add_argument("--resume", action="store_true", help="Resume from fox_final.pt and hens_final.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA available")
    train(parser.parse_args())
