"""Unit tests and PettingZoo API validation for FuchsUndHenneEnv."""

import numpy as np
import pytest
from pettingzoo.test import api_test

from fuchs_und_henne.env import FuchsUndHenneEnv, env


def test_pettingzoo_api():
    """Verify that the environment passes PettingZoo's standard API test suite."""
    test_env = env(max_turns=30)
    api_test(test_env, num_cycles=100)


def test_random_rollout():
    """Run full episode rollouts with random legal action selection."""
    test_env = env(max_turns=60)
    test_env.reset(seed=42)

    total_steps = 0
    last_rewards = {"hens": 0.0, "foxes": 0.0}

    for agent in test_env.agent_iter():
        obs = test_env.observe(agent)
        mask = obs["action_mask"]

        if test_env.terminations[agent] or test_env.truncations[agent]:
            action = None
        else:
            legal_actions = np.where(mask == 1)[0]
            assert len(legal_actions) > 0, f"No legal moves for {agent}!"
            action = np.random.choice(legal_actions)

        test_env.step(action)
        total_steps += 1

        for a in ("hens", "foxes"):
            if a in test_env.rewards and test_env.rewards[a] != 0:
                last_rewards[a] = test_env.rewards[a]

        if total_steps > 300:
            break

    # Ensure cumulative rewards are within expected bounds
    assert total_steps > 0
    assert last_rewards["hens"] in [-1.0, 0.0, 1.0]
    assert last_rewards["foxes"] in [-1.0, 0.0, 1.0]


def test_reward_shaping():
    """Verify that intermediate reward shaping operates correctly."""
    shaped_env = env(max_turns=50, reward_shaping=True)
    shaped_env.reset(seed=123)

    received_non_zero = False
    for agent in shaped_env.agent_iter():
        obs = shaped_env.observe(agent)
        mask = obs["action_mask"]
        if shaped_env.terminations[agent] or shaped_env.truncations[agent]:
            action = None
        else:
            legal = np.where(mask == 1)[0]
            action = int(np.random.choice(legal))

        shaped_env.step(action)
        if any(r != 0.0 for r in shaped_env.rewards.values()):
            received_non_zero = True

        if shaped_env.terminations[agent] or shaped_env.truncations[agent]:
            break

    assert received_non_zero, "Expected intermediate shaped rewards during rollout"


