"""Unit tests for Fuchs und Henne board topology, move generator, and rules."""

import numpy as np
import pytest

from fuchs_und_henne.board import (
    BOARD_MASK,
    COORD_TO_NODE,
    HEN_ALLOWED_DIRECTIONS,
    NODE_COORDS,
    NUM_ACTIONS,
    NUM_NODES,
    STALL_NODES,
    START_FOX_NODES,
    START_HEN_NODES,
    STEP_TARGET,
    apply_action,
    check_game_over,
    decode_action,
    encode_action,
    get_action_mask,
    get_legal_actions,
    get_observation_tensor,
)


def test_board_topology():
    assert len(NODE_COORDS) == 33
    assert NUM_NODES == 33
    assert BOARD_MASK.sum() == 33.0
    assert len(STALL_NODES) == 9
    assert NUM_ACTIONS == 528


def test_initial_pieces():
    assert len(START_FOX_NODES) == 2
    assert len(START_HEN_NODES) == 20
    # No overlap
    assert set(START_FOX_NODES).isdisjoint(set(START_HEN_NODES))
    # Foxes are at (2,2) and (2,4)
    assert NODE_COORDS[START_FOX_NODES[0]] == (2, 2)
    assert NODE_COORDS[START_FOX_NODES[1]] == (2, 4)


def test_diagonal_lattice_constraints():
    # Diagonal inside Stall: (0,2) [node 0] to (1,3) [node 4] (direction 7: Down-Right)
    n0 = COORD_TO_NODE[(0, 2)]
    n4 = COORD_TO_NODE[(1, 3)]
    assert STEP_TARGET[n0, 7] == n4

    # Cut-out corner must NOT have diagonal connection:
    # (1,2) [node 3] to (2,1) [node 7] (direction 6: Down-Left)
    n1_2 = COORD_TO_NODE[(1, 2)]
    assert STEP_TARGET[n1_2, 6] == -1  # Invalid because (1,1) is not on board

    # (1,4) [node 5] to (2,5) [node 11] (direction 7: Down-Right)
    n1_4 = COORD_TO_NODE[(1, 4)]
    assert STEP_TARGET[n1_4, 7] == -1  # Invalid because (1,5) is not on board


def test_hen_initial_moves():
    foxes = set(START_FOX_NODES)
    hens = set(START_HEN_NODES)

    legal_actions = get_legal_actions(foxes, hens, "hens")
    assert len(legal_actions) > 0

    # Ensure all hen moves use only allowed forward/sideways directions and no jumps
    for act in legal_actions:
        origin, direction = decode_action(act)
        assert origin in hens
        assert direction in HEN_ALLOWED_DIRECTIONS
        assert direction < 8  # No jumps for hens


def test_fox_jump_capture_orthogonal():
    # Fox at (2,3) [node 9], Hen at (3,3) [node 16], (4,3) [node 23] empty
    foxes = {COORD_TO_NODE[(2, 3)]}
    hens = {COORD_TO_NODE[(3, 3)]}

    # Jump-Down is direction 9 (step dir 1: Down, + 8 = 9)
    legal = get_legal_actions(foxes, hens, "foxes")
    jump_action = encode_action(COORD_TO_NODE[(2, 3)], 9)
    assert jump_action in legal

    new_foxes, new_hens, is_capture, captured_node = apply_action(
        foxes, hens, jump_action, "foxes"
    )
    assert is_capture is True
    assert captured_node == COORD_TO_NODE[(3, 3)]
    assert new_foxes == {COORD_TO_NODE[(4, 3)]}
    assert len(new_hens) == 0


def test_fox_jump_capture_diagonal():
    # Fox at (1,2), Hen at (2,3), Landing at (3,4) empty
    foxes = {COORD_TO_NODE[(1, 2)]}
    hens = {COORD_TO_NODE[(2, 3)]}

    # Direction Down-Right is 7, Jump-Down-Right is 15 (7 + 8)
    legal = get_legal_actions(foxes, hens, "foxes")
    jump_action = encode_action(COORD_TO_NODE[(1, 2)], 15)
    assert jump_action in legal

    new_foxes, new_hens, is_capture, captured_node = apply_action(
        foxes, hens, jump_action, "foxes"
    )
    assert is_capture is True
    assert captured_node == COORD_TO_NODE[(2, 3)]
    assert new_foxes == {COORD_TO_NODE[(3, 4)]}
    assert len(new_hens) == 0


def test_terminal_stall_filled():
    # All 9 stall nodes occupied by hens
    foxes = {COORD_TO_NODE[(3, 0)], COORD_TO_NODE[(3, 6)]}
    hens = set(STALL_NODES)
    is_done, winner, reason = check_game_over(foxes, hens, 20)
    assert is_done is True
    assert winner == 1
    assert reason == "stall_filled"


def test_terminal_foxes_trapped():
    # Fox at (0,2), surrounding nodes (0,3) and (1,2) and (1,3) blocked by hens
    foxes = {COORD_TO_NODE[(0, 2)]}
    hens = {
        COORD_TO_NODE[(0, 3)],
        COORD_TO_NODE[(1, 2)],
        COORD_TO_NODE[(1, 3)],
        # block jump landings
        COORD_TO_NODE[(0, 4)],
        COORD_TO_NODE[(2, 2)],
        COORD_TO_NODE[(2, 4)],
    }
    is_done, winner, reason = check_game_over(foxes, hens, 30)
    assert is_done is True
    assert winner == 1
    assert reason == "foxes_trapped"


def test_terminal_hens_depleted():
    foxes = set(START_FOX_NODES)
    hens = {13, 14, 15, 16, 17, 18, 19, 20}  # exactly 8 hens
    is_done, winner, reason = check_game_over(foxes, hens, 40)
    assert is_done is True
    assert winner == -1
    assert reason == "hens_depleted"


def test_terminal_hens_trapped():
    # 10 Hens (len > 8) completely blocked in upper rows with no forward/sideways moves
    foxes = {COORD_TO_NODE[(2, 2)], COORD_TO_NODE[(2, 4)]}
    hens = {
        COORD_TO_NODE[(0, 2)], COORD_TO_NODE[(0, 3)], COORD_TO_NODE[(0, 4)],
        COORD_TO_NODE[(1, 2)], COORD_TO_NODE[(1, 3)], COORD_TO_NODE[(1, 4)],
        COORD_TO_NODE[(2, 0)], COORD_TO_NODE[(2, 1)],
        COORD_TO_NODE[(2, 5)], COORD_TO_NODE[(2, 6)],
    }
    assert len(hens) == 10  # More than 8 hens!
    is_done, winner, reason = check_game_over(foxes, hens, 50)
    assert is_done is True
    assert winner == -1
    assert reason == "hens_trapped"


def test_observation_tensor():
    foxes = set(START_FOX_NODES)
    hens = set(START_HEN_NODES)
    obs = get_observation_tensor(foxes, hens)
    assert obs.shape == (4, 7, 7)
    assert obs[0].sum() == 2.0   # 2 foxes
    assert obs[1].sum() == 20.0  # 20 hens
    assert obs[2].sum() == 33.0  # 33 valid board nodes
    assert obs[3].sum() == 9.0   # 9 stall nodes


def test_action_mask():
    foxes = set(START_FOX_NODES)
    hens = set(START_HEN_NODES)
    mask = get_action_mask(foxes, hens, "hens")
    assert mask.shape == (528,)
    assert mask.dtype == np.float32
    assert mask.sum() > 0
    # Every 1 in mask corresponds to a legal action
    legal = get_legal_actions(foxes, hens, "hens")
    assert mask.sum() == len(legal)


def test_3_leerzuege_rule():
    """Verify that moving back and forth along the same track is limited to max 3 Leerzüge (3 moves)."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from spectator import SpectatorApp

    spec = SpectatorApp(max_turns=300)
    spec.reset_game()

    # Node (2,0) and Node (2,1)
    n_2_0 = COORD_TO_NODE[(2, 0)]
    n_2_1 = COORD_TO_NODE[(2, 1)]

    # Initially 0 Leerzüge
    assert spec.consecutive_leerzuege == 0

    # 1st move: 2,0 -> 2,1 (Leerzug 1/3)
    spec.record_hen_move(n_2_0, n_2_1)
    assert spec.consecutive_leerzuege == 1

    # 2nd move: 2,1 -> 2,0 (Leerzug 2/3)
    spec.record_hen_move(n_2_1, n_2_0)
    assert spec.consecutive_leerzuege == 2

    # 3rd move: 2,0 -> 2,1 (Leerzug 3/3)
    spec.record_hen_move(n_2_0, n_2_1)
    assert spec.consecutive_leerzuege == 3

    # Now at Leerzug limit! Moving back along the track (2,1 -> 2,0) must be FORBIDDEN by filter_hen_legal_actions!
    # Direction for 2,1 -> 2,0 is Left (dir 2)
    action_return = encode_action(n_2_1, 2)
    # Dummy alternative action so fallback doesn't trigger
    n_dummy = COORD_TO_NODE[(3, 2)]
    action_other = encode_action(n_dummy, 0)
    filtered = spec.filter_hen_legal_actions([action_return, action_other])
    assert action_return not in filtered, "4th move along the same track MUST be prohibited!"
    assert action_other in filtered

    # Also test wing nodes (3,0) and (3,1) to ensure lateral moves on row 3 are tracked identically
    spec.reset_game()
    n_3_0 = COORD_TO_NODE[(3, 0)]
    n_3_1 = COORD_TO_NODE[(3, 1)]
    spec.record_hen_move(n_3_1, n_3_0)
    spec.record_hen_move(n_3_0, n_3_1)
    spec.record_hen_move(n_3_1, n_3_0)
    assert spec.consecutive_leerzuege == 3
    action_return_wing = encode_action(n_3_0, 3) # 3,0 -> 3,1 (Right)
    filtered_wing = spec.filter_hen_legal_actions([action_return_wing, action_other])
    assert action_return_wing not in filtered_wing, "Wing 4th move along same track MUST be prohibited!"

    # Forward move: e.g. from row 3 to row 2 resets the Leerzug counter
    n_3_2 = COORD_TO_NODE[(3, 2)]
    n_2_2 = COORD_TO_NODE[(2, 2)]
    spec.record_hen_move(n_3_2, n_2_2)
    assert spec.consecutive_leerzuege == 0
    assert spec.track_min is None
    assert spec.track_max is None


def test_hen_trapped_by_leerzug_limit_in_env():
    """Verify that when hens have no moves remaining due to 3 Leerzüge, env terminates safely."""
    from fuchs_und_henne.env import env
    from fuchs_und_henne.board import COORD_TO_NODE, encode_action

    e = env(max_turns=100)
    e.reset()

    # Position hens such that the only raw moves are between 2,0 and 2,1
    n_2_0 = COORD_TO_NODE[(2, 0)]
    n_2_1 = COORD_TO_NODE[(2, 1)]
    # Place 9 hens on board (valid count)
    e.hens = {n_2_0, COORD_TO_NODE[(0, 2)], COORD_TO_NODE[(0, 3)], COORD_TO_NODE[(0, 4)],
              COORD_TO_NODE[(1, 2)], COORD_TO_NODE[(1, 3)], COORD_TO_NODE[(1, 4)],
              COORD_TO_NODE[(2, 2)], COORD_TO_NODE[(2, 3)]}
    # Block nodes around stall
    e.foxes = {COORD_TO_NODE[(3, 3)], COORD_TO_NODE[(3, 4)]}

    # Simulate 3 consecutive Leerzug moves on track (2,0 <-> 2,1)
    e.consecutive_leerzuege = 3
    e.leerzug_half_moves = 3
    e.track_min = min(n_2_0, n_2_1)
    e.track_max = max(n_2_0, n_2_1)

    # Ensure stepping with an action with direction >= 8 does NOT raise IndexError
    invalid_jump_action = encode_action(n_2_0, 8)
    e.step(invalid_jump_action)


