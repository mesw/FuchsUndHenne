"""Board topology, constants, move generator, and rules for Fuchs und Henne.

Implements the traditional German 33-point cross board with full diagonal
lattice across every 2x2 square, 2 Foxes, 20 Hens, and stall-filling victory.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Board Geometry & Node Indices
# ---------------------------------------------------------------------------
GRID_SIZE = 7
NUM_NODES = 33
NUM_DIRECTIONS = 16  # 8 step directions + 8 jump directions
NUM_ACTIONS = NUM_NODES * NUM_DIRECTIONS  # 528 discrete actions

# Node (r, c) coordinates in row-major order
NODE_COORDS: Tuple[Tuple[int, int], ...] = (
    # Row 0 (Stall top)
    (0, 2), (0, 3), (0, 4),
    # Row 1 (Stall mid)
    (1, 2), (1, 3), (1, 4),
    # Row 2 (Cross top bar / Stall bottom)
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    # Row 3 (Cross center horizontal bar)
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    # Row 4 (Cross bottom horizontal bar)
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    # Row 5 (Bottom arm)
    (5, 2), (5, 3), (5, 4),
    # Row 6 (Bottom arm end)
    (6, 2), (6, 3), (6, 4),
)

COORD_TO_NODE: Dict[Tuple[int, int], int] = {
    coord: idx for idx, coord in enumerate(NODE_COORDS)
}

# 7x7 Static Board Mask (1.0 for valid 33 nodes, 0.0 for eliminated corners)
BOARD_MASK = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
for r, c in NODE_COORDS:
    BOARD_MASK[r, c] = 1.0

# 9 Stall Nodes ("Der Stall"): Rows 0, 1, 2 and Cols 2, 3, 4
STALL_COORDS: Set[Tuple[int, int]] = {
    (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4),
}
STALL_NODES: Tuple[int, ...] = tuple(sorted(COORD_TO_NODE[c] for c in STALL_COORDS))
# Indices: (0, 1, 2, 3, 4, 5, 8, 9, 10)

STALL_MASK = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
for r, c in STALL_COORDS:
    STALL_MASK[r, c] = 1.0

# Initial Piece Positions according to German rules
# 2 Foxes at Stall lower corners: (2,2) -> Node 8 and (2,4) -> Node 10
START_FOX_NODES: Tuple[int, ...] = (8, 10)

# 20 Hens on rows 3, 4, 5, 6 (Nodes 13 to 32)
START_HEN_NODES: Tuple[int, ...] = tuple(range(13, 33))

# ---------------------------------------------------------------------------
# Direction Definitions (16 directions)
# ---------------------------------------------------------------------------
# 0-7: 1-step moves
# 8-15: 2-step capture jumps
DIRECTION_DELTAS: Tuple[Tuple[int, int], ...] = (
    # Orthogonal Steps (0..3)
    (-1, 0),   # 0: Up
    (1, 0),    # 1: Down
    (0, -1),   # 2: Left
    (0, 1),    # 3: Right
    # Diagonal Steps (4..7)
    (-1, -1),  # 4: Up-Left
    (-1, 1),   # 5: Up-Right
    (1, -1),   # 6: Down-Left
    (1, 1),    # 7: Down-Right
    # Orthogonal Jumps (8..11)
    (-2, 0),   # 8: Jump-Up
    (2, 0),    # 9: Jump-Down
    (0, -2),   # 10: Jump-Left
    (0, 2),    # 11: Jump-Right
    # Diagonal Jumps (12..15)
    (-2, -2),  # 12: Jump-Up-Left
    (-2, 2),   # 13: Jump-Up-Right
    (2, -2),   # 14: Jump-Down-Left
    (2, 2),    # 15: Jump-Down-Right
)

# Hens can only move forward or sideways (Up, Left, Right, Up-Left, Up-Right)
HEN_ALLOWED_DIRECTIONS: Tuple[int, ...] = (0, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# Precomputed Step & Jump Target Lookup Tables
# ---------------------------------------------------------------------------
# STEP_TARGET[node, dir_id] -> target node index or -1
STEP_TARGET = np.full((NUM_NODES, 8), -1, dtype=np.int32)

# JUMP_OVER[node, dir_id_0_to_7] -> jumped node index or -1
JUMP_OVER = np.full((NUM_NODES, 8), -1, dtype=np.int32)

# JUMP_LAND[node, dir_id_0_to_7] -> landing node index or -1
JUMP_LAND = np.full((NUM_NODES, 8), -1, dtype=np.int32)


def _init_lookup_tables() -> None:
    for u, (r, c) in enumerate(NODE_COORDS):
        for d in range(8):
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = r + dr, c + dc
            target = COORD_TO_NODE.get((nr, nc), -1)

            if target != -1:
                if d < 4:
                    # Orthogonal: always valid if target exists on board
                    STEP_TARGET[u, d] = target
                else:
                    # Diagonal: valid if both intermediate corners also exist
                    # (i.e. all 4 vertices of the 1x1 cell belong to the cross board)
                    corner1 = (r, nc)
                    corner2 = (nr, c)
                    if corner1 in COORD_TO_NODE and corner2 in COORD_TO_NODE:
                        STEP_TARGET[u, d] = target

    # Precompute jumps using consecutive valid steps in the same direction
    for u in range(NUM_NODES):
        for d in range(8):
            mid = STEP_TARGET[u, d]
            if mid != -1:
                land = STEP_TARGET[mid, d]
                if land != -1:
                    JUMP_OVER[u, d] = mid
                    JUMP_LAND[u, d] = land


_init_lookup_tables()


# ---------------------------------------------------------------------------
# Action Encoding & Move Validation
# ---------------------------------------------------------------------------
def encode_action(origin_node: int, direction_id: int) -> int:
    """Encode (origin_node, direction_id) into a discrete action integer in [0, 527]."""
    return origin_node * NUM_DIRECTIONS + direction_id


def decode_action(action: int) -> Tuple[int, int]:
    """Decode discrete action integer into (origin_node, direction_id)."""
    return divmod(action, NUM_DIRECTIONS)


def get_legal_actions(
    foxes: Set[int],
    hens: Set[int],
    current_player: str,
    mandatory_jumps: bool = False,
) -> List[int]:
    """Return list of legal action IDs for current_player ('foxes' or 'hens').

    Args:
        foxes: Set of node indices occupied by foxes.
        hens: Set of node indices occupied by hens.
        current_player: 'foxes' or 'hens'.
        mandatory_jumps: If True and current_player == 'foxes', if any capture jump
            exists, only jump actions are legal. Defaults to False for pure RL policy exploration.
    """
    occupied = foxes | hens
    legal_actions: List[int] = []

    if current_player == "hens":
        for u in hens:
            for d in HEN_ALLOWED_DIRECTIONS:
                dst = STEP_TARGET[u, d]
                if dst != -1 and dst not in occupied:
                    legal_actions.append(encode_action(u, d))

    elif current_player == "foxes":
        step_actions: List[int] = []
        jump_actions: List[int] = []

        for u in foxes:
            for d in range(8):
                # 1. Check Step
                dst = STEP_TARGET[u, d]
                if dst != -1 and dst not in occupied:
                    step_actions.append(encode_action(u, d))

                # 2. Check Jump
                mid = JUMP_OVER[u, d]
                land = JUMP_LAND[u, d]
                if mid != -1 and (mid in hens) and (land not in occupied):
                    jump_actions.append(encode_action(u, d + 8))

        if mandatory_jumps and jump_actions:
            return jump_actions
        return jump_actions + step_actions

    return legal_actions


def get_action_mask(
    foxes: Set[int],
    hens: Set[int],
    current_player: str,
    mandatory_jumps: bool = False,
) -> np.ndarray:
    """Return binary mask M in {0.0, 1.0}^528 indicating legal actions."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for act in get_legal_actions(foxes, hens, current_player, mandatory_jumps):
        mask[act] = 1.0
    return mask


def apply_action(
    foxes: Set[int],
    hens: Set[int],
    action: int,
    current_player: str,
) -> Tuple[Set[int], Set[int], bool, int]:
    """Apply action and return updated state.

    Returns:
        (new_foxes, new_hens, is_capture, captured_node)
    """
    origin, direction = decode_action(action)
    new_foxes = set(foxes)
    new_hens = set(hens)

    if current_player == "hens":
        if direction >= 8:
            # Hens cannot jump; safety guard against invalid action indices
            return new_foxes, new_hens, False, -1
        dst = STEP_TARGET[origin, direction]
        if dst != -1 and origin in new_hens:
            new_hens.remove(origin)
            new_hens.add(dst)
        return new_foxes, new_hens, False, -1

    # current_player == "foxes"
    if direction < 8:
        # Step move
        dst = STEP_TARGET[origin, direction]
        if dst != -1 and origin in new_foxes:
            new_foxes.remove(origin)
            new_foxes.add(dst)
        return new_foxes, new_hens, False, -1
    else:
        # Jump capture
        step_dir = direction - 8
        mid = JUMP_OVER[origin, step_dir]
        land = JUMP_LAND[origin, step_dir]
        if mid != -1 and land != -1 and origin in new_foxes:
            new_foxes.remove(origin)
            new_foxes.add(land)
            if mid in new_hens:
                new_hens.remove(mid)
            return new_foxes, new_hens, True, mid
        return new_foxes, new_hens, False, -1


def check_game_over(
    foxes: Set[int],
    hens: Set[int],
    turn_count: int,
    max_turns: Optional[int] = 250,
) -> Tuple[bool, int, str]:
    """Check terminal state according to traditional German rules.

    Returns:
        (is_done, winner, reason)
        winner: +1 for Hens win, -1 for Foxes win, 0 for Draw/In-progress
    """
    # 1. Hen Victory: All 9 Stall nodes occupied by Hens
    if all(s in hens for s in STALL_NODES):
        return True, 1, "stall_filled"

    # 2. Hen Victory: Both Foxes have 0 legal moves (trapped)
    fox_legal = get_legal_actions(foxes, hens, "foxes")
    if len(fox_legal) == 0:
        return True, 1, "foxes_trapped"

    # 3. Fox Victory: Hens have 0 legal moves (immobilized / blocked)
    hen_legal = get_legal_actions(foxes, hens, "hens")
    if len(hen_legal) == 0:
        return True, -1, "hens_trapped"

    # 4. Fox Victory: Hens <= 8 (cannot fill the 9-node Stall)
    if len(hens) <= 8:
        return True, -1, "hens_depleted"

    # 5. Truncation / Draw (if max_turns is specified)
    if max_turns is not None and turn_count >= max_turns:
        return True, 0, "max_turns_reached"

    return False, 0, ""


def get_observation_tensor(foxes: Set[int], hens: Set[int]) -> np.ndarray:
    """Generate state tensor S in R^{4 x 7 x 7} for neural network input.

    Channels:
        0: Fox binary mask
        1: Hen binary mask
        2: Valid board mask
        3: Stall mask
    """
    obs = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for f in foxes:
        r, c = NODE_COORDS[f]
        obs[0, r, c] = 1.0

    for h in hens:
        r, c = NODE_COORDS[h]
        obs[1, r, c] = 1.0

    obs[2] = BOARD_MASK
    obs[3] = STALL_MASK

    return obs
