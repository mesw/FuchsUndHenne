"""Interactive Pygame Spectator & Playtest Harness for Fuchs und Henne.

Features:
- Modes: AI vs AI, Human (Hens) vs AI (Foxes), Human (Foxes) vs AI (Hens).
- Full 33-point cross board with orthogonal and diagonal lattice rendering.
- Visual Stall highlight (top 9 nodes).
- AI Policy decision inspector and Critic win-probability bar.
- Interactive click-to-move piece controls.
- Speed control (1 to 60 FPS), Pause/Play, and Restart.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pygame
import torch

from fuchs_und_henne.board import (
    COORD_TO_NODE,
    GRID_SIZE,
    JUMP_LAND,
    JUMP_OVER,
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
from fuchs_und_henne.models import ActorCritic

# ---------------------------------------------------------------------------
# Visual & Layout Constants
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 920
WINDOW_HEIGHT = 680
BOARD_SIZE = 560
CELL_SIZE = 70
OFFSET_X = 60
OFFSET_Y = 60

# Modern Color Palette
COLOR_BG = (26, 29, 36)            # Dark slate background
COLOR_BOARD_BG = (35, 39, 49)      # Subtle board surface
COLOR_LINE = (70, 78, 96)          # Grid connection lines
COLOR_STALL_BG = (45, 52, 68)      # Highlighted Stall area
COLOR_STALL_BORDER = (100, 140, 210)
COLOR_NODE = (55, 62, 78)          # Empty node circle
COLOR_NODE_BORDER = (90, 100, 120)

COLOR_FOX = (235, 75, 75)          # Crimson red for Fox
COLOR_FOX_RING = (255, 120, 120)
COLOR_HEN = (245, 190, 65)         # Golden amber for Hens
COLOR_HEN_RING = (255, 220, 120)

COLOR_SELECT = (60, 220, 130)      # Neon green selection ring
COLOR_LEGAL_MOVE = (60, 220, 130)  # Move destination dots
COLOR_PANEL_BG = (32, 36, 46)      # Sidebar info panel
COLOR_TEXT_MAIN = (240, 243, 248)
COLOR_TEXT_MUTED = (145, 155, 175)
COLOR_ACCENT = (80, 160, 255)


def get_node_pixel(node_idx: int) -> Tuple[int, int]:
    """Calculate center pixel coordinates (x, y) for a given node index."""
    r, c = NODE_COORDS[node_idx]
    x = OFFSET_X + c * CELL_SIZE + CELL_SIZE // 2
    y = OFFSET_Y + r * CELL_SIZE + CELL_SIZE // 2
    return x, y


def find_latest_checkpoint(role: str) -> Optional[Path]:
    """Find the final model or most recent checkpoint file for a role."""
    final_name = "fox_final.pt" if role == "foxes" else "hens_final.pt"
    final_p = Path(final_name)
    if final_p.exists():
        return final_p

    p = Path(f"checkpoints/{role}")
    if not p.exists():
        return None
    pts = list(p.glob(f"{role}_step_*.pt"))
    if not pts:
        return None
    pts.sort(key=lambda x: x.stat().st_mtime)
    return pts[-1]


class SpectatorApp:
    def __init__(
        self,
        fox_path: Optional[str] = None,
        hen_path: Optional[str] = None,
        max_turns: Optional[int] = 300,
    ):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Fuchs und Henne - Spectator & Playtest")
        self.clock = pygame.time.Clock()
        self.max_turns = max_turns

        self.font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 22, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.fox_model = ActorCritic().to(self.device).eval()
        self.hen_model = ActorCritic().to(self.device).eval()
        self.fox_path = fox_path
        self.hen_path = hen_path
        self._load_models()

        # Game Modes: 0 = AI vs AI, 1 = Human as Hens, 2 = Human as Fox
        self.mode = 0
        self.mode_names = ["AI vs AI", "Human (Hens) vs AI (Fox)", "Human (Fox) vs AI (Hens)"]

        # Execution Speed & State
        self.fps_target = 3
        self.is_paused = False
        self.stochastic = True  # True = realistic varied play; False = greedy argmax
        self.tactical_guard = True  # True = Hen AI avoids 1-ply blunders into immediate Fox jumps
        self.selected_node: Optional[int] = None
        self.legal_destinations: Dict[int, int] = {}  # target_node -> action_id
        self.state_history: List[Tuple[frozenset, frozenset, str]] = []

        # Internal Board
        self.reset_game()

    def _load_models(self) -> None:
        # Load Fox
        fox_file = Path(self.fox_path) if self.fox_path else find_latest_checkpoint("foxes")
        if fox_file and fox_file.exists():
            print(f"[*] Loaded Fox model from: {fox_file}")
            self.fox_model.load_state_dict(torch.load(fox_file, map_location=self.device, weights_only=True))
            self.fox_loaded_name = fox_file.stem
        else:
            self.fox_loaded_name = "Initial / Random"

        # Load Hen
        hen_file = Path(self.hen_path) if self.hen_path else find_latest_checkpoint("hens")
        if hen_file and hen_file.exists():
            print(f"[*] Loaded Hen model from: {hen_file}")
            self.hen_model.load_state_dict(torch.load(hen_file, map_location=self.device, weights_only=True))
            self.hen_loaded_name = hen_file.stem
        else:
            self.hen_loaded_name = "Initial / Random"

    def reset_game(self) -> None:
        self.foxes: Set[int] = set(START_FOX_NODES)
        self.hens: Set[int] = set(START_HEN_NODES)
        self.current_agent = "hens"
        self.turn_count = 0
        self.active_jumping_fox: Optional[int] = None
        self.selected_node = None
        self.legal_destinations = {}
        self.state_history = []
        self.consecutive_leerzuege = 0
        self.leerzug_half_moves = 0
        self.track_min: Optional[int] = None
        self.track_max: Optional[int] = None
        self.board_history: List[Tuple[frozenset, frozenset, str]] = []
        self.leerzug_warning = ""
        self.game_over = False
        self.winner = 0
        self.termination_reason = ""
        self.critic_eval = 0.0

    def would_cause_repetition(self, action: int) -> bool:
        next_foxes, next_hens, _, _ = apply_action(self.foxes, self.hens, action, "hens")
        candidate_state = (frozenset(next_foxes), frozenset(next_hens), "foxes")
        return self.board_history.count(candidate_state) >= 2

    def filter_hen_legal_actions(self, actions: List[int]) -> List[int]:
        if self.current_agent != "hens":
            return actions

        filtered = []
        for act in actions:
            orig, d = decode_action(act)
            if d >= 8:
                continue
            dest = STEP_TARGET[orig, d]
            if dest == -1:
                continue

            # 1. 3-Leerzüge rule: after 3 moves on the same track, moving along that track is prohibited!
            if self.leerzug_half_moves >= 3 and self.track_min is not None:
                cur_min = min(orig, dest)
                cur_max = max(orig, dest)
                if cur_min == self.track_min and cur_max == self.track_max:
                    continue

            # 2. Anti-repetition: Prohibit any move that would cause a 3-fold board repetition!
            if self.would_cause_repetition(act):
                continue

            filtered.append(act)

        # Fallback if filter leaves no moves
        if not filtered and actions:
            return actions

        return filtered

    def record_hen_move(self, src: int, dst: int) -> None:
        r_src, _ = NODE_COORDS[src]
        r_dst, _ = NODE_COORDS[dst]

        if r_dst < r_src:
            # Genuine forward advancement towards stall: reset counter & repetition history!
            self.consecutive_leerzuege = 0
            self.leerzug_half_moves = 0
            self.track_min = None
            self.track_max = None
            self.board_history.clear()
            self.leerzug_warning = ""
        else:
            # Lateral / non-advancing move (Leerzug)
            cur_min = min(src, dst)
            cur_max = max(src, dst)

            if cur_min == self.track_min and cur_max == self.track_max:
                self.leerzug_half_moves += 1
            else:
                self.track_min = cur_min
                self.track_max = cur_max
                self.leerzug_half_moves = 1

            self.consecutive_leerzuege = min(3, self.leerzug_half_moves)

            if self.leerzug_half_moves >= 3:
                self.leerzug_warning = "3 Leerzüge erreicht! Anderer Zug ist Pflicht."
            elif self.leerzug_half_moves == 2:
                self.leerzug_warning = "Leerzug 2/3"
            else:
                self.leerzug_warning = "Leerzug 1/3"

        self.board_history.append((frozenset(self.foxes), frozenset(self.hens), self.current_agent))

    def step_ai(self) -> None:
        if self.game_over:
            return

        is_fox = self.current_agent == "foxes"
        model = self.fox_model if is_fox else self.hen_model

        obs_np = get_observation_tensor(self.foxes, self.hens)

        # If Fox is continuing a multi-jump sequence, only jumps from the active fox piece are legal
        if is_fox and self.active_jumping_fox is not None:
            mask_np = np.zeros(NUM_ACTIONS, dtype=np.float32)
            u = self.active_jumping_fox
            occupied = self.foxes | self.hens
            for d in range(8):
                mid = JUMP_OVER[u, d]
                land = JUMP_LAND[u, d]
                if mid != -1 and (mid in self.hens) and (land not in occupied):
                    mask_np[encode_action(u, d + 8)] = 1.0
        else:
            mask_np = get_action_mask(self.foxes, self.hens, self.current_agent)

        if mask_np.sum() == 0:
            is_done, winner, reason = check_game_over(self.foxes, self.hens, self.turn_count, self.max_turns)
            self.game_over = True
            self.winner = winner
            self.termination_reason = reason
            self.active_jumping_fox = None
            return

        # Enforce 3-Leerzüge rule for Hens
        if self.current_agent == "hens":
            raw_legal = np.where(mask_np == 1.0)[0].tolist()
            filtered_legal = self.filter_hen_legal_actions(raw_legal)
            if not filtered_legal:
                filtered_legal = raw_legal
            mask_np = np.zeros(NUM_ACTIONS, dtype=np.float32)
            mask_np[filtered_legal] = 1.0

            # Tactical Guard for Hens: Filter out blunder moves that leave immediate jumps for the Fox
            if self.tactical_guard:
                safe_actions = []
                for act in filtered_legal:
                    next_f, next_h, _, _ = apply_action(self.foxes, self.hens, act, "hens")
                    fox_moves = get_legal_actions(next_f, next_h, "foxes")
                    if not any(decode_action(fa)[1] >= 8 for fa in fox_moves):
                        safe_actions.append(act)
                if safe_actions:
                    guard_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    guard_mask[safe_actions] = 1.0
                    mask_np = guard_mask

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_t, _, _, value_t = model.get_action_and_value(
                obs_t, mask_t, deterministic=not self.stochastic
            )
            self.critic_eval = value_t.item()

        action = action_t.item()
        orig, direction = decode_action(action)
        was_hen = self.current_agent == "hens"
        was_fox_capture = is_fox and (direction >= 8)

        self.foxes, self.hens, _, _ = apply_action(self.foxes, self.hens, action, self.current_agent)

        if was_hen:
            if direction < 8:
                dest = STEP_TARGET[orig, direction]
                if dest != -1:
                    self.record_hen_move(orig, dest)
        else:
            if was_fox_capture:
                self.consecutive_leerzuege = 0
                self.leerzug_half_moves = 0
                self.track_min = None
                self.track_max = None
                self.board_history.clear()
                self.leerzug_warning = ""
            self.board_history.append((frozenset(self.foxes), frozenset(self.hens), self.current_agent))

        # Check multi-jump continuation for foxes
        has_continuation = False
        if is_fox:
            orig, direction = decode_action(action)
            if direction >= 8:
                step_dir = direction - 8
                land_pos = JUMP_LAND[orig, step_dir]
                occupied = self.foxes | self.hens
                for d in range(8):
                    mid = JUMP_OVER[land_pos, d]
                    land = JUMP_LAND[land_pos, d]
                    if mid != -1 and (mid in self.hens) and (land not in occupied):
                        has_continuation = True
                        break
                if has_continuation:
                    self.active_jumping_fox = land_pos
                else:
                    self.active_jumping_fox = None
            else:
                self.active_jumping_fox = None
        else:
            self.active_jumping_fox = None

        # Check game over
        is_done, winner, reason = check_game_over(self.foxes, self.hens, self.turn_count, self.max_turns)
        if is_done:
            self.game_over = True
            self.winner = winner
            self.termination_reason = reason
            self.active_jumping_fox = None
        elif not has_continuation:
            self.turn_count += 1
            self.current_agent = "foxes" if self.current_agent == "hens" else "hens"

    def handle_click(self, pos: Tuple[int, int]) -> None:
        if self.game_over:
            return

        # Determine if it's human turn
        human_turn = False
        if self.mode == 1 and self.current_agent == "hens":
            human_turn = True
        elif self.mode == 2 and self.current_agent == "foxes":
            human_turn = True

        if not human_turn:
            return

        # Find clicked node
        clicked_node: Optional[int] = None
        for idx in range(NUM_NODES):
            nx, ny = get_node_pixel(idx)
            dist = np.hypot(pos[0] - nx, pos[1] - ny)
            if dist <= CELL_SIZE * 0.45:
                clicked_node = idx
                break

        if clicked_node is None:
            # If in multi-jump, don't clear destinations on misclick
            if self.active_jumping_fox is None:
                self.selected_node = None
                self.legal_destinations = {}
            return

        # 1. Active Multi-Jump state for Human Fox
        if self.active_jumping_fox is not None:
            # Clicking the jumping fox piece again finishes the chain voluntarily
            if clicked_node == self.active_jumping_fox:
                self.active_jumping_fox = None
                self.selected_node = None
                self.legal_destinations = {}
                self.turn_count += 1
                self.current_agent = "hens"
                return

            # Clicking a valid chained jump destination executes the next jump!
            if clicked_node in self.legal_destinations:
                action = self.legal_destinations[clicked_node]
                orig, direction = decode_action(action)
                step_dir = direction - 8
                land_pos = JUMP_LAND[orig, step_dir]
                self.foxes, self.hens, _, _ = apply_action(self.foxes, self.hens, action, "foxes")
                self.consecutive_leerzuege = 0
                self.leerzug_half_moves = 0
                self.last_hen_src = None
                self.last_hen_dst = None
                self.board_history.clear()
                self.leerzug_warning = ""

                is_done, winner, reason = check_game_over(self.foxes, self.hens, self.turn_count, self.max_turns)
                if is_done:
                    self.game_over = True
                    self.winner = winner
                    self.termination_reason = reason
                    self.active_jumping_fox = None
                    self.selected_node = None
                    self.legal_destinations = {}
                    return

                # Check for further chained jumps
                continuation_actions = {}
                occupied = self.foxes | self.hens
                for d in range(8):
                    mid = JUMP_OVER[land_pos, d]
                    land = JUMP_LAND[land_pos, d]
                    if mid != -1 and (mid in self.hens) and (land not in occupied):
                        continuation_actions[land] = encode_action(land_pos, d + 8)

                if continuation_actions:
                    self.active_jumping_fox = land_pos
                    self.selected_node = land_pos
                    self.legal_destinations = continuation_actions
                else:
                    self.active_jumping_fox = None
                    self.selected_node = None
                    self.legal_destinations = {}
                    self.turn_count += 1
                    self.current_agent = "hens"
            return

        # 2. Normal move execution: If a piece was selected and clicked node is a legal destination -> Move!
        if self.selected_node is not None and clicked_node in self.legal_destinations:
            action = self.legal_destinations[clicked_node]
            orig, direction = decode_action(action)
            was_hen = self.current_agent == "hens"
            was_fox_capture = (self.current_agent == "foxes") and (direction >= 8)

            self.foxes, self.hens, _, _ = apply_action(self.foxes, self.hens, action, self.current_agent)

            if was_hen:
                self.record_hen_move(orig, clicked_node)
            else:
                if was_fox_capture:
                    self.consecutive_leerzuege = 0
                    self.leerzug_half_moves = 0
                    self.track_min = None
                    self.track_max = None
                    self.board_history.clear()
                    self.leerzug_warning = ""
                self.board_history.append((frozenset(self.foxes), frozenset(self.hens), self.current_agent))

            is_done, winner, reason = check_game_over(self.foxes, self.hens, self.turn_count, self.max_turns)
            if is_done:
                self.game_over = True
                self.winner = winner
                self.termination_reason = reason
                self.selected_node = None
                self.legal_destinations = {}
                self.active_jumping_fox = None
                return

            # Check if this was a Fox jump that has a multi-jump continuation
            has_continuation = False
            if self.current_agent == "foxes" and direction >= 8:
                step_dir = direction - 8
                land_pos = JUMP_LAND[orig, step_dir]
                continuation_actions = {}
                occupied = self.foxes | self.hens
                for d in range(8):
                    mid = JUMP_OVER[land_pos, d]
                    land = JUMP_LAND[land_pos, d]
                    if mid != -1 and (mid in self.hens) and (land not in occupied):
                        continuation_actions[land] = encode_action(land_pos, d + 8)

                if continuation_actions:
                    has_continuation = True
                    self.active_jumping_fox = land_pos
                    self.selected_node = land_pos
                    self.legal_destinations = continuation_actions

            if not has_continuation:
                self.selected_node = None
                self.legal_destinations = {}
                self.active_jumping_fox = None
                self.turn_count += 1
                self.current_agent = "foxes" if self.current_agent == "hens" else "hens"
            return

        # 3. Otherwise select own piece
        is_valid_piece = (self.current_agent == "hens" and clicked_node in self.hens) or (
            self.current_agent == "foxes" and clicked_node in self.foxes
        )

        if is_valid_piece:
            self.selected_node = clicked_node
            self.legal_destinations = {}
            legal_actions = get_legal_actions(self.foxes, self.hens, self.current_agent)
            if self.current_agent == "hens":
                legal_actions = self.filter_hen_legal_actions(legal_actions)
            for act in legal_actions:
                orig, d = decode_action(act)
                if orig == clicked_node:
                    if d < 8:
                        dest = STEP_TARGET[orig, d]
                    else:
                        step_d = d - 8
                        dest = JUMP_LAND[orig, step_d]
                    if dest != -1:
                        self.legal_destinations[dest] = act
        else:
            self.selected_node = None
            self.legal_destinations = {}

    def draw_board(self) -> None:
        # Background
        self.screen.fill(COLOR_BG)

        # Highlight Stall Area (Rows 0, 1, 2 and Cols 2, 3, 4)
        stall_x = OFFSET_X + 2 * CELL_SIZE
        stall_y = OFFSET_Y + 0 * CELL_SIZE
        stall_w = 3 * CELL_SIZE
        stall_h = 3 * CELL_SIZE
        pygame.draw.rect(self.screen, COLOR_STALL_BG, (stall_x, stall_y, stall_w, stall_h), border_radius=6)
        pygame.draw.rect(self.screen, COLOR_STALL_BORDER, (stall_x, stall_y, stall_w, stall_h), width=2, border_radius=6)

        # Connection Lines (Orthogonal & Diagonal Lattice)
        for u in range(NUM_NODES):
            ux, uy = get_node_pixel(u)
            for d in range(8):
                v = STEP_TARGET[u, d]
                if v != -1 and v > u:  # Draw each edge once
                    vx, vy = get_node_pixel(v)
                    pygame.draw.line(self.screen, COLOR_LINE, (ux, uy), (vx, vy), 2)

        # Draw Node Bases
        for u in range(NUM_NODES):
            ux, uy = get_node_pixel(u)
            pygame.draw.circle(self.screen, COLOR_NODE, (ux, uy), 8)
            pygame.draw.circle(self.screen, COLOR_NODE_BORDER, (ux, uy), 8, 1)

        # Legal Move Indicators (when a piece is selected by human)
        for dest, act in self.legal_destinations.items():
            dx, dy = get_node_pixel(dest)
            is_blunder = False
            if self.current_agent == "hens":
                next_f, next_h, _, _ = apply_action(self.foxes, self.hens, act, "hens")
                fox_moves = get_legal_actions(next_f, next_h, "foxes")
                is_blunder = any(decode_action(fa)[1] >= 8 for fa in fox_moves)

            col = (255, 140, 40) if is_blunder else COLOR_LEGAL_MOVE  # Orange warning if move allows jump
            pygame.draw.circle(self.screen, col, (dx, dy), 10, width=3)
            pygame.draw.circle(self.screen, col, (dx, dy), 4)

        # Draw Hens
        for h in self.hens:
            hx, hy = get_node_pixel(h)
            pygame.draw.circle(self.screen, COLOR_HEN, (hx, hy), 18)
            pygame.draw.circle(self.screen, COLOR_HEN_RING, (hx, hy), 18, 2)
            if self.selected_node == h:
                pygame.draw.circle(self.screen, COLOR_SELECT, (hx, hy), 22, 3)

        # Draw Foxes
        for f in self.foxes:
            fx, fy = get_node_pixel(f)
            pygame.draw.circle(self.screen, COLOR_FOX, (fx, fy), 20)
            pygame.draw.circle(self.screen, COLOR_FOX_RING, (fx, fy), 20, 3)
            if self.selected_node == f:
                pygame.draw.circle(self.screen, COLOR_SELECT, (fx, fy), 24, 3)
            if self.active_jumping_fox == f:
                # Prominent golden multi-jump indicator
                pygame.draw.circle(self.screen, (255, 225, 70), (fx, fy), 28, 3)

    def draw_sidebar(self) -> None:
        panel_x = 600
        panel_y = 20
        panel_w = 300
        panel_h = WINDOW_HEIGHT - 40

        pygame.draw.rect(self.screen, COLOR_PANEL_BG, (panel_x, panel_y, panel_w, panel_h), border_radius=12)

        # Title
        title_surf = self.font_title.render("Fuchs und Henne", True, COLOR_TEXT_MAIN)
        self.screen.blit(title_surf, (panel_x + 20, panel_y + 20))

        subtitle = self.font_small.render("Asymmetric Deep RL Spectator", True, COLOR_TEXT_MUTED)
        self.screen.blit(subtitle, (panel_x + 20, panel_y + 50))

        y = panel_y + 90

        # Mode Info
        mode_label = self.font_small.render("MODE (Press 'M' to switch):", True, COLOR_TEXT_MUTED)
        self.screen.blit(mode_label, (panel_x + 20, y))
        mode_val = self.font_body.render(self.mode_names[self.mode], True, COLOR_ACCENT)
        self.screen.blit(mode_val, (panel_x + 20, y + 20))

        y += 60

        # Game Stats
        limit_str = f" / {self.max_turns}" if self.max_turns else " (Unlimited)"
        turn_txt = self.font_body.render(f"Turn: {self.turn_count}{limit_str}", True, COLOR_TEXT_MAIN)
        self.screen.blit(turn_txt, (panel_x + 20, y))

        active_color = COLOR_FOX if self.current_agent == "foxes" else COLOR_HEN
        active_name = "FOX (MULTI-JUMP!)" if self.active_jumping_fox is not None else self.current_agent.upper()
        active_txt = self.font_body.render(f"Active: {active_name}", True, (255, 220, 80) if self.active_jumping_fox is not None else active_color)
        self.screen.blit(active_txt, (panel_x + 20, y + 25))

        if self.active_jumping_fox is not None:
            jump_alert = self.font_small.render("Multi-jump ready! Click jump dot,", True, (255, 225, 70))
            jump_sub = self.font_small.render("or click Fox to finish turn.", True, COLOR_TEXT_MUTED)
            self.screen.blit(jump_alert, (panel_x + 20, y + 48))
            self.screen.blit(jump_sub, (panel_x + 20, y + 66))
            hens_txt = self.font_body.render(f"Hens Remaining: {len(self.hens)} / 20", True, COLOR_TEXT_MAIN)
            self.screen.blit(hens_txt, (panel_x + 20, y + 90))
            y += 125
        else:
            hens_txt = self.font_body.render(f"Hens Remaining: {len(self.hens)} / 20", True, COLOR_TEXT_MAIN)
            self.screen.blit(hens_txt, (panel_x + 20, y + 50))
            y += 80

        if self.consecutive_leerzuege > 0:
            lz_color = (255, 120, 120) if self.consecutive_leerzuege >= 3 else (255, 220, 80)
            lz_txt = self.font_small.render(f"Leerzüge: {self.consecutive_leerzuege}/3 (hin & her)", True, lz_color)
            self.screen.blit(lz_txt, (panel_x + 20, y))
            y += 24

        # Critic Win-Probability Assessment (-1 Fox to +1 Hens)
        critic_lbl = self.font_small.render(f"Critic Win Evaluation: {self.critic_eval:+.2f}", True, COLOR_TEXT_MUTED)
        self.screen.blit(critic_lbl, (panel_x + 20, y))

        bar_x = panel_x + 20
        bar_y = y + 22
        bar_w = 260
        bar_h = 12
        pygame.draw.rect(self.screen, (50, 55, 68), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        # Center is 0.0, left is -1.0 (Fox), right is +1.0 (Hens)
        fill_pos = int((self.critic_eval + 1.0) / 2.0 * bar_w)
        fill_pos = max(0, min(bar_w, fill_pos))
        eval_color = COLOR_HEN if self.critic_eval > 0 else COLOR_FOX
        pygame.draw.rect(self.screen, eval_color, (bar_x, bar_y, fill_pos, bar_h), border_radius=6)

        y += 55

        # Checkpoints Loaded
        chk_title = self.font_small.render("Loaded Policies:", True, COLOR_TEXT_MUTED)
        self.screen.blit(chk_title, (panel_x + 20, y))
        f_info = self.font_small.render(f"Fox: {self.fox_loaded_name}", True, COLOR_FOX)
        h_info = self.font_small.render(f"Hen: {self.hen_loaded_name}", True, COLOR_HEN)
        self.screen.blit(f_info, (panel_x + 20, y + 20))
        self.screen.blit(h_info, (panel_x + 20, y + 38))

        y += 80

        # Controls & Hotkeys
        ctrl_title = self.font_small.render("Controls:", True, COLOR_TEXT_MUTED)
        self.screen.blit(ctrl_title, (panel_x + 20, y))
        c1 = self.font_small.render("SPACE : Pause / Resume", True, COLOR_TEXT_MAIN)
        c2 = self.font_small.render("R     : Reset / New Game", True, COLOR_TEXT_MAIN)
        c3 = self.font_small.render("L     : Reload Checkpoints", True, COLOR_TEXT_MAIN)
        c4 = self.font_small.render(f"T     : Style ({'Stochastic' if self.stochastic else 'Greedy'})", True, COLOR_ACCENT)
        c5 = self.font_small.render(f"G     : Guard ({'ON' if self.tactical_guard else 'OFF'})", True, COLOR_ACCENT if self.tactical_guard else COLOR_TEXT_MUTED)
        c6 = self.font_small.render(f"UP/DN : Speed ({self.fps_target} FPS)", True, COLOR_TEXT_MAIN)
        self.screen.blit(c1, (panel_x + 20, y + 20))
        self.screen.blit(c2, (panel_x + 20, y + 36))
        self.screen.blit(c3, (panel_x + 20, y + 52))
        self.screen.blit(c4, (panel_x + 20, y + 68))
        self.screen.blit(c5, (panel_x + 20, y + 84))
        self.screen.blit(c6, (panel_x + 20, y + 100))

        # Game Over Banner
        if self.game_over:
            y_banner = WINDOW_HEIGHT - 80
            pygame.draw.rect(self.screen, (20, 24, 32), (panel_x + 10, y_banner, 280, 56), border_radius=8)
            winner_str = "HENS WIN!" if self.winner == 1 else ("FOXES WIN!" if self.winner == -1 else "DRAW!")
            w_color = COLOR_HEN if self.winner == 1 else (COLOR_FOX if self.winner == -1 else COLOR_TEXT_MUTED)
            win_surf = self.font_title.render(winner_str, True, w_color)
            reason_surf = self.font_small.render(f"Reason: {self.termination_reason}", True, COLOR_TEXT_MUTED)
            self.screen.blit(win_surf, (panel_x + 20, y_banner + 6))
            self.screen.blit(reason_surf, (panel_x + 20, y_banner + 32))

    def run(self) -> None:
        running = True
        move_timer = 0

        while running:
            dt = self.clock.tick(60)
            move_timer += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_l:
                        self._load_models()
                    elif event.key == pygame.K_t:
                        self.stochastic = not self.stochastic
                    elif event.key == pygame.K_g:
                        self.tactical_guard = not self.tactical_guard
                    elif event.key == pygame.K_m:
                        self.mode = (self.mode + 1) % 3
                        self.selected_node = None
                        self.legal_destinations = {}
                        self.active_jumping_fox = None
                    elif event.key == pygame.K_UP:
                        self.fps_target = min(30, self.fps_target + 1)
                    elif event.key == pygame.K_DOWN:
                        self.fps_target = max(1, self.fps_target - 1)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            # Auto-play AI turns if not paused
            time_per_move_ms = 1000.0 / self.fps_target
            if not self.is_paused and not self.game_over and move_timer >= time_per_move_ms:
                is_ai_turn = False
                if self.mode == 0:
                    is_ai_turn = True
                elif self.mode == 1 and self.current_agent == "foxes":
                    is_ai_turn = True
                elif self.mode == 2 and self.current_agent == "hens":
                    is_ai_turn = True

                if is_ai_turn:
                    self.step_ai()
                    move_timer = 0

            # Render
            self.draw_board()
            self.draw_sidebar()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pygame Spectator for Fuchs und Henne")
    parser.add_argument("--fox", type=str, default=None, help="Path to Fox model .pt checkpoint")
    parser.add_argument("--hens", type=str, default=None, help="Path to Hen model .pt checkpoint")
    parser.add_argument("--max_turns", type=int, default=300, help="Max turns before draw (0 for unlimited)")
    args = parser.parse_args()

    max_turns = args.max_turns if args.max_turns > 0 else None
    app = SpectatorApp(fox_path=args.fox, hen_path=args.hens, max_turns=max_turns)
    app.run()
