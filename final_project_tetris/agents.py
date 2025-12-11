# agents.py
import os
import random
import numpy as np

from config import (
    ATTACK_WEIGHT, TETRIS_BONUS, B2B_BREAK_PENALTY,
    ML_BASE_WEIGHTS, ML_LEARNED_PATH,
    H1_WEIGHTS, H2_WEIGHTS, H2_GAMMA, ML_LEARNED_GAMMA,
)
from tetris_core import (
    extract_features,
    enumerate_root_moves,
    enumerate_child_moves_no_hold,
)

# ----- Utility bonuses for heuristics -----

def tspin_heuristic_bonus(mv) -> float:
    t = mv.get("t_spin_type", 0)
    if t == 1:
        return 4.0
    elif t == 2:
        return 8.0
    elif t >= 3:
        return 12.0
    return 0.0

def b2b_break_penalty(mv) -> float:
    return B2B_BREAK_PENALTY if mv.get("breaks_b2b", False) else 0.0

# ----- Agents -----

class RandomAgent:
    def select_move(self, game):
        moves = enumerate_root_moves(game)
        if not moves:
            return None
        return random.choice(moves)

class Heuristic1Agent:
    """Single-ply heuristic agent."""
    def __init__(self):
        self.weights = H1_WEIGHTS.astype(np.float32)

    def select_move(self, game):
        moves = enumerate_root_moves(game)
        if not moves:
            return None

        best_score = -1e9
        best_move = None

        for mv in moves:
            feat = extract_features(mv["board_before"], mv["board_after"])
            tetris_bonus = TETRIS_BONUS if mv["lines"] >= 4 else 0.0
            t_spin_bonus = tspin_heuristic_bonus(mv)
            b2b_pen = b2b_break_penalty(mv)

            score = float(
                np.dot(self.weights, feat)
                + ATTACK_WEIGHT * mv["attack"]
                + tetris_bonus
                + t_spin_bonus
                - b2b_pen
            )
            if score > best_score:
                best_score = score
                best_move = mv
        return best_move

class Heuristic2Agent:
    """Two-ply heuristic agent with discount."""
    def __init__(self):
        self.weights = H2_WEIGHTS.astype(np.float32)
        self.gamma = H2_GAMMA

    def select_move(self, game):
        first_moves = enumerate_root_moves(game)
        if not first_moves:
            return None

        best_score = -1e9
        best_move = None

        for mv in first_moves:
            feat1 = extract_features(mv["board_before"], mv["board_after"])
            tetris_bonus1 = TETRIS_BONUS if mv["lines"] >= 4 else 0.0
            t_spin_bonus1 = tspin_heuristic_bonus(mv)
            b2b_pen1 = b2b_break_penalty(mv)
            score1 = float(
                np.dot(self.weights, feat1)
                + ATTACK_WEIGHT * mv["attack"]
                + tetris_bonus1
                + t_spin_bonus1
                - b2b_pen1
            )

            sim1 = mv["sim"]
            second_moves = enumerate_child_moves_no_hold(sim1)
            if second_moves:
                best2 = -1e9
                for mv2 in second_moves:
                    feat2 = extract_features(mv2["board_before"], mv2["board_after"])
                    tetris_bonus2 = TETRIS_BONUS if mv2["lines"] >= 4 else 0.0
                    t_spin_bonus2 = tspin_heuristic_bonus(mv2)
                    b2b_pen2 = b2b_break_penalty(mv2)
                    score2 = float(
                        np.dot(self.weights, feat2)
                        + ATTACK_WEIGHT * mv2["attack"]
                        + tetris_bonus2
                        + t_spin_bonus2
                        - b2b_pen2
                    )
                    if score2 > best2:
                        best2 = score2
                total = score1 + self.gamma * best2
            else:
                total = score1

            if total > best_score:
                best_score = total
                best_move = mv

        return best_move

class MLBaseAgent:
    """Single-ply linear ML agent with fixed weights."""
    def __init__(self):
        self.weights = ML_BASE_WEIGHTS.astype(np.float32)

    def select_move(self, game):
        moves = enumerate_root_moves(game)
        if not moves:
            return None
        best_score = -1e9
        best_move = None
        for mv in moves:
            feat = extract_features(mv["board_before"], mv["board_after"])
            score = float(self.weights @ feat)
            if score > best_score:
                best_score = score
                best_move = mv
        return best_move

class MLTrainAgent:
    """Two-ply ML agent that uses learned weights if available."""
    def __init__(self):
        if os.path.exists(ML_LEARNED_PATH):
            w = np.load(ML_LEARNED_PATH).astype(np.float32)
            if w.shape != ML_BASE_WEIGHTS.shape:
                print("Learned weights wrong shape; using base weights.")
                w = ML_BASE_WEIGHTS.copy()
            self.weights = w
            print("Loaded learned weights:", self.weights)
        else:
            print("No learned weights file; using base weights for ml_learned.")
            self.weights = ML_BASE_WEIGHTS.copy()
        self.gamma = ML_LEARNED_GAMMA

    def select_move(self, game):
        first_moves = enumerate_root_moves(game)
        if not first_moves:
            return None

        best_score = -1e9
        best_move = None

        for mv in first_moves:
            feat1 = extract_features(mv["board_before"], mv["board_after"])
            v1 = float(self.weights @ feat1)

            sim1 = mv["sim"]
            second_moves = enumerate_child_moves_no_hold(sim1)
            if second_moves:
                best_child = -1e9
                for mv2 in second_moves:
                    feat2 = extract_features(mv2["board_before"], mv2["board_after"])
                    v2 = float(self.weights @ feat2)
                    if v2 > best_child:
                        best_child = v2
                total = v1 + self.gamma * best_child
            else:
                total = v1

            if total > best_score:
                best_score = total
                best_move = mv

        return best_move


def make_agent(kind: str):
    kind = kind.lower()
    if kind == "random":
        return RandomAgent()
    if kind == "heuristic1":
        return Heuristic1Agent()
    if kind == "heuristic2":
        return Heuristic2Agent()
    if kind == "ml_base":
        return MLBaseAgent()
    if kind == "ml_learned":
        return MLTrainAgent()
    raise ValueError(f"Unknown agent type: {kind}")
