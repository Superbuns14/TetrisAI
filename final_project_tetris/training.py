# training.py
import os
import random
import numpy as np

from config import (
    NUM_BAGS,
    ML_BASE_WEIGHTS, ML_LEARNED_PATH,
    H1_WEIGHTS,
    TRAIN_GAMMA_DEFAULT,
    RL_EPISODES_DEFAULT, RL_MAX_STEPS_DEFAULT,
    RL_GAMMA_DEFAULT, RL_ALPHA_DEFAULT, RL_EPSILON_DEFAULT,
)
from tetris_core import (
    TetrisGame,
    generate_bag_sequence,
    extract_features,
    enumerate_root_moves_fast_training,
)
from agents import tspin_heuristic_bonus, b2b_break_penalty

from config import ATTACK_WEIGHT, TETRIS_BONUS

# ----- Teacher-based training (supervised) -----

def _score_move_h1_train(mv):
    feat = extract_features(mv["board_before"], mv["board_after"])
    tetris_bonus = TETRIS_BONUS if mv["lines"] >= 4 else 0.0
    t_spin_bonus = tspin_heuristic_bonus(mv)
    b2b_pen = b2b_break_penalty(mv)
    return float(
        np.dot(H1_WEIGHTS, feat)
        + ATTACK_WEIGHT * mv["attack"]
        + tetris_bonus
        + t_spin_bonus
        - b2b_pen
    )

def _play_one_training_game(max_moves: int):
    seq = generate_bag_sequence(NUM_BAGS)
    game = TetrisGame(seq)

    feats = []
    rewards = []
    moves_done = 0

    while not game.game_over and moves_done < max_moves:
        moves = enumerate_root_moves_fast_training(game)
        if not moves:
            game.game_over = True
            break

        mv = max(moves, key=_score_move_h1_train)

        piece = game.current_piece
        piece.rotation = mv["rot"]
        piece.x = mv["x"]
        piece.y = 0
        if not game.valid_position(piece):
            game.game_over = True
            break

        board_before = [row[:] for row in game.board]
        attack_before = game.total_attack

        game.hard_drop_current()

        board_after = [row[:] for row in game.board]
        attack_gained = game.total_attack - attack_before

        feat = extract_features(board_before, board_after)
        feats.append(feat)
        rewards.append(attack_gained)

        moves_done += 1

    return feats, rewards

def train_ml_from_selfplay(
    num_games: int = 20,
    max_moves: int = 150,
    gamma: float = TRAIN_GAMMA_DEFAULT,
):
    """Supervised + return-based regression from teacher (Heuristic2-like) moves."""
    all_features = []
    all_targets = []

    for g in range(num_games):
        feats, rewards = _play_one_training_game(max_moves)
        if not feats:
            continue

        G = 0.0
        for f, r in reversed(list(zip(feats, rewards))):
            G = r + gamma * G
            all_features.append(f)
            all_targets.append(G)

        print(f"[TEACHER] Game {g+1}/{num_games}: {len(feats)} moves, final return {G:.2f}")

    if not all_features:
        print("No teacher data collected; aborting.")
        return

    X = np.stack(all_features)
    y = np.array(all_targets)
    d = X.shape[1]

    if os.path.exists(ML_LEARNED_PATH):
        w0 = np.load(ML_LEARNED_PATH).astype(np.float32)
        print("Loaded prior learned weights:", w0)
        if w0.shape[0] != d:
            print("Prior dimension mismatch; falling back to ML_BASE_WEIGHTS.")
            w0 = ML_BASE_WEIGHTS.copy()
    else:
        print("No learned weights found; using ML_BASE_WEIGHTS as prior.")
        w0 = ML_BASE_WEIGHTS.copy()

    lambda_prior = 1e-2
    A = X.T @ X + lambda_prior * np.eye(d)
    b = X.T @ y + lambda_prior * w0

    w_new = np.linalg.solve(A, b).astype(np.float32)

    blend = 0.7
    w_final = blend * w0 + (1.0 - blend) * w_new

    np.save(ML_LEARNED_PATH, w_final)
    print("[TEACHER] Training complete.")
    print("Old prior:", w0)
    print("New fit  :", w_new)
    print("Final    :", w_final)
    print(f"Saved to {ML_LEARNED_PATH}")


# ----- RL training -----


# safety constants
DELTA_CLIP = 10.0       # max magnitude of TD error
W_NORM_CLIP = 50.0      # max L2 norm of weight vector


def _safe_value(w, f):
    """Compute w·f with basic overflow protection."""
    v = float(np.dot(w, f))
    if not np.isfinite(v):
        return 0.0
    return v


def _choose_move_epsilon_greedy(game, w, epsilon: float):
    moves = enumerate_root_moves_fast_training(game)
    if not moves:
        return None, None

    feats = []
    values = []
    for mv in moves:
        f = extract_features(mv["board_before"], mv["board_after"])
        feats.append(f)
        values.append(_safe_value(w, f))

    # ε-greedy
    if random.random() < epsilon:
        idx = random.randrange(len(moves))
    else:
        idx = max(range(len(moves)), key=lambda i: values[i])

    return moves[idx], feats[idx]


def _estimate_state_value(game, w):
    moves = enumerate_root_moves_fast_training(game)
    if not moves:
        return 0.0
    best_val = -1e9
    for mv in moves:
        f = extract_features(mv["board_before"], mv["board_after"])
        val = _safe_value(w, f)
        if val > best_val:
            best_val = val
    return best_val


def train_ml_rl(
    episodes: int = RL_EPISODES_DEFAULT,
    max_steps: int = RL_MAX_STEPS_DEFAULT,
    gamma: float = RL_GAMMA_DEFAULT,
    alpha: float = RL_ALPHA_DEFAULT,
    epsilon: float = RL_EPSILON_DEFAULT,
):
    """Simple value-function RL training with bootstrapping, with stability safeguards."""
    # Start from learned weights if available, else base
    if os.path.exists(ML_LEARNED_PATH):
        w = np.load(ML_LEARNED_PATH).astype(np.float32)
        if w.shape != ML_BASE_WEIGHTS.shape:
            print("Learned weights wrong shape; resetting to base.")
            w = ML_BASE_WEIGHTS.copy()
        else:
            print("Loaded existing learned weights for RL training:", w)
    else:
        print("No learned weights found; starting RL from ML_BASE_WEIGHTS.")
        w = ML_BASE_WEIGHTS.copy()

    # Make sure we're float32 or float64
    w = w.astype(np.float32)

    for ep in range(episodes):
        seq = generate_bag_sequence(NUM_BAGS)
        game = TetrisGame(seq)

        steps = 0
        total_attack = 0.0

        while not game.game_over and steps < max_steps:
            mv, feat = _choose_move_epsilon_greedy(game, w, epsilon)
            if mv is None:
                game.game_over = True
                break

            # play move
            piece = game.current_piece
            piece.rotation = mv["rot"]
            piece.x = mv["x"]
            piece.y = 0
            if not game.valid_position(piece):
                game.game_over = True
                break

            attack_before = game.total_attack
            game.hard_drop_current()
            attack_gained = game.total_attack - attack_before
            r = attack_gained
            total_attack += r

            # current value
            V = _safe_value(w, feat)

            # next state's value
            if game.game_over or steps + 1 >= max_steps:
                V_next = 0.0
            else:
                V_next = _estimate_state_value(game, w)

            # TD error with clipping
            delta = r + gamma * V_next - V
            if not np.isfinite(delta):
                delta = 0.0
            if delta > DELTA_CLIP:
                delta = DELTA_CLIP
            elif delta < -DELTA_CLIP:
                delta = -DELTA_CLIP

            # Update weights
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                # skip pathological features
                pass
            else:
                w += alpha * delta * feat

            # Weight norm clipping
            norm = np.linalg.norm(w)
            if not np.isfinite(norm) or norm > W_NORM_CLIP:
                if not np.isfinite(norm):
                    print(f"[RL] Warning: non-finite weight norm, resetting to base.")
                    w = ML_BASE_WEIGHTS.copy().astype(np.float32)
                else:
                    w = (w / norm) * W_NORM_CLIP

            steps += 1

        print(f"[RL] Episode {ep+1}/{episodes}: steps={steps}, total_attack={total_attack:.1f}")

    np.save(ML_LEARNED_PATH, w.astype(np.float32))
    print("[RL] Training complete. Final learned weights:", w)
    print(f"Saved to {ML_LEARNED_PATH}")
