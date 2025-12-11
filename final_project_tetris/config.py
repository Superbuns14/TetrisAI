# config.py
import numpy as np

# Board / UI configuration
ROWS = 20
COLS = 10
CELL_SIZE = 40

BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE

PANEL_WIDTH = 160
MARGIN = 20
FPS = 60

# Default pieces-per-second throttle
TARGET_PPS_DEFAULT = 2.0  # can be overridden from CLI
NEXT_PREVIEW_COUNT = 5
NUM_BAGS = 1000

# Random seed (for deterministic sequences when desired)
RNG_SEED = 12345

# Attack / heuristic constants
ATTACK_WEIGHT = 20.0
TETRIS_BONUS = 10.0

T_SPIN_SINGLE_ATTACK = 2
T_SPIN_DOUBLE_ATTACK = 4
T_SPIN_TRIPLE_ATTACK = 6
PERFECT_CLEAR_ATTACK_BONUS = 10

T_SPIN_SINGLE_BONUS = 4.0
T_SPIN_DOUBLE_BONUS = 8.0
T_SPIN_TRIPLE_BONUS = 12.0

B2B_BREAK_PENALTY = 8.0

# Base ML weights (5-dim feature vector)
ML_BASE_WEIGHTS = np.array([-0.3, -3.5, -0.2, -0.4, 7.0], dtype=np.float32)

# Path for learned weights
ML_LEARNED_PATH = "ml_weights_learned.npy"

# Default training hyperparameters (can be overridden from CLI)
TRAIN_GAMMA_DEFAULT = 0.95

# Teacher-training defaults
TRAIN_GAMES_DEFAULT = 20
TRAIN_MAX_MOVES_DEFAULT = 150

# RL defaults
RL_EPISODES_DEFAULT = 200
RL_MAX_STEPS_DEFAULT = 200
RL_GAMMA_DEFAULT = 0.95
RL_ALPHA_DEFAULT = 0.001
RL_EPSILON_DEFAULT = 0.1

# Heuristic1 weights (used both by the runtime agent and teacher)
H1_WEIGHTS = np.array([-0.5, -3.0, -0.3, -0.7, 5.0], dtype=np.float32)

# Heuristic2 weights (two-ply)
H2_WEIGHTS = np.array([-0.4, -2.0, -0.25, -0.5, 6.0], dtype=np.float32)
H2_GAMMA = 0.8

# ML 2-ply discount
ML_LEARNED_GAMMA = 0.8
