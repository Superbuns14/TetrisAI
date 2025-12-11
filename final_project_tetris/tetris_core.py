# tetris_core.py
import copy
import random
import numpy as np

from config import (
    ROWS, COLS,
    NEXT_PREVIEW_COUNT,
    T_SPIN_SINGLE_ATTACK, T_SPIN_DOUBLE_ATTACK, T_SPIN_TRIPLE_ATTACK,
    PERFECT_CLEAR_ATTACK_BONUS,
)

# ----- Piece definitions -----

TETROMINOES = {
    'I': [
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
    ],
    'O': [
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
    ],
    'T': [
        [[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
    ],
    'S': [
        [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
        [[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
    ],
    'Z': [
        [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]],
    ],
    'J': [
        [[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]],
        [[1,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
    ],
    'L': [
        [[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]],
        [[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]],
    ],
}

PIECE_COLORS = {
    'I': (0,255,255),
    'O': (255,255,0),
    'T': (128,0,128),
    'S': (0,255,0),
    'Z': (255,0,0),
    'J': (0,0,255),
    'L': (255,165,0),
}

GARBAGE_COLOR = (80,80,80)


def generate_bag_sequence(num_bags: int):
    seq = []
    for _ in range(num_bags):
        bag = list(TETROMINOES.keys())
        random.shuffle(bag)
        seq.extend(bag)
    return seq


class Piece:
    def __init__(self, shape: str):
        self.shape = shape
        self.rotation = 0
        self.x = COLS // 2 - 2
        self.y = 0

    @property
    def matrix(self):
        return TETROMINOES[self.shape][self.rotation]


def is_t_spin(board_with_piece, piece: Piece) -> bool:
    """3-corner T-spin detection (after placement & clear)."""
    if piece.shape != 'T':
        return False

    cx = piece.x + 1
    cy = piece.y + 1
    corners = [
        (cx - 1, cy - 1),
        (cx + 1, cy - 1),
        (cx - 1, cy + 1),
        (cx + 1, cy + 1),
    ]

    filled = 0
    for (x, y) in corners:
        if x < 0 or x >= COLS or y < 0 or y >= ROWS:
            filled += 1
        elif board_with_piece[y][x] != (0,0,0):
            filled += 1
    return filled >= 3


class TetrisGame:
    def __init__(self, sequence):
        self.board = [[(0,0,0) for _ in range(COLS)] for _ in range(ROWS)]
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False

        self.sequence = sequence
        self.seq_index = 0

        self.next_queue = []
        self.hold_piece = None
        self.hold_used = False

        self.combo = 0
        self.total_attack = 0
        self.total_placements = 0

        self.start_time_ms = None  # used by UI for PPS/APM

        self.last_t_spin_type = 0
        self.back_to_back = False

        self._init_pieces()

    # ----- piece sequence -----

    def _get_next_shape(self):
        if self.seq_index >= len(self.sequence):
            self.seq_index = 0
        shape = self.sequence[self.seq_index]
        self.seq_index += 1
        return shape

    def _add_to_queue(self):
        shape = self._get_next_shape()
        self.next_queue.append(Piece(shape))

    def _init_pieces(self):
        self.next_queue = []
        for _ in range(NEXT_PREVIEW_COUNT + 1):
            self._add_to_queue()
        self.current_piece = self.next_queue.pop(0)

    def reset(self):
        self.board = [[(0,0,0) for _ in range(COLS)] for _ in range(ROWS)]
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False

        self.seq_index = 0
        self.next_queue = []
        self.hold_piece = None
        self.hold_used = False
        self.combo = 0
        self.total_attack = 0
        self.total_placements = 0
        self.start_time_ms = None

        self.last_t_spin_type = 0
        self.back_to_back = False

        self._init_pieces()

    def clone(self):
        new = TetrisGame(self.sequence)
        new.board = copy.deepcopy(self.board)
        new.score = self.score
        new.lines_cleared_total = self.lines_cleared_total
        new.game_over = self.game_over

        new.sequence = self.sequence
        new.seq_index = self.seq_index
        new.next_queue = [copy.deepcopy(p) for p in self.next_queue]
        new.current_piece = copy.deepcopy(self.current_piece)

        new.hold_piece = copy.deepcopy(self.hold_piece) if self.hold_piece else None
        new.hold_used = self.hold_used
        new.combo = self.combo
        new.total_attack = self.total_attack
        new.total_placements = self.total_placements
        new.start_time_ms = self.start_time_ms

        new.last_t_spin_type = self.last_t_spin_type
        new.back_to_back = self.back_to_back
        return new

    # ----- movement / hold -----

    def valid_position(self, piece: Piece, offset_x=0, offset_y=0, rotation=None):
        if rotation is None:
            rotation = piece.rotation
        matrix = TETROMINOES[piece.shape][rotation]
        for r in range(4):
            for c in range(4):
                if matrix[r][c]:
                    x = piece.x + c + offset_x
                    y = piece.y + r + offset_y
                    if x < 0 or x >= COLS or y < 0 or y >= ROWS:
                        return False
                    if self.board[y][x] != (0,0,0):
                        return False
        return True

    def can_hold(self):
        return not self.hold_used

    def apply_hold(self):
        if not self.can_hold():
            return
        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.current_piece = self.next_queue.pop(0)
            self._add_to_queue()
        else:
            self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
        self.current_piece.x = COLS // 2 - 2
        self.current_piece.y = 0
        self.current_piece.rotation = 0
        self.hold_used = True

    # ----- line clearing / locking -----

    def clear_lines(self):
        new_board = []
        lines_cleared = 0
        for row in self.board:
            if all(cell != (0,0,0) for cell in row):
                lines_cleared += 1
            else:
                new_board.append(row)
        for _ in range(lines_cleared):
            new_board.insert(0, [(0,0,0) for _ in range(COLS)])
        self.board = new_board
        return lines_cleared

    def lock_piece(self):
        piece = self.current_piece
        matrix = piece.matrix

        for r in range(4):
            for c in range(4):
                if matrix[r][c]:
                    x = piece.x + c
                    y = piece.y + r
                    if 0 <= y < ROWS:
                        self.board[y][x] = PIECE_COLORS[piece.shape]
                    else:
                        self.game_over = True

        board_with_piece = [row[:] for row in self.board]

        # lines / attack
        lines = self.clear_lines()

        t_spin_type = 0
        if piece.shape == 'T' and lines > 0 and is_t_spin(board_with_piece, piece):
            t_spin_type = min(lines, 3)
        self.last_t_spin_type = t_spin_type

        is_perfect_clear = all(
            all(cell == (0,0,0) for cell in row) for row in self.board
        ) and lines > 0

        attack = 0
        if lines > 0:
            if t_spin_type > 0:
                if t_spin_type == 1:
                    attack = T_SPIN_SINGLE_ATTACK
                elif t_spin_type == 2:
                    attack = T_SPIN_DOUBLE_ATTACK
                else:
                    attack = T_SPIN_TRIPLE_ATTACK
            else:
                if lines == 1:
                    attack = 0
                elif lines == 2:
                    attack = 1
                elif lines == 3:
                    attack = 2
                else:
                    attack = 4  # base Tetris

            if self.combo > 0:
                attack += 1

            if is_perfect_clear:
                attack += PERFECT_CLEAR_ATTACK_BONUS

        if lines > 0:
            self.combo += 1
        else:
            self.combo = 0

        # B2B
        is_big = (t_spin_type > 0) or (lines >= 4)
        if is_big:
            if self.back_to_back:
                attack += 1
            self.back_to_back = True
        elif lines > 0:
            self.back_to_back = False

        self.total_attack += attack
        self.total_placements += 1
        self.lines_cleared_total += lines

        # score (for fun)
        if lines == 1:
            self.score += 40
        elif lines == 2:
            self.score += 100
        elif lines == 3:
            self.score += 300
        elif lines >= 4:
            self.score += 1200

        # next piece
        self.current_piece = self.next_queue.pop(0)
        self._add_to_queue()
        self.hold_used = False

        if not self.valid_position(self.current_piece):
            self.game_over = True

        return lines, attack

    def hard_drop_current(self):
        if self.game_over:
            return 0, 0
        while self.valid_position(self.current_piece, offset_y=1):
            self.current_piece.y += 1
        return self.lock_piece()


def add_garbage(game: TetrisGame, n_lines: int):
    if n_lines <= 0 or game.game_over:
        return
    for _ in range(n_lines):
        game.board.pop(0)
        hole_col = random.randint(0, COLS-1)
        row = [GARBAGE_COLOR for _ in range(COLS)]
        row[hole_col] = (0,0,0)
        game.board.append(row)

# ----- Feature extraction -----

def extract_board_heights(board):
    heights = [0]*COLS
    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] != (0,0,0):
                heights[c] = ROWS - r
                break
    return heights

def count_holes(board):
    holes = 0
    for c in range(COLS):
        seen = False
        for r in range(ROWS):
            if board[r][c] != (0,0,0):
                seen = True
            elif seen:
                holes += 1
    return holes

def aggregate_height(heights):
    return sum(heights)

def bumpiness(heights):
    return sum(abs(heights[c] - heights[c+1]) for c in range(COLS-1))

def max_height(heights):
    return max(heights) if heights else 0

def lines_cleared_by_placement(board_before, board_after):
    def full_rows(b):
        return sum(1 for row in b if all(cell != (0,0,0) for cell in row))
    return full_rows(board_after) - full_rows(board_before)

def extract_features(board_before, board_after):
    heights = extract_board_heights(board_after)
    agg_h = aggregate_height(heights)
    holes = count_holes(board_after)
    bump = bumpiness(heights)
    max_h = max_height(heights)
    lines = lines_cleared_by_placement(board_before, board_after)
    return np.array([agg_h, holes, bump, max_h, lines], dtype=np.float32)

# ----- Move simulation -----

def simulate_move(game: TetrisGame, use_hold: bool, rot: int, x: int):
    """Simulate a root move: optional hold, rotation, x placement, then hard drop."""
    sim = game.clone()
    was_b2b = sim.back_to_back

    if use_hold and sim.can_hold():
        sim.apply_hold()

    piece = sim.current_piece
    rotations = len(TETROMINOES[piece.shape])
    if rot < 0 or rot >= rotations:
        return None

    piece.rotation = rot
    piece.x = x
    piece.y = 0

    if not sim.valid_position(piece):
        return None

    while sim.valid_position(piece, offset_y=1):
        piece.y += 1

    board_before = [row[:] for row in sim.board]
    lines_before = sim.lines_cleared_total
    attack_before = sim.total_attack

    sim.hard_drop_current()
    board_after = [row[:] for row in sim.board]

    lines_gained = sim.lines_cleared_total - lines_before
    attack_gained = sim.total_attack - attack_before
    t_spin_type = sim.last_t_spin_type
    is_t_spin_flag = t_spin_type > 0

    is_big = (t_spin_type > 0) or (lines_gained >= 4)
    breaks_b2b = was_b2b and lines_gained > 0 and (not is_big)

    return {
        "sim": sim,
        "use_hold": use_hold,
        "rot": rot,
        "x": x,
        "board_before": board_before,
        "board_after": board_after,
        "lines": lines_gained,
        "attack": attack_gained,
        "t_spin_type": t_spin_type,
        "is_t_spin": is_t_spin_flag,
        "breaks_b2b": breaks_b2b,
    }

def enumerate_root_moves(game: TetrisGame):
    """Enumerate all root moves (with/without hold, all rotations, all x in a wide range)."""
    moves = []
    piece = game.current_piece
    rot_count = len(TETROMINOES[piece.shape])

    for rot in range(rot_count):
        for x in range(-2, COLS):
            res = simulate_move(game, False, rot, x)
            if res is not None:
                moves.append(res)

    if game.can_hold():
        tmp = game.clone()
        tmp.apply_hold()
        held_shape = tmp.current_piece.shape
        held_rot_count = len(TETROMINOES[held_shape])
        for rot in range(held_rot_count):
            for x in range(-2, COLS):
                res = simulate_move(game, True, rot, x)
                if res is not None:
                    moves.append(res)

    return moves

def enumerate_child_moves_no_hold(game: TetrisGame):
    moves = []
    piece = game.current_piece
    rot_count = len(TETROMINOES[piece.shape])
    for rot in range(rot_count):
        for x in range(-2, COLS):
            res = simulate_move(game, False, rot, x)
            if res is not None:
                moves.append(res)
    return moves

def enumerate_root_moves_fast_training(game: TetrisGame):
    """Simpler, faster move enumeration used only for training."""
    moves = []
    piece = game.current_piece
    rot_count = len(TETROMINOES[piece.shape])
    for rot in range(rot_count):
        for x in range(COLS):
            res = simulate_move(game, False, rot, x)
            if res is not None:
                moves.append(res)
    return moves
