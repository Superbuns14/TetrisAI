# game_loop.py
import pygame
import random

from config import (
    ROWS, COLS,
    BOARD_WIDTH, BOARD_HEIGHT,
    PANEL_WIDTH, MARGIN, FPS,
    TARGET_PPS_DEFAULT,
    NUM_BAGS, RNG_SEED,
)
from tetris_core import TetrisGame, generate_bag_sequence, PIECE_COLORS, GARBAGE_COLOR
from agents import make_agent
from tetris_core import TETROMINOES, add_garbage

def draw_mini_piece(screen, shape_key, origin_x, origin_y, cell_size):
    from tetris_core import TETROMINOES, PIECE_COLORS
    matrix = TETROMINOES[shape_key][0]
    color = PIECE_COLORS[shape_key]
    for r in range(4):
        for c in range(4):
            if matrix[r][c]:
                x = origin_x + c * cell_size
                y = origin_y + r * cell_size
                pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

def draw_board_with_panel(screen, game, origin_x, origin_y,
                          label, controller_name, apm, pps, elapsed_seconds):
    CELL_SIZE = int(BOARD_HEIGHT / ROWS)  # consistent scale

    # board
    pygame.draw.rect(screen, (0,0,0), (origin_x, origin_y, BOARD_WIDTH, BOARD_HEIGHT))
    for r in range(ROWS):
        for c in range(COLS):
            color = game.board[r][c]
            if color != (0,0,0):
                pygame.draw.rect(
                    screen, color,
                    (origin_x + c*CELL_SIZE, origin_y + r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

    # current piece
    if not game.game_over:
        piece = game.current_piece
        matrix = piece.matrix
        for r in range(4):
            for c in range(4):
                if matrix[r][c]:
                    x = piece.x + c
                    y = piece.y + r
                    if 0 <= x < COLS and 0 <= y < ROWS:
                        pygame.draw.rect(
                            screen, PIECE_COLORS[piece.shape],
                            (origin_x + x*CELL_SIZE, origin_y + y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        )

    # grid
    for x in range(COLS+1):
        pygame.draw.line(
            screen, (40,40,40),
            (origin_x + x*CELL_SIZE, origin_y),
            (origin_x + x*CELL_SIZE, origin_y + BOARD_HEIGHT), 1
        )
    for y in range(ROWS+1):
        pygame.draw.line(
            screen, (40,40,40),
            (origin_x, origin_y + y*CELL_SIZE),
            (origin_x + BOARD_WIDTH, origin_y + y*CELL_SIZE), 1
        )

    # side panel
    panel_x = origin_x + BOARD_WIDTH + 5
    panel_y = origin_y
    pygame.draw.rect(
        screen, (20,20,20),
        (panel_x, panel_y, PANEL_WIDTH, BOARD_HEIGHT)
    )

    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 28)

    sy = panel_y + 5
    title = big_font.render(label, True, (255,255,255))
    screen.blit(title, (panel_x + 5, sy))
    sy += 30
    ctrl_surf = font.render(f"Agent: {controller_name}", True, (200,200,200))
    screen.blit(ctrl_surf, (panel_x + 5, sy))
    sy += 25

    time_surf  = font.render(f"Time: {elapsed_seconds:5.1f}s", True, (255,255,255))
    lines_surf = font.render(f"Lines: {game.lines_cleared_total}", True, (255,255,255))
    atk_surf   = font.render(f"ATK: {game.total_attack}", True, (255,255,255))
    apm_surf   = font.render(f"APM: {apm:.1f}", True, (255,255,255))
    pps_surf   = font.render(f"PPS: {pps:.2f}", True, (255,255,255))

    screen.blit(time_surf,  (panel_x + 5, sy)); sy += 20
    screen.blit(lines_surf, (panel_x + 5, sy)); sy += 20
    screen.blit(atk_surf,   (panel_x + 5, sy)); sy += 20
    screen.blit(apm_surf,   (panel_x + 5, sy)); sy += 20
    screen.blit(pps_surf,   (panel_x + 5, sy)); sy += 30

    from config import NEXT_PREVIEW_COUNT
    mini_size = CELL_SIZE // 2

    hold_label = big_font.render("HOLD", True, (255,255,255))
    screen.blit(hold_label, (panel_x + 5, sy))
    sy += 25
    hold_box_y = sy
    pygame.draw.rect(
        screen, (40,40,40),
        (panel_x + 5, hold_box_y, mini_size*4 + 4, mini_size*4 + 4), 1
    )
    if game.hold_piece is not None:
        draw_mini_piece(screen, game.hold_piece.shape, panel_x + 7, hold_box_y + 2, mini_size)
    sy += mini_size*4 + 12

    next_label = big_font.render("NEXT", True, (255,255,255))
    screen.blit(next_label, (panel_x + 5, sy))
    sy += 25
    for i, p in enumerate(game.next_queue[:NEXT_PREVIEW_COUNT]):
        box_y = sy + i*(mini_size*4 + 10)
        pygame.draw.rect(
            screen, (40,40,40),
            (panel_x + 5, box_y, mini_size*4 + 4, mini_size*4 + 4), 1
        )
        draw_mini_piece(screen, p.shape, panel_x + 7, box_y + 2, mini_size)

def run_single(agent_name: str, pps: float = TARGET_PPS_DEFAULT):
    pygame.init()
    total_width = BOARD_WIDTH + PANEL_WIDTH + 2*MARGIN
    total_height = BOARD_HEIGHT + 2*MARGIN
    screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Tetris Single (bot)")
    clock = pygame.time.Clock()

    random.seed(RNG_SEED)
    sequence = generate_bag_sequence(NUM_BAGS)
    game = TetrisGame(sequence)
    agent = make_agent(agent_name)

    min_ms_per_piece = int(1000.0 / max(pps, 0.01))

    last_placement_time = pygame.time.get_ticks()
    running = True

    stats_frozen = False
    frozen_pps = 0.0
    frozen_apm = 0.0
    frozen_time = 0.0

    while running:
        dt = clock.tick(FPS)
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.game_over and now - last_placement_time >= min_ms_per_piece:
            mv = agent.select_move(game)
            if mv is None:
                game.game_over = True
            else:
                if mv.get("use_hold", False) and game.can_hold():
                    game.apply_hold()
                if not game.game_over:
                    piece = game.current_piece
                    piece.rotation = mv["rot"]
                    piece.x = mv["x"]
                    piece.y = 0
                    if not game.valid_position(piece):
                        game.game_over = True
                    else:
                        if game.start_time_ms is None:
                            game.start_time_ms = now
                        game.hard_drop_current()
                        last_placement_time = now
                        pygame.time.delay(50)  # small visual delay

        if game.start_time_ms is not None and game.total_placements > 0:
            elapsed = max(1e-6, (now - game.start_time_ms)/1000.0)
            current_pps = game.total_placements / elapsed
            current_apm = (game.total_attack / elapsed) * 60.0

            if game.game_over and not stats_frozen:
                frozen_pps = current_pps
                frozen_apm = current_apm
                frozen_time = elapsed
                stats_frozen = True

            if stats_frozen:
                pps_val = frozen_pps
                apm_val = frozen_apm
                elapsed_seconds = frozen_time
            else:
                pps_val = current_pps
                apm_val = current_apm
                elapsed_seconds = elapsed
        else:
            pps_val = 0.0
            apm_val = 0.0
            elapsed_seconds = 0.0

        screen.fill((0,0,0))
        ox = MARGIN
        oy = MARGIN
        label = "Single (Game Over)" if game.game_over else "Single"
        draw_board_with_panel(screen, game, ox, oy, label, agent_name, apm_val, pps_val, elapsed_seconds)
        pygame.display.flip()

    pygame.quit()

def run_vs(agent1_name: str, agent2_name: str, pps: float = TARGET_PPS_DEFAULT):
    pygame.init()
    total_width = 2*(BOARD_WIDTH + PANEL_WIDTH) + 3*MARGIN
    total_height = BOARD_HEIGHT + 2*MARGIN
    screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Tetris VS (bots)")
    clock = pygame.time.Clock()

    random.seed(RNG_SEED)
    sequence = generate_bag_sequence(NUM_BAGS)
    game1 = TetrisGame(sequence)
    game2 = TetrisGame(sequence)
    agent1 = make_agent(agent1_name)
    agent2 = make_agent(agent2_name)

    min_ms_per_piece = int(1000.0 / max(pps, 0.01))

    last_time1 = pygame.time.get_ticks()
    last_time2 = pygame.time.get_ticks()

    match_over = False
    winner_text = ""

    frozen_pps1 = frozen_apm1 = frozen_time1 = None
    frozen_pps2 = frozen_apm2 = frozen_time2 = None

    running = True
    while running:
        dt = clock.tick(FPS)
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not match_over:
            if not game1.game_over and now - last_time1 >= min_ms_per_piece:
                mv1 = agent1.select_move(game1)
                if mv1 is None:
                    game1.game_over = True
                else:
                    if mv1.get("use_hold", False) and game1.can_hold():
                        game1.apply_hold()
                    if not game1.game_over:
                        piece = game1.current_piece
                        piece.rotation = mv1["rot"]
                        piece.x = mv1["x"]
                        piece.y = 0
                        if not game1.valid_position(piece):
                            game1.game_over = True
                        else:
                            if game1.start_time_ms is None:
                                game1.start_time_ms = now
                            _, attack = game1.hard_drop_current()
                            if attack > 0:
                                add_garbage(game2, attack)
                            last_time1 = now
                            pygame.time.delay(30)

            if not game2.game_over and now - last_time2 >= min_ms_per_piece:
                mv2 = agent2.select_move(game2)
                if mv2 is None:
                    game2.game_over = True
                else:
                    if mv2.get("use_hold", False) and game2.can_hold():
                        game2.apply_hold()
                    if not game2.game_over:
                        piece = game2.current_piece
                        piece.rotation = mv2["rot"]
                        piece.x = mv2["x"]
                        piece.y = 0
                        if not game2.valid_position(piece):
                            game2.game_over = True
                        else:
                            if game2.start_time_ms is None:
                                game2.start_time_ms = now
                            _, attack = game2.hard_drop_current()
                            if attack > 0:
                                add_garbage(game1, attack)
                            last_time2 = now
                            pygame.time.delay(30)

            if game1.game_over or game2.game_over:
                match_over = True
                if game1.game_over and not game2.game_over:
                    winner_text = "P2 WINS"
                elif game2.game_over and not game1.game_over:
                    winner_text = "P1 WINS"
                else:
                    winner_text = "DRAW"

                if game1.start_time_ms is not None and game1.total_placements > 0:
                    elapsed1 = max(1e-6, (now - game1.start_time_ms)/1000.0)
                    frozen_pps1 = game1.total_placements / elapsed1
                    frozen_apm1 = (game1.total_attack / elapsed1) * 60.0
                    frozen_time1 = elapsed1
                else:
                    frozen_pps1 = 0.0
                    frozen_apm1 = 0.0
                    frozen_time1 = 0.0

                if game2.start_time_ms is not None and game2.total_placements > 0:
                    elapsed2 = max(1e-6, (now - game2.start_time_ms)/1000.0)
                    frozen_pps2 = game2.total_placements / elapsed2
                    frozen_apm2 = (game2.total_attack / elapsed2) * 60.0
                    frozen_time2 = elapsed2
                else:
                    frozen_pps2 = 0.0
                    frozen_apm2 = 0.0
                    frozen_time2 = 0.0

        if match_over:
            pps1, apm1, elapsed1 = frozen_pps1, frozen_apm1, frozen_time1
            pps2, apm2, elapsed2 = frozen_pps2, frozen_apm2, frozen_time2
        else:
            if game1.start_time_ms is not None and game1.total_placements > 0:
                elapsed1 = max(1e-6, (now - game1.start_time_ms)/1000.0)
                pps1 = game1.total_placements / elapsed1
                apm1 = (game1.total_attack / elapsed1) * 60.0
            else:
                pps1 = apm1 = 0.0
                elapsed1 = 0.0

            if game2.start_time_ms is not None and game2.total_placements > 0:
                elapsed2 = max(1e-6, (now - game2.start_time_ms)/1000.0)
                pps2 = game2.total_placements / elapsed2
                apm2 = (game2.total_attack / elapsed2) * 60.0
            else:
                pps2 = apm2 = 0.0
                elapsed2 = 0.0

        screen.fill((0,0,0))
        ox1 = MARGIN
        oy1 = MARGIN
        label1 = "P1 (Game Over)" if game1.game_over else "P1"
        label2 = "P2 (Game Over)" if game2.game_over else "P2"

        draw_board_with_panel(screen, game1, ox1, oy1, label1, agent1_name, apm1, pps1, elapsed1)
        ox2 = ox1 + BOARD_WIDTH + PANEL_WIDTH + MARGIN
        oy2 = MARGIN
        draw_board_with_panel(screen, game2, ox2, oy2, label2, agent2_name, apm2, pps2, elapsed2)

        if match_over:
            font = pygame.font.Font(None, 48)
            text_surf = font.render(winner_text, True, (255,255,255))
            text_rect = text_surf.get_rect(center=(total_width//2, total_height//2))
            screen.blit(text_surf, text_rect)

        pygame.display.flip()

    pygame.quit()
