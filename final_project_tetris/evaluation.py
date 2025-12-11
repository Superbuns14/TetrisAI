# evaluation.py
import random
import statistics
import argparse

from config import NUM_BAGS, RNG_SEED
from tetris_core import TetrisGame, generate_bag_sequence, add_garbage
from agents import make_agent


def _play_vs_one_game(agent1_name: str, agent2_name: str,
                      max_moves: int = 2000, pps: float = 2.0):
    """
    Headless (no GUI) VS game for evaluation.
    Returns a dict with winner + stats for each player.
    """
    # Same piece sequence for fairness
    random.seed(random.randrange(1_000_000_000))  # new random seed per game
    sequence = generate_bag_sequence(NUM_BAGS)

    game1 = TetrisGame(sequence)
    game2 = TetrisGame(sequence)

    agent1 = make_agent(agent1_name)
    agent2 = make_agent(agent2_name)

    moves1 = moves2 = 0

    while not (game1.game_over or game2.game_over):
        # Player 1 move
        if not game1.game_over and moves1 < max_moves:
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
                        _, attack = game1.hard_drop_current()
                        if attack > 0:
                            add_garbage(game2, attack)
                        moves1 += 1

        # Player 2 move
        if not game2.game_over and moves2 < max_moves:
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
                        _, attack = game2.hard_drop_current()
                        if attack > 0:
                            add_garbage(game1, attack)
                        moves2 += 1

        if moves1 >= max_moves and moves2 >= max_moves:
            break

    # Decide winner
    if game1.game_over and not game2.game_over:
        winner = "P2"
    elif game2.game_over and not game1.game_over:
        winner = "P1"
    else:
        # Either both topped out same turn or both hit max_moves
        # Tie-breaker: higher total attack wins; otherwise draw
        if game1.total_attack > game2.total_attack:
            winner = "P1"
        elif game2.total_attack > game1.total_attack:
            winner = "P2"
        else:
            winner = "draw"

    # Approximate "APM at given pps"
    # time ~ moves / pps seconds
    def apm_for(game, moves):
        if moves <= 0:
            return 0.0
        time_sec = moves / max(pps, 1e-6)
        return (game.total_attack / max(time_sec, 1e-6)) * 60.0

    apm1 = apm_for(game1, moves1)
    apm2 = apm_for(game2, moves2)

    result = {
        "winner": winner,
        "p1": {
            "moves": moves1,
            "lines": game1.lines_cleared_total,
            "attack": game1.total_attack,
            "apm": apm1,
        },
        "p2": {
            "moves": moves2,
            "lines": game2.lines_cleared_total,
            "attack": game2.total_attack,
            "apm": apm2,
        },
    }
    return result


def evaluate_vs(agent1_name: str, agent2_name: str,
                games: int = 50, max_moves: int = 2000, pps: float = 2.0):
    """
    Run many headless VS games and print a summary.
    """
    print(f"Evaluating {agent1_name} vs {agent2_name}")
    print(f"Games: {games}, max_moves per player: {max_moves}, assumed PPS={pps}\n")

    wins_p1 = wins_p2 = draws = 0
    p1_apms, p2_apms = [], []
    p1_atks, p2_atks = [], []
    p1_moves, p2_moves = [], []

    for g in range(games):
        res = _play_vs_one_game(agent1_name, agent2_name, max_moves=max_moves, pps=pps)
        w = res["winner"]
        if w == "P1":
            wins_p1 += 1
        elif w == "P2":
            wins_p2 += 1
        else:
            draws += 1

        p1_apms.append(res["p1"]["apm"])
        p2_apms.append(res["p2"]["apm"])
        p1_atks.append(res["p1"]["attack"])
        p2_atks.append(res["p2"]["attack"])
        p1_moves.append(res["p1"]["moves"])
        p2_moves.append(res["p2"]["moves"])

        print(
            f"Game {g+1:3d}: winner={w}, "
            f"P1 atk={res['p1']['attack']:4.1f}, APM≈{res['p1']['apm']:5.1f}; "
            f"P2 atk={res['p2']['attack']:4.1f}, APM≈{res['p2']['apm']:5.1f}"
        )

    def avg(xs):
        return statistics.mean(xs) if xs else 0.0

    print("\n=== SUMMARY ===")
    print(f"P1 ({agent1_name}) wins: {wins_p1}")
    print(f"P2 ({agent2_name}) wins: {wins_p2}")
    print(f"Draws: {draws}")
    print(f"Win rate P1: {wins_p1 / max(games,1):.3f}")
    print(f"Win rate P2: {wins_p2 / max(games,1):.3f}")
    print()
    print(f"P1 avg attack: {avg(p1_atks):.2f}")
    print(f"P2 avg attack: {avg(p2_atks):.2f}")
    print(f"P1 avg APM (approx): {avg(p1_apms):.2f}")
    print(f"P2 avg APM (approx): {avg(p2_apms):.2f}")
    print(f"P1 avg moves: {avg(p1_moves):.1f}")
    print(f"P2 avg moves: {avg(p2_moves):.1f}")


# Allow running this file directly too
def main():
    parser = argparse.ArgumentParser(
        description="Headless evaluation of Tetris AI agents (VS mode)."
    )
    parser.add_argument("--agent1", type=str, default="ml_base",
                        help="Agent for player 1.")
    parser.add_argument("--agent2", type=str, default="heuristic2",
                        help="Agent for player 2.")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games to run.")
    parser.add_argument("--max-moves", type=int, default=2000,
                        help="Maximum moves per player per game.")
    parser.add_argument("--pps", type=float, default=2.0,
                        help="Assumed pieces per second (for approximate APM).")

    args = parser.parse_args()
    evaluate_vs(
        agent1_name=args.agent1,
        agent2_name=args.agent2,
        games=args.games,
        max_moves=args.max_moves,
        pps=args.pps,
    )


if __name__ == "__main__":
    main()
