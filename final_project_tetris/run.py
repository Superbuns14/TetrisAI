# run.py
import argparse

from config import (
    TARGET_PPS_DEFAULT,
    TRAIN_GAMES_DEFAULT, TRAIN_MAX_MOVES_DEFAULT,
)
from game_loop import run_single, run_vs
from training import train_ml_from_selfplay, train_ml_rl
from evaluation import evaluate_vs  # NEW


def main():
    parser = argparse.ArgumentParser(
        description="Tetris AI - play, train, and evaluate different agents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- play single ---
    sp_single = subparsers.add_parser("play-single", help="Run a single-board bot game.")
    sp_single.add_argument(
        "--agent1", type=str, default="heuristic2",
        help="Agent for the single board (random, heuristic1, heuristic2, ml_base, ml_learned)."
    )
    sp_single.add_argument(
        "--pps", type=float, default=TARGET_PPS_DEFAULT,
        help="Pieces per second (visual speed)."
    )

    # --- play vs ---
    sp_vs = subparsers.add_parser("play-vs", help="Run a versus match between two bots.")
    sp_vs.add_argument(
        "--agent1", type=str, default="ml_base",
        help="Agent for player 1."
    )
    sp_vs.add_argument(
        "--agent2", type=str, default="heuristic2",
        help="Agent for player 2."
    )
    sp_vs.add_argument(
        "--pps", type=float, default=TARGET_PPS_DEFAULT,
        help="Pieces per second (visual speed)."
    )

    # --- train teacher ---
    sp_tt = subparsers.add_parser("train-teacher", help="Teacher-based ML training from self-play.")
    sp_tt.add_argument(
        "--games", type=int, default=TRAIN_GAMES_DEFAULT,
        help="Number of teacher games to generate."
    )
    sp_tt.add_argument(
        "--max-moves", type=int, default=TRAIN_MAX_MOVES_DEFAULT,
        help="Max moves per game."
    )
    sp_tt.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount factor for returns."
    )

    # --- train RL ---
    sp_rl = subparsers.add_parser("train-rl", help="Reinforcement learning training (self-play).")
    sp_rl.add_argument(
        "--episodes", type=int, default=200,
        help="Number of RL episodes."
    )
    sp_rl.add_argument(
        "--max-steps", type=int, default=200,
        help="Max steps per episode."
    )
    sp_rl.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount factor."
    )
    sp_rl.add_argument(
        "--alpha", type=float, default=0.001,
        help="Learning rate."
    )
    sp_rl.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Exploration rate."
    )

    # --- eval vs (headless) ---
    sp_eval = subparsers.add_parser("eval-vs", help="Headless evaluation of two agents in VS mode.")
    sp_eval.add_argument(
        "--agent1", type=str, default="ml_base",
        help="Agent for player 1."
    )
    sp_eval.add_argument(
        "--agent2", type=str, default="heuristic2",
        help="Agent for player 2."
    )
    sp_eval.add_argument(
        "--games", type=int, default=20,
        help="Number of games to run."
    )
    sp_eval.add_argument(
        "--max-moves", type=int, default=2000,
        help="Maximum moves per player per game."
    )
    sp_eval.add_argument(
        "--pps", type=float, default=2.0,
        help="Assumed pieces per second (for approximate APM)."
    )

    args = parser.parse_args()

    if args.command == "play-single":
        run_single(agent_name=args.agent1, pps=args.pps)

    elif args.command == "play-vs":
        run_vs(agent1_name=args.agent1, agent2_name=args.agent2, pps=args.pps)

    elif args.command == "train-teacher":
        train_ml_from_selfplay(
            num_games=args.games,
            max_moves=args.max_moves,
            gamma=args.gamma,
        )

    elif args.command == "train-rl":
        train_ml_rl(
            episodes=args.episodes,
            max_steps=args.max_steps,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=args.epsilon,
        )

    elif args.command == "eval-vs":
        evaluate_vs(
            agent1_name=args.agent1,
            agent2_name=args.agent2,
            games=args.games,
            max_moves=args.max_moves,
            pps=args.pps,
        )


if __name__ == "__main__":
    main()
