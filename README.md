# Tetris AI: Heuristic and Machine Learning Agents

This project implements a real-time Tetris engine and several AI agents of increasing sophistication.  
It includes:

- A complete Tetris game engine with:
  - hold piece, next queue, T-Spin detection (3-corner rule), perfect clears  
  - back-to-back bonuses, garbage interaction, line clear logic  
- A real-time GUI (Pygame) for visual play
- Multiple AI agents:
  - `random` — baseline  
  - `heuristic1` — single-ply board evaluation  
  - `heuristic2` — two-ply lookahead  
  - `ml_base` — linear model with hand-tuned weights  
  - `ml_learned` — machine-learned model (supervised / teacher-trained)  
  - RL training code (experimental; see below)

The system supports:

- **Single-player bot play**
- **Bot-vs-bot play** (with garbage)
- **Teacher-based ML training**
- **Reinforcement learning (experimental)**
- **Headless evaluation** over many matches

---

# Requirements

### Python
- Python **3.9+**

### Dependencies
Install via:

```bash
pip install numpy pygame
````

No other packages are required.

---

# Project Structure

```
config.py               # Shared constants and hyperparameters
tetris_core.py          # Core Tetris logic: board, pieces, collisions, line clears, T-Spins
agents.py               # Agent implementations (random, heuristic1/2, ML models)
training.py             # Teacher-based training and reinforcement learning code
game_loop.py            # Pygame GUI loops for single-player and VS matches
evaluation.py           # Headless VS evaluation of agent matchups
run.py                  # Main CLI entrypoint for all modes (play, train, eval)
ml_weights_learned.npy  # (optional) saved weights from teacher-training
README.md               # This file
```

---

# Basic Usage

All interaction goes through:

```bash
python run.py <command> [options]
```

Available commands:

* `play-single`
* `play-vs`
* `train-teacher`
* `train-rl` (experimental)
* `eval-vs`

---

## 1. Single-Player Bot (GUI)

Run:

```bash
python run.py play-single --agent1 heuristic2 --pps 2.0
```

Options:

* `--agent1` may be:

  * `random`
  * `heuristic1`
  * `heuristic2`
  * `ml_base`
  * `ml_learned`
* `--pps` = pieces per second (visual speed)

---

## 2. Bot vs Bot (GUI)

```bash
python run.py play-vs --agent1 ml_learned --agent2 heuristic2 --pps 2.0
```

Both boards:

* use **identical piece sequences**
* send **garbage lines** to each other based on attack rules

---

# Machine Learning Training

The ML agent uses a **linear value function** over hand-crafted board features
(aggregate height, holes, bumpiness, attack, lines, B2B state, etc.).

Two training modes exist.

---

## 3. Teacher-Based Training (Recommended)

This mode imitates the stronger heuristic agent (`heuristic2`) to learn better weights.

Run:

```bash
python run.py train-teacher --games 40 --max-moves 200 --gamma 0.95
```

This:

* generates several heuristic2 games,
* saves weights to `ml_weights_learned.npy`,
* updates the performance of `ml_learned`.

These weights make **ml_learned** competitive and strong.

---

## 4. Reinforcement Learning (Experimental)

```bash
python run.py train-rl --episodes 200 --max-steps 200 --gamma 0.95 --alpha 0.0001 --epsilon 0.1
```

**Important note:**
Due to extremely sparse rewards (attack only), this simple RL setup is unstable and generally **reduces** performance.
This code is included as part of the research exploration, but **teacher training produces much better results**.

---

# Evaluating Agents (No GUI)

To benchmark agents over many games:

```bash
python run.py eval-vs --agent1 heuristic1 --agent2 heuristic2 --games 50 --max-moves 2000 --pps 2.0
```

This prints:

* match winner per game
* total attack & approximate APM
* overall win rates
* average attack / APM / moves

You can also directly use:

```bash
python evaluation.py --agent1 heuristic1 --agent2 heuristic2 --games 50
```
