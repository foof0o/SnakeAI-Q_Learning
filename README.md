# SnakeAI-Q_Learning

SnakeAI-Q_Learning is a Python reinforcement learning project where an autonomous agent learns to play Snake using tabular Q-learning. The project includes a playable PyGame visualization, headless training mode, , A* training mode, persistent Q-table saves, configurable reward settings, and multiple game modes for testing the agent against different environments.


## Features

- Q-learning agent that updates action values with the Bellman equation.
- A* pathfinding guidance to improve early exploration and stabilize training.
- Epsilon-greedy action selection with decaying exploration.
- Three game modes:
  - Mode 1: single Snake agent.
  - Mode 2: two-snake environment with separate Q-table training for the second snake.
  - Mode 3: single Snake agent with static obstacles.
  - Mode 4: two-snake environment where one operates via A* and the other is an agent.
- Headless training for faster long-running experiments.
- Configurable reward and learning parameters through a custom settings menu.
- Save/load system that stores Q-tables and training metrics by mode and configuration hash.
- Matplotlib charts for score, reward, and episode length trends.

## Tech Stack

- Python
- PyGame
- Matplotlib
- Reinforcement learning with tabular Q-learning
- A* search for path recommendation

## Getting Started

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the visual version:

```bash
python Snake_Game.py
```

## How It Works

The agent represents each state using nearby danger information, current movement direction, and the relative direction of the food. For each state, it chooses between three relative actions: continue straight, turn left, or turn right.

During training, the agent balances exploration and exploitation using epsilon-greedy selection. A* pathfinding provides a recommended action, especially useful early in training when the Q-table is still sparse. After each move, the Q-table is updated based on the reward received and the best estimated future value.

Rewards are shaped around food collection, death avoidance, distance to food, and anti-stalling behavior. This helps the agent learn more efficiently than relying only on a sparse food/death reward.

## Saved Training Data

Training data is saved in the directory. Save files include:

- Q-table values.
- Current epsilon and A* bias.
- Episode number.
- Score, reward, and length history.
- The reward and learning settings used for that training run.

The save system uses a short hash of the active mode and learning settings so different experiments do not overwrite each other.

## Project Highlights

This project demonstrates practical experience with reinforcement learning fundamentals, game simulation, pathfinding, state design, reward shaping, CLI tooling, data persistence, and training visualization in Python.
