# AI Pong Duel

I wanted a simple sandbox to keep my PyTorch muscles fresh, so I rebuilt Pong with two agents that never stop learning:

- **Left paddle** – a PyTorch MLP with two tanh hidden layers that trains every frame against a hand-written teacher signal.
- **Right paddle** – a tiny DQN that reads the same signals and learns directly from sparse reward, exploration, and paddle hits.
- **Shared inputs** – paddle Y, ball X/Y, opponent paddle Y, and normalized ball velocity components.
- **Visuals** – a lightweight Pygame render (paddles, ball, center stripe, HUD for losses/epsilon/reward).

### Skills on display
- Real-time PyTorch training loops without datasets (online supervised + on-policy TD learning).
- Classic game-loop engineering in Pygame with deterministic physics and HUD overlays.
- Feature engineering + normalization for stable learning signals.
- Reward shaping and epsilon schedules to keep agents from collapsing.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`

Dependencies: `pygame`, `numpy`, `torch` (CPU build installs via pip by default).

## Running the Simulation

```bash
python main.py
```

Controls:

- `Esc` – Quit the simulation
- `R` – Reset the match (scores + ball)

## How It Works

| Component | Key Idea |
|-----------|----------|
| MLP agent | PyTorch Sequential net with two tanh hidden layers (32 + 32) fed by `[paddle_y, ball_x, ball_y, enemy_paddle_y, ball_dx, ball_dy]`. Trains every frame via cross-entropy against a “keep aligned with the ball” teacher; EMA loss is shown in the HUD. |
| RL agent  | Lightweight DQN that ingests the same signals from the right paddle’s perspective (including normalized velocities). Targets are computed on-policy each frame (reward + γ max Q′) with an ε-greedy policy and reward shaping for hits/misses/scores. |
| Environment | Deterministic Pong physics: rectangular paddles, reflecting ball, paddle-sensitive bounce angles, scoring with resets. Real-time HUD shows scores, MLP loss EMA, RL epsilon, reward EMA, and DQN loss EMA. |

Both agents update entirely on the fly—no pretraining, no datasets.

## Customization

Most of the knobs live at the top of `main.py`:

- `FPS`, `PADDLE_SPEED`, `BALL_SPEED`
- Reward magnitudes (`REWARD_*`)
- PyTorch optimizers, learning rates, ε schedules

The code is <400 lines, so it’s an easy playground if you want to bolt on replay buffers, curriculum resets, or different heads.

## Troubleshooting

- If the Pygame window fails to open on Windows, ensure you’re running from a local shell (not remote headless) and that GPU drivers are up to date.
- When running inside WSL/SSH without an X server, set `SDL_VIDEODRIVER=dummy` to disable rendering (you’ll still get console logs).

Enjoy watching the agents learn! Opening an issue/PR is welcome if you add new training modes or experiments.

