import math
import random
from typing import Optional, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# --- Game constants ---------------------------------------------------------

WIDTH, HEIGHT = 960, 540
PADDLE_WIDTH, PADDLE_HEIGHT = 12, 88
BALL_SIZE = 14
PADDLE_SPEED = 6
BALL_SPEED = 6
FPS = 60

ACTIONS = (-1, 0, 1)  # up, stay, down

COLOR_BG = (8, 10, 18)
COLOR_LINES = (200, 200, 200)
COLOR_LEFT = (80, 196, 255)
COLOR_RIGHT = (255, 102, 134)
COLOR_BALL = (253, 255, 150)

# Reward shaping for the right (RL) paddle
REWARD_HIT = 0.35
REWARD_RIVAL_HIT = -0.10
REWARD_SCORE = 1.0
REWARD_CONCEDE = -1.0

TEACHER_TOLERANCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


class Paddle:
    def __init__(self, x: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = (HEIGHT - PADDLE_HEIGHT) / 2
        self.color = color

    def move(self, delta: float) -> None:
        self.y = clamp(self.y + delta, 0, HEIGHT - PADDLE_HEIGHT)

    def reset(self) -> None:
        self.y = (HEIGHT - PADDLE_HEIGHT) / 2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), PADDLE_WIDTH, PADDLE_HEIGHT)

    @property
    def center_y(self) -> float:
        return self.y + PADDLE_HEIGHT / 2


class Ball:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.reset()

    def reset(self, direction: Optional[int] = None) -> None:
        self.x = WIDTH / 2 - BALL_SIZE / 2
        self.y = HEIGHT / 2 - BALL_SIZE / 2
        dir_x = random.choice([-1, 1]) if direction is None else int(math.copysign(1, direction))
        angle = random.uniform(-0.4, 0.4)
        speed = BALL_SPEED
        self.vx = dir_x * speed * math.cos(angle)
        self.vy = speed * math.sin(angle)

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy

        if self.y <= 0:
            self.y = 0
            self.vy *= -1
        elif self.y + BALL_SIZE >= HEIGHT:
            self.y = HEIGHT - BALL_SIZE
            self.vy *= -1

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), BALL_SIZE, BALL_SIZE)

    @property
    def dir_components(self) -> Tuple[float, float]:
        max_val = max(BALL_SPEED, abs(self.vx), abs(self.vy), 1e-6)
        return (self.vx / max_val, self.vy / max_val)


def handle_paddle_collision(ball: Ball, paddle: Paddle, is_left: bool) -> bool:
    if not ball.rect.colliderect(paddle.rect):
        return False

    if is_left:
        ball.x = paddle.x + PADDLE_WIDTH
    else:
        ball.x = paddle.x - BALL_SIZE

    ball.vx *= -1
    offset = (ball.y + BALL_SIZE / 2) - paddle.center_y
    norm = offset / (PADDLE_HEIGHT / 2)
    ball.vy = clamp(norm, -1.0, 1.0) * BALL_SPEED
    return True


class MLPPaddleAgent:
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int] = (32, 32),
                 lr: float = 1e-3, epsilon: float = 0.02):
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, len(ACTIONS)),
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.loss_ema: Optional[float] = None

    @staticmethod
    def _tensorize(features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    def select_action(self, features: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        self.net.eval()
        x = self._tensorize(features)
        with torch.no_grad():
            logits = self.net(x)
        return int(torch.argmax(logits, dim=1).item())

    def update(self, features: np.ndarray, target_idx: int) -> float:
        self.net.train()
        x = self._tensorize(features)
        target = torch.tensor([target_idx], dtype=torch.long, device=DEVICE)
        logits = self.net(x)
        loss = self.criterion(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            self.loss_ema = 0.98 * self.loss_ema + 0.02 * loss_value
        return loss_value


class RLPaddleAgent:
    def __init__(self, input_dim: int = 6, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.95, epsilon: float = 0.3,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.999):
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_ema = 0.0
        self.loss_ema: Optional[float] = None

    @staticmethod
    def _tensorize(state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        self.q_net.eval()
        state_t = self._tensorize(state)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return int(torch.argmax(q_vals, dim=1).item())

    def update(self, state: np.ndarray, action_idx: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        self.q_net.train()
        state_t = self._tensorize(state)
        next_state_t = self._tensorize(next_state)

        q_values = self.q_net(state_t)
        q_selected = q_values[0, action_idx].unsqueeze(0)

        with torch.no_grad():
            next_q = self.q_net(next_state_t).max(dim=1).values
            target_value = reward if done else reward + self.gamma * float(next_q.item())
        target = torch.tensor([target_value], dtype=torch.float32, device=DEVICE)

        loss = self.criterion(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_value = float(loss.item())
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            self.loss_ema = 0.98 * self.loss_ema + 0.02 * loss_value
        self.reward_ema = 0.97 * self.reward_ema + 0.03 * reward
        decay = 0.5 if done else self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon * decay)


def build_left_features(left: Paddle, right: Paddle, ball: Ball) -> np.ndarray:
    dx, dy = ball.dir_components
    return np.array([
        left.y / (HEIGHT - PADDLE_HEIGHT),
        ball.x / WIDTH,
        ball.y / HEIGHT,
        right.y / (HEIGHT - PADDLE_HEIGHT),
        dx,
        dy,
    ], dtype=np.float32)


def build_right_features(right: Paddle, left: Paddle, ball: Ball) -> np.ndarray:
    dx, dy = ball.dir_components
    return np.array([
        right.y / (HEIGHT - PADDLE_HEIGHT),
        ball.x / WIDTH,
        ball.y / HEIGHT,
        left.y / (HEIGHT - PADDLE_HEIGHT),
        dx,
        dy,
    ], dtype=np.float32)


def teacher_action(ball: Ball, paddle: Paddle, tolerance: float = TEACHER_TOLERANCE) -> int:
    ball_center = ball.y + BALL_SIZE / 2
    paddle_center = paddle.center_y
    if ball_center < paddle_center - tolerance:
        return 0  # move up
    if ball_center > paddle_center + tolerance:
        return 2  # move down
    return 1  # stay


def draw_center_line(surface: pygame.Surface) -> None:
    segment_height = 12
    gap = 12
    x = WIDTH // 2 - 1
    for y in range(0, HEIGHT, segment_height + gap):
        pygame.draw.rect(surface, COLOR_LINES, (x, y, 2, segment_height))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("MLP vs RL Pong")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Consolas", 32)
    font_small = pygame.font.SysFont("Consolas", 20)

    left_paddle = Paddle(40, COLOR_LEFT)
    right_paddle = Paddle(WIDTH - 40 - PADDLE_WIDTH, COLOR_RIGHT)
    ball = Ball()
    mlp_agent = MLPPaddleAgent(input_dim=6)
    rl_agent = RLPaddleAgent()

    score_left = 0
    score_right = 0
    running = True

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    score_left = 0
                    score_right = 0
                    ball.reset()
                    left_paddle.reset()
                    right_paddle.reset()

        left_features = build_left_features(left_paddle, right_paddle, ball)
        target_idx = teacher_action(ball, left_paddle)
        left_action_idx = mlp_agent.select_action(left_features)
        left_paddle.move(ACTIONS[left_action_idx] * PADDLE_SPEED)

        right_features = build_right_features(right_paddle, left_paddle, ball)
        right_action_idx = rl_agent.select_action(right_features)
        right_paddle.move(ACTIONS[right_action_idx] * PADDLE_SPEED)

        ball.update()

        reward_right = 0.0
        if handle_paddle_collision(ball, left_paddle, True):
            reward_right += REWARD_RIVAL_HIT
        if handle_paddle_collision(ball, right_paddle, False):
            reward_right += REWARD_HIT

        point_scored = False
        if ball.x + BALL_SIZE < 0:
            score_right += 1
            reward_right += REWARD_SCORE
            ball.reset(direction=1)
            left_paddle.reset()
            right_paddle.reset()
            point_scored = True
        elif ball.x > WIDTH:
            score_left += 1
            reward_right += REWARD_CONCEDE
            ball.reset(direction=-1)
            left_paddle.reset()
            right_paddle.reset()
            point_scored = True

        next_right_features = build_right_features(right_paddle, left_paddle, ball)
        rl_agent.update(right_features, right_action_idx, reward_right, next_right_features, point_scored)
        mlp_agent.update(left_features, target_idx)

        screen.fill(COLOR_BG)
        draw_center_line(screen)
        pygame.draw.rect(screen, COLOR_LEFT, left_paddle.rect)
        pygame.draw.rect(screen, COLOR_RIGHT, right_paddle.rect)
        pygame.draw.rect(screen, COLOR_BALL, ball.rect)

        score_surface = font_big.render(f"{score_left} : {score_right}", True, COLOR_LINES)
        screen.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, 16))

        mlp_loss = mlp_agent.loss_ema if mlp_agent.loss_ema is not None else 0.0
        rl_loss = rl_agent.loss_ema if rl_agent.loss_ema is not None else 0.0
        hud_lines = [
            f"Left (MLP) loss EMA: {mlp_loss:.3f}",
            f"Right (DQN) loss EMA: {rl_loss:.3f}",
            f"Right epsilon: {rl_agent.epsilon:.2f}  reward EMA: {rl_agent.reward_ema:.3f}",
            "Esc: quit    R: reset scores",
        ]
        for i, text in enumerate(hud_lines):
            surface = font_small.render(text, True, COLOR_LINES)
            screen.blit(surface, (20, HEIGHT - (len(hud_lines) - i) * 24))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

