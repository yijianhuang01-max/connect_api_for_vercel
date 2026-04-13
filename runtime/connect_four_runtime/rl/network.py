from __future__ import annotations

import torch
import torch.nn as nn

from ..core.config import GameConfig


class ConvQNetwork(nn.Module):
    def __init__(self, game_config: GameConfig):
        super().__init__()
        size = game_config.board_size
        hidden = max(128, size * size * 16)
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * size * size * size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, game_config.action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        return self.net(x)


class MLPQNetwork(nn.Module):
    def __init__(self, game_config: GameConfig):
        super().__init__()
        features = game_config.board_size**3
        hidden = max(128, features * 4)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, game_config.action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.net(x)


def build_q_network(game_config: GameConfig, architecture: str = "conv") -> nn.Module:
    architecture = architecture.lower()
    if architecture == "conv":
        return ConvQNetwork(game_config)
    if architecture == "mlp":
        return MLPQNetwork(game_config)
    raise ValueError(f"Unsupported network architecture: {architecture}")


class DQN(nn.Module):
    def __init__(self, game_config: GameConfig | None = None, architecture: str = "conv"):
        from ..core.config import GameConfig as _GameConfig

        super().__init__()
        self.game_config = game_config or _GameConfig()
        self.model = build_q_network(self.game_config, architecture=architecture)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
