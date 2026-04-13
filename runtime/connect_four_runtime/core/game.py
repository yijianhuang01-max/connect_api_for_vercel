from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable

import numpy as np

from .config import GameConfig, StepResult


@dataclass(frozen=True)
class GameState:
    board: np.ndarray
    current_player: int
    done: bool = False
    winner: int = 0
    last_action: int | None = None
    last_position: tuple[int, int, int] | None = None
    moves_played: int = 0

    def clone(self) -> "GameState":
        return replace(self, board=self.board.copy())


def player_view(board: np.ndarray, player: int) -> np.ndarray:
    return (board * player).astype(np.float32, copy=False)


def _canonical_directions() -> list[tuple[int, int, int]]:
    directions: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                for component in (dz, dy, dx):
                    if component == 0:
                        continue
                    if component > 0:
                        directions.append((dz, dy, dx))
                    break
    return directions


def _in_bounds(coord: tuple[int, int, int], board_size: int) -> bool:
    return all(0 <= value < board_size for value in coord)


@lru_cache(maxsize=None)
def _cached_winning_lines(
    board_size: int, win_length: int
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    lines: set[tuple[tuple[int, int, int], ...]] = set()
    for z in range(board_size):
        for y in range(board_size):
            for x in range(board_size):
                start = (z, y, x)
                for dz, dy, dx in _canonical_directions():
                    end = (
                        z + (win_length - 1) * dz,
                        y + (win_length - 1) * dy,
                        x + (win_length - 1) * dx,
                    )
                    if not _in_bounds(end, board_size):
                        continue
                    line = tuple(
                        (z + step * dz, y + step * dy, x + step * dx)
                        for step in range(win_length)
                    )
                    lines.add(line)
    return tuple(sorted(lines))


class ConnectNGame:
    def __init__(self, config: GameConfig):
        config.validate()
        self.config = config
        self._winning_lines = _cached_winning_lines(config.board_size, config.win_length)
        self._position_to_lines = self._index_lines(self._winning_lines)

    def _index_lines(
        self, lines: Iterable[tuple[tuple[int, int, int], ...]]
    ) -> dict[tuple[int, int, int], tuple[tuple[tuple[int, int, int], ...], ...]]:
        mapping: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], ...]]] = {}
        for line in lines:
            for cell in line:
                mapping.setdefault(cell, []).append(line)
        return {cell: tuple(cell_lines) for cell, cell_lines in mapping.items()}

    def new_board(self) -> np.ndarray:
        return np.zeros(self.config.board_shape, dtype=np.int8)

    def initial_state(self) -> GameState:
        return GameState(board=self.new_board(), current_player=self.config.red_value)

    def action_to_coords(self, action: int) -> tuple[int, int]:
        if not 0 <= action < self.config.action_size:
            raise ValueError(f"Action {action} is out of range.")
        return divmod(action, self.config.board_size)

    def coords_to_action(self, y: int, x: int) -> int:
        return y * self.config.board_size + x

    def column_height(self, board: np.ndarray, y: int, x: int) -> int:
        filled = np.flatnonzero(board[:, y, x] == self.config.empty_value)
        return int(filled[0]) if filled.size else self.config.board_size

    def find_drop_z(self, board: np.ndarray, y: int, x: int) -> int | None:
        for z in range(self.config.board_size):
            if board[z, y, x] == self.config.empty_value:
                return z
        return None

    def legal_action_mask(self, board: np.ndarray) -> np.ndarray:
        top_layer = board[self.config.board_size - 1]
        return (top_layer == self.config.empty_value).reshape(-1)

    def legal_actions(self, board: np.ndarray) -> list[int]:
        mask = self.legal_action_mask(board)
        return np.flatnonzero(mask).tolist()

    def check_win_from_position(
        self, board: np.ndarray, player: int, position: tuple[int, int, int]
    ) -> bool:
        for line in self._position_to_lines.get(position, ()):
            if all(board[cell] == player for cell in line):
                return True
        return False

    def is_draw(self, board: np.ndarray) -> bool:
        return not np.any(board == self.config.empty_value)

    def step(self, state: GameState, action: int) -> tuple[GameState, StepResult]:
        board = state.board.copy()
        legal_mask = self.legal_action_mask(board)
        if state.done:
            step_result = StepResult(
                reward=0.0,
                done=True,
                invalid=True,
                winner=state.winner,
                legal_action_mask=tuple(bool(x) for x in legal_mask),
                action=action,
                player=state.current_player,
                position=state.last_position,
            )
            return state.clone(), step_result

        if action < 0 or action >= self.config.action_size or not legal_mask[action]:
            winner = -state.current_player
            step_result = StepResult(
                reward=self.config.rewards.invalid,
                done=True,
                invalid=True,
                winner=winner,
                legal_action_mask=tuple(bool(x) for x in legal_mask),
                action=action,
                player=state.current_player,
                position=None,
            )
            return (
                GameState(
                    board=board,
                    current_player=state.current_player,
                    done=True,
                    winner=winner,
                    last_action=state.last_action,
                    last_position=state.last_position,
                    moves_played=state.moves_played,
                ),
                step_result,
            )

        y, x = self.action_to_coords(action)
        z = self.find_drop_z(board, y, x)
        if z is None:
            winner = -state.current_player
            step_result = StepResult(
                reward=self.config.rewards.invalid,
                done=True,
                invalid=True,
                winner=winner,
                legal_action_mask=tuple(bool(x) for x in legal_mask),
                action=action,
                player=state.current_player,
                position=None,
            )
            return (
                GameState(
                    board=board,
                    current_player=state.current_player,
                    done=True,
                    winner=winner,
                    last_action=state.last_action,
                    last_position=state.last_position,
                    moves_played=state.moves_played,
                ),
                step_result,
            )

        board[z, y, x] = state.current_player
        position = (z, y, x)
        won = self.check_win_from_position(board, state.current_player, position)
        draw = self.is_draw(board)
        done = won or draw
        winner = state.current_player if won else self.config.empty_value
        reward = (
            self.config.rewards.win
            if won
            else self.config.rewards.draw
            if draw
            else self.config.rewards.step
        )
        next_player = state.current_player if done else -state.current_player
        next_state = GameState(
            board=board,
            current_player=next_player,
            done=done,
            winner=winner,
            last_action=action,
            last_position=position,
            moves_played=state.moves_played + 1,
        )
        next_mask = self.legal_action_mask(board)
        step_result = StepResult(
            reward=reward,
            done=done,
            invalid=False,
            winner=winner,
            legal_action_mask=tuple(bool(x) for x in next_mask),
            action=action,
            player=state.current_player,
            position=position,
        )
        return next_state, step_result
