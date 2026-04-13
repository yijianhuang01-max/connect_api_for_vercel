from __future__ import annotations

import random
from typing import Callable

import numpy as np

from ..core.game import ConnectNGame, GameState, player_view
from .search import SearchConfig, epsilon_search_action, search_action


def _legal_indices(legal_mask: np.ndarray) -> np.ndarray:
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise ValueError("No legal actions available.")
    return legal_indices


def random_action(legal_mask: np.ndarray, rng: random.Random | None = None) -> int:
    rng = rng or random
    legal_indices = _legal_indices(legal_mask)
    return int(rng.choice(legal_indices.tolist()))


def _predict_q_values(model, state: GameState) -> np.ndarray:
    state_batch = np.expand_dims(player_view(state.board, state.current_player), axis=0)
    if hasattr(model, "predict"):
        output = model.predict(state_batch)
    else:
        output = model(state_batch)
    return np.asarray(output, dtype=np.float32).reshape(-1)


def masked_argmax(q_values: np.ndarray, legal_mask: np.ndarray) -> int:
    masked = np.where(legal_mask, q_values, -np.inf)
    return int(np.argmax(masked))


def greedy_action(model, state: GameState, game: ConnectNGame, device: str) -> int:
    legal_mask = game.legal_action_mask(state.board)
    q_values = _predict_q_values(model, state)
    return masked_argmax(q_values, legal_mask)


def epsilon_greedy_action(
    model,
    state: GameState,
    game: ConnectNGame,
    device: str,
    epsilon: float,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random
    legal_mask = game.legal_action_mask(state.board)
    if rng.random() < epsilon:
        return random_action(legal_mask, rng=rng)
    return greedy_action(model, state, game, device)


def tactical_action(state: GameState, game: ConnectNGame, rng: random.Random | None = None) -> int:
    rng = rng or random
    legal_actions = game.legal_actions(state.board)
    # Immediate win.
    for action in legal_actions:
        next_state, result = game.step(state, action)
        if not result.invalid and result.winner == state.current_player:
            return action
        del next_state
    # Block opponent win.
    opponent = -state.current_player
    opponent_state = GameState(
        board=state.board.copy(),
        current_player=opponent,
        done=state.done,
        winner=state.winner,
        last_action=state.last_action,
        last_position=state.last_position,
        moves_played=state.moves_played,
    )
    threats: list[int] = []
    for action in legal_actions:
        next_state, result = game.step(opponent_state, action)
        if not result.invalid and result.winner == opponent:
            threats.append(action)
        del next_state
    if threats:
        return int(threats[0])
    return random_action(game.legal_action_mask(state.board), rng=rng)


def choose_action(
    policy_name: str,
    state: GameState,
    game: ConnectNGame,
    rng: random.Random | None = None,
    model=None,
    device: str = "cpu",
    epsilon: float = 0.0,
    search_config: SearchConfig | None = None,
    search_depth: int | None = None,
) -> int:
    policy_name = policy_name.lower()
    if policy_name == "random":
        return random_action(game.legal_action_mask(state.board), rng=rng)
    if policy_name == "tactical":
        return tactical_action(state, game, rng=rng)
    if policy_name == "search":
        return search_action(
            state,
            game,
            model=model,
            device=device,
            search_config=search_config,
            depth=search_depth,
        ).action
    if policy_name in {"greedy", "model"}:
        if model is None:
            raise ValueError("Model policy requested without a model.")
        return greedy_action(model, state, game, device)
    if policy_name == "epsilon_greedy":
        if model is None:
            raise ValueError("Epsilon-greedy policy requested without a model.")
        return epsilon_greedy_action(model, state, game, device, epsilon=epsilon, rng=rng)
    if policy_name == "epsilon_search":
        return epsilon_search_action(
            state,
            game,
            model=model,
            device=device,
            epsilon=epsilon,
            rng=rng,
            search_config=search_config,
            depth=search_depth,
        ).action
    raise ValueError(f"Unknown policy: {policy_name}")
