from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np

from ..core.game import ConnectNGame, GameState, player_view


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = True
    max_depth: int = 3
    time_limit_ms: int = 0
    use_transposition: bool = True
    top_k_ordering: int = 8
    network_guidance_weight: float = 0.15
    heuristic_weight: float = 1.0


@dataclass(frozen=True)
class SearchResult:
    action: int
    score: float
    depth: int
    nodes: int


def _terminal_score(state: GameState, root_player: int) -> float:
    if state.winner == 0:
        return 0.0
    return 10000.0 if state.winner == root_player else -10000.0


def _count_immediate_wins(game: ConnectNGame, state: GameState, player: int) -> list[int]:
    probe_state = GameState(
        board=state.board.copy(),
        current_player=player,
        done=False,
        winner=0,
        last_action=state.last_action,
        last_position=state.last_position,
        moves_played=state.moves_played,
    )
    winning_actions: list[int] = []
    for action in game.legal_actions(probe_state.board):
        _, result = game.step(probe_state, action)
        if not result.invalid and result.winner == player:
            winning_actions.append(action)
    return winning_actions


def _center_preference(game: ConnectNGame, board: np.ndarray, perspective_player: int) -> float:
    size = game.config.board_size
    center = (size - 1) / 2.0
    score = 0.0
    for z in range(size):
        for y in range(size):
            for x in range(size):
                piece = board[z, y, x]
                if piece == 0:
                    continue
                horizontal_distance = abs(y - center) + abs(x - center)
                vertical_distance = abs(z - center)
                contribution = (size - horizontal_distance) * 2.0 + (size - vertical_distance) * 0.1
                score += contribution if piece == perspective_player else -contribution
    return score


def _line_weights(win_length: int) -> dict[int, float]:
    weights: dict[int, float] = {}
    base = 1.0
    for count in range(1, win_length):
        weights[count] = base
        base *= 4.0
    return weights


def _network_leaf_score(
    model,
    state: GameState,
    game: ConnectNGame,
    device: str,
) -> float:
    if model is None:
        return 0.0
    legal_mask = game.legal_action_mask(state.board)
    if not np.any(legal_mask):
        return 0.0
    state_batch = np.expand_dims(player_view(state.board, state.current_player), axis=0)
    if hasattr(model, "predict"):
        q_values = np.asarray(model.predict(state_batch), dtype=np.float32).reshape(-1)
    else:
        q_values = np.asarray(model(state_batch), dtype=np.float32).reshape(-1)
    masked = np.where(legal_mask, q_values, -np.inf)
    return float(np.tanh(np.max(masked) / 5.0))


def evaluate_state_heuristic(
    state: GameState,
    game: ConnectNGame,
    perspective_player: int,
    *,
    model=None,
    device: str = "cpu",
    network_guidance_weight: float = 0.0,
    heuristic_weight: float = 1.0,
) -> float:
    if state.done:
        return _terminal_score(state, perspective_player)

    board = state.board
    opponent = -perspective_player
    weights = _line_weights(game.config.win_length)
    score = 0.0
    for line in game._winning_lines:
        current_count = 0
        opponent_count = 0
        for cell in line:
            piece = board[cell]
            if piece == perspective_player:
                current_count += 1
            elif piece == opponent:
                opponent_count += 1
        if current_count and opponent_count:
            continue
        if current_count:
            score += weights.get(current_count, 0.0)
        elif opponent_count:
            score -= weights.get(opponent_count, 0.0) * 1.15

    own_wins = len(_count_immediate_wins(game, state, perspective_player))
    opp_wins = len(_count_immediate_wins(game, state, opponent))
    score += 60.0 * own_wins
    score -= 70.0 * opp_wins
    if own_wins >= 2:
        score += 40.0
    if opp_wins >= 2:
        score -= 50.0

    score += _center_preference(game, board, perspective_player)
    score *= heuristic_weight

    if network_guidance_weight > 0.0 and model is not None:
        model_state = GameState(
            board=board,
            current_player=perspective_player,
            done=state.done,
            winner=state.winner,
            last_action=state.last_action,
            last_position=state.last_position,
            moves_played=state.moves_played,
        )
        score += network_guidance_weight * _network_leaf_score(model, model_state, game, device) * 50.0
    return float(score)


def _ordered_actions(
    state: GameState,
    game: ConnectNGame,
    model,
    device: str,
    search_config: SearchConfig,
) -> list[int]:
    legal_actions = game.legal_actions(state.board)
    if len(legal_actions) <= 1:
        return legal_actions

    winning_actions = set(_count_immediate_wins(game, state, state.current_player))
    opponent = -state.current_player
    opponent_threats = set(_count_immediate_wins(game, state, opponent))

    q_values_np: np.ndarray | None = None
    if model is not None:
        state_batch = np.expand_dims(player_view(state.board, state.current_player), axis=0)
        if hasattr(model, "predict"):
            q_values_np = np.asarray(model.predict(state_batch), dtype=np.float32).reshape(-1)
        else:
            q_values_np = np.asarray(model(state_batch), dtype=np.float32).reshape(-1)

    size = game.config.board_size
    center = (size - 1) / 2.0
    scored_actions: list[tuple[tuple[float, float, float, float], int]] = []
    for action in legal_actions:
        y, x = game.action_to_coords(action)
        center_bonus = -(abs(y - center) + abs(x - center))
        q_bonus = 0.0
        if q_values_np is not None:
            q_bonus = float(q_values_np[action])
        scored_actions.append(
            (
                (
                    1.0 if action in winning_actions else 0.0,
                    1.0 if action in opponent_threats else 0.0,
                    q_bonus if search_config.top_k_ordering > 0 else 0.0,
                    center_bonus,
                ),
                action,
            )
        )
    scored_actions.sort(key=lambda item: item[0], reverse=True)
    return [action for _, action in scored_actions]


def search_action(
    state: GameState,
    game: ConnectNGame,
    *,
    model=None,
    device: str = "cpu",
    search_config: SearchConfig | None = None,
    depth: int | None = None,
) -> SearchResult:
    search_config = search_config or SearchConfig()
    legal_actions = game.legal_actions(state.board)
    if not legal_actions:
        raise ValueError("No legal actions available for search.")
    if not search_config.enabled:
        return SearchResult(action=legal_actions[0], score=0.0, depth=0, nodes=0)

    max_depth = depth if depth is not None else search_config.max_depth
    max_depth = max(1, min(max_depth, search_config.max_depth))
    deadline = (
        time.perf_counter() + search_config.time_limit_ms / 1000.0
        if search_config.time_limit_ms > 0
        else None
    )
    transposition: dict[tuple[bytes, int, int, int], tuple[float, int | None]] = {}
    nodes = 0
    root_player = state.current_player

    def alpha_beta(
        node: GameState,
        depth_left: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, int | None]:
        nonlocal nodes
        nodes += 1
        if deadline is not None and time.perf_counter() >= deadline:
            return (
                evaluate_state_heuristic(
                    node,
                    game,
                    root_player,
                    model=model,
                    device=device,
                    network_guidance_weight=search_config.network_guidance_weight,
                    heuristic_weight=search_config.heuristic_weight,
                ),
                None,
            )
        if node.done:
            return _terminal_score(node, root_player), None
        if depth_left == 0:
            return (
                evaluate_state_heuristic(
                    node,
                    game,
                    root_player,
                    model=model,
                    device=device,
                    network_guidance_weight=search_config.network_guidance_weight,
                    heuristic_weight=search_config.heuristic_weight,
                ),
                None,
            )

        key = (node.board.tobytes(), node.current_player, depth_left, root_player)
        if search_config.use_transposition and key in transposition:
            return transposition[key]

        maximizing = node.current_player == root_player
        best_action: int | None = None
        ordered = _ordered_actions(node, game, model, device, search_config)

        if maximizing:
            value = -math.inf
            for action in ordered:
                next_state, _ = game.step(node, action)
                child_value, _ = alpha_beta(next_state, depth_left - 1, alpha, beta)
                if child_value > value:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = math.inf
            for action in ordered:
                next_state, _ = game.step(node, action)
                child_value, _ = alpha_beta(next_state, depth_left - 1, alpha, beta)
                if child_value < value:
                    value = child_value
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break

        result = (float(value), best_action)
        if search_config.use_transposition:
            transposition[key] = result
        return result

    score, action = alpha_beta(state, max_depth, -math.inf, math.inf)
    if action is None:
        ordered = _ordered_actions(state, game, model, device, search_config)
        action = ordered[0]
    return SearchResult(action=int(action), score=float(score), depth=max_depth, nodes=nodes)


def epsilon_search_action(
    state: GameState,
    game: ConnectNGame,
    *,
    model=None,
    device: str = "cpu",
    epsilon: float = 0.0,
    rng: random.Random | None = None,
    search_config: SearchConfig | None = None,
    depth: int | None = None,
) -> SearchResult:
    rng = rng or random
    legal_actions = game.legal_actions(state.board)
    if rng.random() < epsilon:
        action = int(rng.choice(legal_actions))
        return SearchResult(action=action, score=0.0, depth=0, nodes=0)
    return search_action(
        state,
        game,
        model=model,
        device=device,
        search_config=search_config,
        depth=depth,
    )
