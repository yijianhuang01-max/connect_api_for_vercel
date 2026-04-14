from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT
for candidate in [SERVICE_ROOT, *SERVICE_ROOT.parents]:
    if (candidate / "connect_four").exists():
        PROJECT_ROOT = candidate
        break
CONNECT_ROOT = PROJECT_ROOT / "connect_four"
FORCE_BUNDLED_RUNTIME = os.getenv("CONNECT_API_USE_BUNDLED_RUNTIME") == "1"
if CONNECT_ROOT.exists() and not FORCE_BUNDLED_RUNTIME and str(CONNECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONNECT_ROOT))

try:
    if FORCE_BUNDLED_RUNTIME:
        raise ImportError("Bundled runtime forced by environment.")
    import torch
    import constants  # type: ignore  # noqa: E402
    from agents import ModelAgent, list_model_choices  # type: ignore  # noqa: E402
    from core.config import GameConfig  # type: ignore  # noqa: E402
    from core.game import ConnectNGame, GameState  # type: ignore  # noqa: E402
    from rl.policies import greedy_action, tactical_action  # type: ignore  # noqa: E402
    from rl.search import search_action  # type: ignore  # noqa: E402

    USING_LOCAL_CONNECT_FOUR = True
except ImportError:
    from runtime.connect_four_runtime import constants  # type: ignore  # noqa: E402
    from runtime.connect_four_runtime.core.config import GameConfig  # type: ignore  # noqa: E402
    from runtime.connect_four_runtime.core.game import ConnectNGame, GameState  # type: ignore  # noqa: E402
    from runtime.connect_four_runtime.rl.checkpoints import read_leaderboard  # type: ignore  # noqa: E402
    from runtime.connect_four_runtime.rl.onnx_model import ONNXQModel  # type: ignore  # noqa: E402
    from runtime.connect_four_runtime.rl.policies import greedy_action, tactical_action  # type: ignore  # noqa: E402
    from runtime.connect_four_runtime.rl.search import SearchConfig, search_action  # type: ignore  # noqa: E402

    USING_LOCAL_CONNECT_FOUR = False


GAME_CONFIG = GameConfig(board_size=4, win_length=4)
GAME = ConnectNGame(GAME_CONFIG)


def parse_player(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"red", "r", "1"}:
            return 1
        if normalized in {"blue", "b", "-1"}:
            return -1
    if value in {1, -1}:
        return int(value)
    raise ValueError("current_player must be 1/-1 or red/blue.")


def empty_board() -> list[list[list[int]]]:
    return GAME.new_board().tolist()


def board_size() -> int:
    return GAME_CONFIG.board_size


def validate_board_payload(board: Any) -> np.ndarray:
    array = np.asarray(board, dtype=np.int8)
    expected_shape = GAME_CONFIG.board_shape
    if array.shape != expected_shape:
        raise ValueError(f"board must have shape {expected_shape}.")
    if not np.isin(array, [constants.EMPTY, constants.RED, constants.BLUE]).all():
        raise ValueError("board contains values outside {-1, 0, 1}.")

    for y in range(GAME_CONFIG.board_size):
        for x in range(GAME_CONFIG.board_size):
            seen_empty = False
            for z in range(GAME_CONFIG.board_size):
                if array[z, y, x] == constants.EMPTY:
                    seen_empty = True
                elif seen_empty:
                    raise ValueError("board violates gravity in at least one column.")
    return array


def action_to_response(action: int) -> dict[str, int]:
    y, x = GAME.action_to_coords(action)
    return {"action": int(action), "y": int(y), "x": int(x)}


def available_models() -> list[dict[str, Any]]:
    if USING_LOCAL_CONNECT_FOUR:
        models = []
        for choice in list_model_choices(board_size=GAME_CONFIG.board_size):
            models.append(
                {
                    "ordinal": choice.ordinal,
                    "filename": choice.filename,
                    "path": str(choice.path),
                    "score": choice.score,
                    "stage": choice.stage,
                    "stage_step": choice.stage_step,
                }
            )
        return models

    leaderboard_map = {
        entry.ordinal: entry for entry in read_leaderboard(constants.MODEL_DIR, GAME_CONFIG.board_size)
    }
    models = []
    pattern = f"connect_{GAME_CONFIG.board_size}_*_dqn.pt"
    discovered: list[tuple[int, Path]] = []
    for path in constants.MODEL_DIR.glob(pattern):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        discovered.append((int(parts[2]), path))
    for ordinal, path in sorted(discovered, key=lambda item: item[0], reverse=True):
        entry = leaderboard_map.get(ordinal)
        models.append(
            {
                "ordinal": ordinal,
                "filename": path.name,
                "path": str(path),
                "score": entry.score if entry is not None else None,
                "stage": entry.stage if entry is not None else None,
                "stage_step": entry.stage_step if entry is not None else None,
            }
        )
    return models


def resolve_model_record(checkpoint_ordinal: int | None) -> dict[str, Any] | None:
    models = available_models()
    if checkpoint_ordinal is None:
        return models[0] if models else None
    for model in models:
        if model["ordinal"] == checkpoint_ordinal:
            return model
    raise ValueError(f"Checkpoint ordinal {checkpoint_ordinal} is not available.")


@lru_cache(maxsize=8)
def load_agent(checkpoint_path: str | None) -> ModelAgent:
    if USING_LOCAL_CONNECT_FOUR:
        return ModelAgent(checkpoint_path=Path(checkpoint_path) if checkpoint_path else None)

    class BundledModelAgent:
        def __init__(self, path: str | None):
            self.config = GameConfig()
            self.game = ConnectNGame(self.config)
            self.device = "cpu"
            self.search_config = SearchConfig(
                enabled=constants.SEARCH_ENABLED,
                max_depth=constants.SEARCH_MAX_DEPTH,
                time_limit_ms=constants.SEARCH_TIME_LIMIT_MS,
                use_transposition=constants.SEARCH_USE_TRANSPOSITION,
                top_k_ordering=constants.SEARCH_TOP_K_ORDERING,
                network_guidance_weight=constants.SEARCH_NETWORK_GUIDANCE_WEIGHT,
                heuristic_weight=constants.SEARCH_HEURISTIC_WEIGHT,
            )
            self.checkpoint_path = Path(path) if path else None
            self.onnx_path = self.checkpoint_path.with_suffix(".onnx") if self.checkpoint_path else None
            self.available = self.onnx_path is not None and self.onnx_path.exists()
            self.model = ONNXQModel(self.onnx_path) if self.available else None

    return BundledModelAgent(checkpoint_path)


def select_move(board: np.ndarray, current_player: int, checkpoint_ordinal: int | None) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    model_record = resolve_model_record(checkpoint_ordinal)
    checkpoint_path = model_record["path"] if model_record is not None else None
    agent = load_agent(checkpoint_path)
    state = GameState(board=board.copy(), current_player=current_player)

    if constants.SEARCH_ENABLED:
        result = search_action(
            state,
            agent.game,
            model=agent.model if agent.available else None,
            device=str(agent.device),
            search_config=agent.search_config,
            depth=constants.INFERENCE_SEARCH_DEPTH,
        )
        return result.action, {
            "depth": result.depth,
            "nodes": result.nodes,
            "score": result.score,
        }, model_record

    if agent.available:
        action = greedy_action(agent.model, state, agent.game, str(agent.device))
        return action, {"depth": 0, "nodes": 0, "score": None}, model_record

    action = tactical_action(state, agent.game)
    return action, {"depth": 0, "nodes": 0, "score": None}, model_record


def apply_action(board: np.ndarray, current_player: int, action: int) -> tuple[GameState, Any]:
    state = GameState(board=board.copy(), current_player=current_player)
    return GAME.step(state, int(action))
