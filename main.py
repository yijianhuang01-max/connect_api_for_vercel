from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from .adapter import (
        action_to_response,
        apply_action,
        available_models,
        board_size,
        empty_board,
        parse_player,
        select_move,
        validate_board_payload,
    )
except ImportError:
    from adapter import (
        action_to_response,
        apply_action,
        available_models,
        board_size,
        empty_board,
        parse_player,
        select_move,
        validate_board_payload,
    )


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOW_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(title="Connect-N Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "board_size": board_size(),
        "available_models": len(available_models()),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return {"board_size": board_size(), "models": available_models()}


@app.post("/new-game")
def new_game(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    checkpoint_ordinal = payload.get("checkpoint_ordinal")
    human_player = payload.get("human_player", 1)
    model = None
    try:
        human_player = parse_player(human_player)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if checkpoint_ordinal is not None:
        try:
            _, _, model = select_move(validate_board_payload(empty_board()), 1, int(checkpoint_ordinal))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        models_payload = available_models()
        model = models_payload[0] if models_payload else None
    board = validate_board_payload(empty_board())
    current_player = 1
    ai_move = None
    search = None
    done = False
    winner = 0

    if human_player == -1:
        try:
            ai_action, search, model = select_move(board, current_player, checkpoint_ordinal)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ai_state, ai_result = apply_action(board, current_player, ai_action)
        board = ai_state.board
        current_player = ai_state.current_player
        done = ai_state.done
        winner = ai_state.winner
        ai_move = action_to_response(ai_action)
        if ai_result.position is not None:
            ai_move["z"] = int(ai_result.position[0])

    return {
        "board": board.tolist(),
        "current_player": current_player,
        "done": done,
        "winner": winner,
        "model": model,
        "human_player": human_player,
        "ai_move": ai_move,
        "search": search,
    }


@app.post("/move")
def move(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        board = validate_board_payload(payload["board"])
        current_player = parse_player(payload["current_player"])
        checkpoint_ordinal = payload.get("checkpoint_ordinal")
        action = payload.get("action")
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if action is None:
        try:
            chosen_action, search_summary, model = select_move(board, current_player, checkpoint_ordinal)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            **action_to_response(chosen_action),
            "board": board.tolist(),
            "current_player": current_player,
            "done": False,
            "winner": 0,
            "model": model,
            "search": search_summary,
        }

    try:
        human_state, human_result = apply_action(board, current_player, int(action))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    human_move = action_to_response(int(action))
    response: dict[str, Any] = {
        "board": human_state.board.tolist(),
        "current_player": human_state.current_player,
        "done": human_state.done,
        "winner": human_state.winner,
        "invalid": human_result.invalid,
        "human_move": human_move,
        "ai_move": None,
        "model": None,
        "search": None,
    }
    if human_result.position is not None:
        response["human_move"]["z"] = int(human_result.position[0])

    if human_result.invalid or human_state.done:
        return response

    try:
        ai_action, search_summary, model = select_move(
            human_state.board,
            human_state.current_player,
            checkpoint_ordinal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ai_state, ai_result = apply_action(human_state.board, human_state.current_player, ai_action)
    ai_move = action_to_response(ai_action)
    if ai_result.position is not None:
        ai_move["z"] = int(ai_result.position[0])

    response.update(
        {
            "board": ai_state.board.tolist(),
            "current_player": ai_state.current_player,
            "done": ai_state.done,
            "winner": ai_state.winner,
            "ai_move": ai_move,
            "model": model,
            "search": search_summary,
        }
    )
    return response
