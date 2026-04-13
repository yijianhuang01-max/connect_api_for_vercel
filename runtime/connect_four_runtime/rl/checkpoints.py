from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpponentMatchSummary:
    ordinal: int
    filename: str
    score: float
    wins: int
    draws: int
    losses: int
    actual_score: float
    expected_score: float
    candidate_score_before: float
    candidate_score_after: float


@dataclass(frozen=True)
class LeaderboardEntry:
    ordinal: int
    filename: str
    score: float
    stage: str | None = None
    stage_step: int | None = None
    parent_ordinal: int | None = None
    opponents: tuple[OpponentMatchSummary, ...] = field(default_factory=tuple)
    match_games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0


def checkpoint_name(board_size: int, ordinal: int) -> str:
    return f"connect_{board_size}_{ordinal}_dqn.pt"


def checkpoint_path(model_dir: Path, board_size: int, ordinal: int) -> Path:
    return model_dir / checkpoint_name(board_size, ordinal)


def leaderboard_path(model_dir: Path, board_size: int) -> Path:
    return model_dir / f"connect_{board_size}_leaderboard.json"


def _checkpoint_regex(board_size: int) -> re.Pattern[str]:
    return re.compile(rf"^connect_{board_size}_(\d+)_dqn\.pt$")


def list_checkpoint_paths(model_dir: Path, board_size: int) -> list[tuple[int, Path]]:
    if not model_dir.exists():
        return []
    matcher = _checkpoint_regex(board_size)
    checkpoints: list[tuple[int, Path]] = []
    for path in model_dir.glob(f"connect_{board_size}_*_dqn.pt"):
        match = matcher.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def latest_checkpoint_path(model_dir: Path, board_size: int) -> Path | None:
    checkpoints = list_checkpoint_paths(model_dir, board_size)
    return checkpoints[-1][1] if checkpoints else None


def latest_checkpoint_ordinal(model_dir: Path, board_size: int) -> int:
    checkpoints = list_checkpoint_paths(model_dir, board_size)
    return checkpoints[-1][0] if checkpoints else 0


def _opponent_summary_from_dict(data: dict[str, Any]) -> OpponentMatchSummary:
    return OpponentMatchSummary(
        ordinal=int(data["ordinal"]),
        filename=str(data["filename"]),
        score=float(data["score"]),
        wins=int(data["wins"]),
        draws=int(data["draws"]),
        losses=int(data["losses"]),
        actual_score=float(data["actual_score"]),
        expected_score=float(data["expected_score"]),
        candidate_score_before=float(data["candidate_score_before"]),
        candidate_score_after=float(data["candidate_score_after"]),
    )


def _entry_from_dict(data: dict[str, Any]) -> LeaderboardEntry:
    return LeaderboardEntry(
        ordinal=int(data["ordinal"]),
        filename=str(data["filename"]),
        score=float(data["score"]),
        stage=data.get("stage"),
        stage_step=data.get("stage_step"),
        parent_ordinal=data.get("parent_ordinal"),
        opponents=tuple(
            _opponent_summary_from_dict(item) for item in data.get("opponents", [])
        ),
        match_games=int(data.get("match_games", 0)),
        wins=int(data.get("wins", 0)),
        draws=int(data.get("draws", 0)),
        losses=int(data.get("losses", 0)),
    )


def read_leaderboard(model_dir: Path, board_size: int) -> list[LeaderboardEntry]:
    path = leaderboard_path(model_dir, board_size)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [_entry_from_dict(item) for item in payload]


def write_leaderboard(model_dir: Path, board_size: int, entries: list[LeaderboardEntry]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = leaderboard_path(model_dir, board_size)
    payload = [asdict(entry) for entry in sorted(entries, key=lambda item: item.ordinal)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _entry_sort_key(entry: LeaderboardEntry) -> tuple[float, int]:
    return (entry.score, entry.ordinal)


def select_top_opponents(
    entries: list[LeaderboardEntry],
    candidate_ordinal: int,
    opponent_number: int,
) -> list[LeaderboardEntry]:
    older_entries = [entry for entry in entries if entry.ordinal < candidate_ordinal]
    older_entries.sort(key=_entry_sort_key, reverse=True)
    return older_entries[:opponent_number]


def _payload_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return {
        "score": payload.get("checkpoint_score"),
        "stage": payload.get("stage"),
        "stage_step": payload.get("stage_step"),
        "parent_ordinal": payload.get("parent_checkpoint_ordinal"),
    }


def bootstrap_leaderboard(
    model_dir: Path,
    board_size: int,
    base_score: float,
) -> list[LeaderboardEntry]:
    existing_entries = {entry.ordinal: entry for entry in read_leaderboard(model_dir, board_size)}
    changed = False
    for ordinal, path in list_checkpoint_paths(model_dir, board_size):
        if ordinal in existing_entries:
            continue
        metadata = _payload_metadata(path)
        score = metadata.get("score")
        if score is None:
            score = base_score
        entry = LeaderboardEntry(
            ordinal=ordinal,
            filename=path.name,
            score=round(float(score), 4),
            stage=metadata.get("stage"),
            stage_step=metadata.get("stage_step"),
            parent_ordinal=metadata.get("parent_ordinal"),
            opponents=tuple(),
            match_games=0,
            wins=0,
            draws=0,
            losses=0,
        )
        existing_entries[ordinal] = entry
        changed = True
    entries = sorted(existing_entries.values(), key=lambda item: item.ordinal)
    if changed or not leaderboard_path(model_dir, board_size).exists():
        write_leaderboard(model_dir, board_size, entries)
    return entries


def upsert_leaderboard_entry(
    model_dir: Path,
    board_size: int,
    entry: LeaderboardEntry,
) -> list[LeaderboardEntry]:
    entries = bootstrap_leaderboard(model_dir, board_size, base_score=entry.score)
    entry_map = {item.ordinal: item for item in entries}
    entry_map[entry.ordinal] = entry
    updated_entries = sorted(entry_map.values(), key=lambda item: item.ordinal)
    write_leaderboard(model_dir, board_size, updated_entries)
    return updated_entries


def latest_leaderboard_entry(entries: list[LeaderboardEntry]) -> LeaderboardEntry | None:
    return max(entries, key=lambda item: item.ordinal) if entries else None


def elo_expected_score(candidate_score: float, opponent_score: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_score - candidate_score) / 400.0))


def rounded_score(score: float) -> float:
    return round(float(score), 4)
