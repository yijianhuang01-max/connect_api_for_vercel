from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import constants


@dataclass(frozen=True)
class RewardConfig:
    win: float = constants.WIN_REWARD
    draw: float = constants.DRAW_REWARD
    step: float = constants.STEP_REWARD
    invalid: float = constants.INVALID_MOVE_PENALTY


@dataclass(frozen=True)
class GameConfig:
    board_size: int = constants.BOARD_SIZE
    win_length: int = constants.WIN_LENGTH
    empty_value: int = constants.EMPTY
    red_value: int = constants.RED
    blue_value: int = constants.BLUE
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    @property
    def board_shape(self) -> tuple[int, int, int]:
        return (self.board_size, self.board_size, self.board_size)

    def validate(self) -> None:
        if self.board_size < 2:
            raise ValueError("BOARD_SIZE must be at least 2.")
        if not 2 <= self.win_length <= self.board_size:
            raise ValueError("WIN_LENGTH must satisfy 2 <= WIN_LENGTH <= BOARD_SIZE.")


@dataclass(frozen=True)
class TrainerConfig:
    game: GameConfig = field(default_factory=GameConfig)
    device: str = constants.DEVICE
    network_arch: str = constants.NETWORK_ARCH
    search_enabled: bool = constants.SEARCH_ENABLED
    search_max_depth: int = constants.SEARCH_MAX_DEPTH
    search_time_limit_ms: int = constants.SEARCH_TIME_LIMIT_MS
    search_use_transposition: bool = constants.SEARCH_USE_TRANSPOSITION
    search_top_k_ordering: int = constants.SEARCH_TOP_K_ORDERING
    search_network_guidance_weight: float = constants.SEARCH_NETWORK_GUIDANCE_WEIGHT
    search_heuristic_weight: float = constants.SEARCH_HEURISTIC_WEIGHT
    self_play_search_depth: int = constants.SELF_PLAY_SEARCH_DEPTH
    inference_search_depth: int = constants.INFERENCE_SEARCH_DEPTH
    eval_search_depth: int = constants.EVAL_SEARCH_DEPTH
    learning_rate: float = constants.LEARNING_RATE
    gamma: float = constants.GAMMA
    replay_capacity: int = constants.REPLAY_CAPACITY
    batch_size: int = constants.BATCH_SIZE
    target_update_interval: int = constants.TARGET_UPDATE_INTERVAL
    online_episodes: int = constants.ONLINE_EPISODES
    offline_epochs: int = constants.OFFLINE_EPOCHS
    checkpoint_episode_interval: int = constants.CHECKPOINT_EPISODE_INTERVAL
    checkpoint_epoch_interval: int = constants.CHECKPOINT_EPOCH_INTERVAL
    opponent_number: int = constants.OPPONENT_NUMBER
    checkpoint_match_games: int = constants.CHECKPOINT_MATCH_GAMES
    checkpoint_base_score: float = constants.CHECKPOINT_BASE_SCORE
    checkpoint_score_k: float = constants.CHECKPOINT_SCORE_K
    eval_interval: int = constants.EVAL_INTERVAL
    offline_log_batch_interval: int = constants.OFFLINE_LOG_BATCH_INTERVAL
    online_log_episode_interval: int = constants.ONLINE_LOG_EPISODE_INTERVAL
    epsilon_start: float = constants.EPSILON_START
    epsilon_min: float = constants.EPSILON_MIN
    epsilon_decay: float = constants.EPSILON_DECAY
    offline_batch_size: int = constants.OFFLINE_BATCH_SIZE
    model_dir: Path = constants.MODEL_DIR
    eval_games: int = constants.EVAL_GAMES
    random_seed: int = constants.RANDOM_SEED


@dataclass(frozen=True)
class DatasetConfig:
    game: GameConfig = field(default_factory=GameConfig)
    dataset_dir: Path = constants.DATASET_DIR
    chunk_size: int = constants.DATASET_CHUNK_SIZE
    episodes: int = constants.DATASET_EPISODES
    augment: bool = constants.DATASET_AUGMENT
    write_compressed: bool = constants.DATASET_WRITE_COMPRESSED
    search_enabled: bool = constants.SEARCH_ENABLED
    search_max_depth: int = constants.SEARCH_MAX_DEPTH
    search_time_limit_ms: int = constants.SEARCH_TIME_LIMIT_MS
    search_use_transposition: bool = constants.SEARCH_USE_TRANSPOSITION
    search_top_k_ordering: int = constants.SEARCH_TOP_K_ORDERING
    search_network_guidance_weight: float = constants.SEARCH_NETWORK_GUIDANCE_WEIGHT
    search_heuristic_weight: float = constants.SEARCH_HEURISTIC_WEIGHT
    self_play_search_depth: int = constants.SELF_PLAY_SEARCH_DEPTH
    invalid_sampling_prob: float = constants.INVALID_SAMPLING_PROB
    tactical_policy_prob: float = constants.TACTICAL_POLICY_PROB
    random_policy_prob: float = constants.RANDOM_POLICY_PROB
    num_workers: int = constants.NUM_DATASET_WORKERS
    random_seed: int = constants.RANDOM_SEED


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    invalid: bool
    winner: int
    legal_action_mask: tuple[bool, ...]
    action: int
    player: int
    position: tuple[int, int, int] | None
