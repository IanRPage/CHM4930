"""
Server settings, read from `MUFU_*` environment variables.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

COVERAGE = 0.95


@dataclass(frozen=True)
class Settings:
    checkpoint: Path
    api_key: str = field(repr=False)
    model_version: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cpu"
    rate_limit_per_minute: int = 30
    max_concurrency: int = 1
    max_smiles_length: int = 1000
    max_body_bytes: int = 4096
    request_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not 0 < self.port < 65536:
            raise ValueError("MUFU_PORT must be between 1 and 65535")
        if not (math.isfinite(self.request_timeout_s) and self.request_timeout_s > 0):
            raise ValueError("MUFU_REQUEST_TIMEOUT_S must be a positive number")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        checkpoint = env.get("MUFU_CHECKPOINT", "").strip()
        api_key = env.get("MUFU_API_KEY", "").strip()
        if not checkpoint:
            raise ValueError("MUFU_CHECKPOINT must be set")
        if not api_key:
            raise ValueError("MUFU_API_KEY must be set")
        optional = {
            "model_version": ("MUFU_MODEL_VERSION", str),
            "host": ("MUFU_HOST", str),
            "port": ("MUFU_PORT", int),
            "device": ("MUFU_DEVICE", str),
            "rate_limit_per_minute": ("MUFU_RATE_LIMIT_PER_MINUTE", int),
            "max_concurrency": ("MUFU_MAX_CONCURRENCY", int),
            "request_timeout_s": ("MUFU_REQUEST_TIMEOUT_S", float),
        }
        return cls(
            checkpoint=Path(checkpoint),
            api_key=api_key,
            **{
                key: cast(env[name])
                for key, (name, cast) in optional.items()
                if env.get(name)
            },
        )
