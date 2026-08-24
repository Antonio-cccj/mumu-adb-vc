from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vision import RecognitionSize


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adb_path: str | None = None
    device_serial: str | None = None
    recognition_width: int = Field(default=1280, gt=0)
    recognition_height: int = Field(default=720, gt=0)
    screenshot_interval_ms: int = Field(default=500, ge=50)
    debug_dir: Path = Path("debug")
    log_dir: Path = Path("logs")
    template_dir: Path = Path("assets/templates")
    command_timeout_sec: float = Field(default=10.0, gt=0)
    emulator_path: Path | None = None
    auto_launch_emulator: bool = True
    emulator_launch_timeout_sec: float = Field(default=90.0, gt=0)
    game_package: str = "com.tencent.tmgp.sgame"
    close_game_after_run: bool = True

    @field_validator("adb_path", "device_serial", "emulator_path", mode="before")
    @classmethod
    def blank_string_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def recognition_size(self) -> RecognitionSize:
        return RecognitionSize(width=self.recognition_width, height=self.recognition_height)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return AppConfig.model_validate(raw)
