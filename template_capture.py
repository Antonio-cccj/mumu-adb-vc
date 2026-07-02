from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2

from adb_controller import AdbController
from vision import RecognitionSize, resize_to_recognition


@dataclass(frozen=True)
class CaptureStage:
    stage_id: str
    label: str

    @property
    def display_text(self) -> str:
        return f"{self.stage_id} - {self.label}"


CAPTURE_STAGES: tuple[CaptureStage, ...] = (
    CaptureStage("01_launcher", "桌面王者荣耀图标"),
    CaptureStage("02_start_game", "开始游戏按钮"),
    CaptureStage("03_popup_close", "弹窗关闭按钮"),
    CaptureStage("04_lobby_farm_entry", "大厅农场入口"),
    CaptureStage("05_farm_loaded", "农场已加载标识"),
    CaptureStage("06_after_move", "走到交互点后"),
    CaptureStage("07_one_key_farm", "一键务农按钮"),
)


SAFE_STAGE_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_stage_id(stage_id: str) -> str:
    sanitized = SAFE_STAGE_PATTERN.sub("_", stage_id.strip()).strip("_")
    return sanitized or "capture"


def stage_id_from_display(display_text: str) -> str:
    return sanitize_stage_id(display_text.split(" - ", 1)[0])


def next_capture_path(source_dir: str | Path, stage_id: str) -> Path:
    source_path = Path(source_dir)
    safe_stage = sanitize_stage_id(stage_id)
    existing = sorted(source_path.glob(f"{safe_stage}_*.png"))
    next_index = len(existing) + 1
    return source_path / f"{safe_stage}_{next_index:03d}.png"


def capture_recognition_source(
    *,
    adb: AdbController,
    source_dir: str | Path,
    stage_id: str,
    recognition_size: RecognitionSize,
) -> Path:
    source_path = Path(source_dir)
    source_path.mkdir(parents=True, exist_ok=True)
    screenshot = adb.screencap_png()
    recognition_image = resize_to_recognition(screenshot, recognition_size)
    output = next_capture_path(source_path, stage_id)
    if not cv2.imwrite(str(output), recognition_image):
        raise OSError(f"Failed to save template source screenshot: {output}")
    return output
