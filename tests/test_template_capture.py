from pathlib import Path

import cv2
import numpy as np

from template_capture import (
    CAPTURE_STAGES,
    capture_recognition_source,
    next_capture_path,
    sanitize_stage_id,
)
from vision import RecognitionSize


def test_sanitize_stage_id_keeps_safe_ascii_name():
    assert sanitize_stage_id("02 start/game") == "02_start_game"
    assert sanitize_stage_id(" 农场入口 ") == "capture"


def test_next_capture_path_uses_incrementing_stage_prefix(tmp_path: Path):
    source_dir = tmp_path / "template_sources"
    source_dir.mkdir()
    (source_dir / "01_launcher_001.png").write_bytes(b"old")
    (source_dir / "01_launcher_002.png").write_bytes(b"old")

    path = next_capture_path(source_dir, "01_launcher")

    assert path == source_dir / "01_launcher_003.png"


def test_capture_stages_include_wzry_flow_names():
    stage_ids = [stage.stage_id for stage in CAPTURE_STAGES]

    assert "01_launcher" in stage_ids
    assert "07_one_key_farm" in stage_ids


def test_capture_recognition_source_saves_1280x720_image(tmp_path: Path):
    class FakeAdb:
        def screencap_png(self):
            return np.zeros((1440, 2560, 3), dtype=np.uint8)

    output = capture_recognition_source(
        adb=FakeAdb(),  # type: ignore[arg-type]
        source_dir=tmp_path,
        stage_id="farm_loaded",
        recognition_size=RecognitionSize(width=1280, height=720),
    )

    assert output.name == "farm_loaded_001.png"
    assert output.exists()
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[:2] == (720, 1280)
