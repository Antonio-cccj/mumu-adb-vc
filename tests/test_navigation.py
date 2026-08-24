from pathlib import Path

import cv2
import numpy as np

from navigation.feature_localizer import FeatureLocalizer
from navigation.route_config import NavigationRoute, load_navigation_route
from navigation.route_navigator import NavigationCallbacks, RouteNavigator
from settings import AppConfig
from vision import RecognitionSize


def _write_textured_image(path: Path, seed: int, marker: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 180, size=(240, 320, 3), dtype=np.uint8)
    cv2.circle(image, marker, 26, (255, 255, 255), -1)
    cv2.line(image, (20, 30), (280, 200), (0, 255, 0), 3)
    cv2.rectangle(image, (marker[0] - 40, marker[1] + 30), (marker[0] + 40, marker[1] + 70), (255, 0, 0), 2)
    cv2.imwrite(str(path), image)
    return image


def test_load_navigation_route_parses_keyframes(tmp_path: Path):
    (tmp_path / "spawn.png").write_bytes(b"placeholder")
    route_path = tmp_path / "route.yaml"
    route_path.write_text(
        """
name: demo
scene_roi: [0, 0, 320, 240]
joystick:
  center: [100, 200]
  distance: 60
  duration_ms: 250
  wait_ms: 10
success:
  templates: [ok.png]
  threshold: 0.9
keyframes:
  - name: spawn
    image: spawn.png
    move: up_left
    min_score: 5
""",
        encoding="utf-8",
    )

    route = load_navigation_route(route_path)

    assert route.name == "demo"
    assert route.joystick.center == [100, 200]
    assert route.keyframes[0].image == "spawn.png"
    assert route.resolve_path("spawn.png") == tmp_path / "spawn.png"


def test_fixed_step_route_accepts_fast_polling_settings(tmp_path: Path):
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "mode": "fixed_step",
            "fixed_step": {
                "direction": "up_left",
                "step_wait_ms": 1200,
                "settle_wait_ms": 400,
                "success_poll_interval_ms": 300,
                "reset_after_ms": 300000,
            },
            "keyframes": [{"name": "anchor", "image": "anchor.png", "move": "wait", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)

    assert route.fixed_step.step_wait_ms == 1200
    assert route.fixed_step.settle_wait_ms == 400
    assert route.fixed_step.success_poll_interval_ms == 300


def test_feature_localizer_prefers_matching_keyframe(tmp_path: Path):
    spawn = _write_textured_image(tmp_path / "spawn.png", 1, (90, 120))
    near = _write_textured_image(tmp_path / "near.png", 2, (220, 100))
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "scene_roi": [0, 0, 320, 240],
            "keyframes": [
                {"name": "spawn", "image": "spawn.png", "move": "left", "min_score": 5},
                {"name": "near", "image": "near.png", "move": "right", "min_score": 5},
            ],
        }
    )
    route.bind_base_dir(tmp_path)
    localizer = FeatureLocalizer(route, RecognitionSize(width=320, height=240))

    result = localizer.locate(near)

    assert result.found is True
    assert result.keyframe_name == "near"
    assert result.index == 1
    assert result.score > 5
    assert localizer.locate(spawn).keyframe_name == "spawn"
    assert result.candidates[0].keyframe_name == "near"
    assert len(result.candidates) == 2


def test_route_navigator_closes_interrupt_before_moving(tmp_path: Path):
    close = np.zeros((20, 20, 3), dtype=np.uint8)
    close[4:16, 4:16] = (255, 255, 255)
    close[8:12, 8:12] = (0, 0, 0)
    cv2.imwrite(str(tmp_path / "close.png"), close)
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    frame[30:50, 260:280] = close
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "interrupt": {"templates": ["close.png"], "threshold": 0.95, "roi": [220, 0, 100, 90], "wait_ms": 0},
            "keyframes": [{"name": "spawn", "image": "close.png", "move": "left", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)
    taps: list[tuple[int, int]] = []
    swipes: list[tuple[int, int, int, int, int]] = []
    callbacks = NavigationCallbacks(
        screencap=lambda: frame.copy(),
        tap=lambda x, y: taps.append((x, y)),
        swipe=lambda x1, y1, x2, y2, duration: swipes.append((x1, y1, x2, y2, duration)),
        sleep_ms=lambda ms: "success",
        emit=lambda *args, **kwargs: None,
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)

    result = navigator.run(max_attempts=1)

    assert result.status == "failed"
    assert taps == [(270, 40)]
    assert swipes == []


def test_route_navigator_moves_by_localized_keyframe(tmp_path: Path):
    frame = _write_textured_image(tmp_path / "spawn.png", 3, (100, 120))
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "scene_roi": [0, 0, 320, 240],
            "joystick": {"center": [100, 200], "distance": 50, "duration_ms": 250, "wait_ms": 0},
            "keyframes": [{"name": "spawn", "image": "spawn.png", "move": "up_right", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)
    swipes: list[tuple[int, int, int, int, int]] = []
    callbacks = NavigationCallbacks(
        screencap=lambda: frame.copy(),
        tap=lambda x, y: None,
        swipe=lambda x1, y1, x2, y2, duration: swipes.append((x1, y1, x2, y2, duration)),
        sleep_ms=lambda ms: "success",
        emit=lambda *args, **kwargs: None,
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)

    result = navigator.run(max_attempts=1)

    assert result.status == "failed"
    assert swipes == [(100, 200, 135, 165, 250)]


def test_route_navigator_uses_keyframe_correction_when_stuck(tmp_path: Path):
    frame = _write_textured_image(tmp_path / "spawn.png", 4, (110, 125))
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "scene_roi": [0, 0, 320, 240],
            "stuck_after": 1,
            "joystick": {"center": [100, 200], "distance": 50, "duration_ms": 250, "wait_ms": 0},
            "keyframes": [
                {
                    "name": "spawn",
                    "image": "spawn.png",
                    "move": "left",
                    "correction_moves": ["right"],
                    "min_score": 5,
                }
            ],
        }
    )
    route.bind_base_dir(tmp_path)
    swipes: list[tuple[int, int, int, int, int]] = []
    events: list[dict] = []
    callbacks = NavigationCallbacks(
        screencap=lambda: frame.copy(),
        tap=lambda x, y: None,
        swipe=lambda x1, y1, x2, y2, duration: swipes.append((x1, y1, x2, y2, duration)),
        sleep_ms=lambda ms: "success",
        emit=lambda level, category, message, **kwargs: events.append({"message": message, **kwargs}),
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)

    result = navigator.run(max_attempts=2)

    assert result.status == "failed"
    assert swipes == [(100, 200, 50, 200, 250), (100, 200, 150, 200, 250)]
    assert events[-1]["data"]["decision"] == "correction"
    assert events[-1]["data"]["same_keyframe_count"] == 2


def test_route_navigator_fixed_step_moves_up_left_and_waits_each_step(tmp_path: Path):
    frame = _write_textured_image(tmp_path / "anchor.png", 5, (120, 120))
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "mode": "fixed_step",
            "fixed_step": {"direction": "up_left", "step_wait_ms": 8000, "reset_after_ms": 300000},
            "joystick": {"center": [100, 200], "distance": 50, "duration_ms": 250, "wait_ms": 0},
            "keyframes": [{"name": "anchor", "image": "anchor.png", "move": "wait", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)
    swipes: list[tuple[int, int, int, int, int]] = []
    waits: list[int] = []
    callbacks = NavigationCallbacks(
        screencap=lambda: frame.copy(),
        tap=lambda x, y: None,
        swipe=lambda x1, y1, x2, y2, duration: swipes.append((x1, y1, x2, y2, duration)),
        sleep_ms=lambda ms: waits.append(ms) or "success",
        emit=lambda *args, **kwargs: None,
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)

    result = navigator.run(max_attempts=2)

    assert result.status == "failed"
    assert swipes == [(100, 200, 65, 165, 250), (100, 200, 65, 165, 250)]
    assert waits == [8000, 8000]


def test_route_navigator_fixed_step_polls_success_after_move(tmp_path: Path):
    ok = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.line(ok, (2, 2), (17, 17), (255, 255, 255), 2)
    cv2.line(ok, (17, 2), (2, 17), (0, 220, 255), 2)
    cv2.rectangle(ok, (6, 6), (13, 13), (20, 80, 230), -1)
    cv2.imwrite(str(tmp_path / "ok.png"), ok)
    before = np.full((240, 320, 3), 20, dtype=np.uint8)
    after = before.copy()
    after[110:130, 230:250] = ok
    frames = [before, after]
    screenshots = 0

    def screencap() -> np.ndarray:
        nonlocal screenshots
        screenshots += 1
        index = min(screenshots - 1, len(frames) - 1)
        return frames[index].copy()

    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "mode": "fixed_step",
            "fixed_step": {
                "direction": "up_left",
                "step_wait_ms": 1200,
                "settle_wait_ms": 400,
                "success_poll_interval_ms": 400,
                "reset_after_ms": 300000,
            },
            "success": {"templates": ["ok.png"], "threshold": 0.95, "roi": [200, 80, 100, 100], "wait_ms": 0},
            "joystick": {"center": [100, 200], "distance": 50, "duration_ms": 250, "wait_ms": 0},
            "keyframes": [{"name": "anchor", "image": "ok.png", "move": "wait", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)
    swipes: list[tuple[int, int, int, int, int]] = []
    waits: list[int] = []
    callbacks = NavigationCallbacks(
        screencap=screencap,
        tap=lambda x, y: None,
        swipe=lambda x1, y1, x2, y2, duration: swipes.append((x1, y1, x2, y2, duration)),
        sleep_ms=lambda ms: waits.append(ms) or "success",
        emit=lambda *args, **kwargs: None,
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)

    result = navigator.run(max_attempts=3)

    assert result.status == "success"
    assert swipes == [(100, 200, 65, 165, 250)]
    assert waits == [400]
    assert screenshots == 2


def test_route_navigator_caches_route_templates(tmp_path: Path, monkeypatch):
    ok = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.line(ok, (2, 2), (17, 17), (255, 255, 255), 2)
    cv2.line(ok, (17, 2), (2, 17), (0, 220, 255), 2)
    cv2.imwrite(str(tmp_path / "ok.png"), ok)
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "mode": "fixed_step",
            "fixed_step": {
                "direction": "up_left",
                "step_wait_ms": 0,
                "settle_wait_ms": 0,
                "success_poll_interval_ms": 400,
                "reset_after_ms": 300000,
            },
            "success": {"templates": ["ok.png"], "threshold": 0.95, "roi": [200, 80, 100, 100], "wait_ms": 0},
            "joystick": {"center": [100, 200], "distance": 50, "duration_ms": 250, "wait_ms": 0},
            "keyframes": [{"name": "anchor", "image": "ok.png", "move": "wait", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)
    original_imread = cv2.imread
    reads: list[str] = []

    def counting_imread(path, flags=cv2.IMREAD_COLOR):
        reads.append(Path(path).name)
        return original_imread(path, flags)

    callbacks = NavigationCallbacks(
        screencap=lambda: frame.copy(),
        tap=lambda x, y: None,
        swipe=lambda x1, y1, x2, y2, duration: None,
        sleep_ms=lambda ms: "success",
        emit=lambda *args, **kwargs: None,
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)
    monkeypatch.setattr("navigation.route_navigator.cv2.imread", counting_imread)

    result = navigator.run(max_attempts=2)

    assert result.status == "failed"
    assert reads.count("ok.png") == 1


def test_route_navigator_fixed_step_clicks_reset_after_timeout(tmp_path: Path):
    reset = np.zeros((20, 20, 3), dtype=np.uint8)
    reset[3:17, 3:17] = (180, 180, 180)
    reset[7:13, 7:13] = (40, 40, 40)
    cv2.imwrite(str(tmp_path / "reset.png"), reset)
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    frame[190:210, 280:300] = reset
    route = NavigationRoute.model_validate(
        {
            "name": "demo",
            "mode": "fixed_step",
            "fixed_step": {"direction": "up_left", "step_wait_ms": 8000, "reset_after_ms": 1},
            "reset": {"templates": ["reset.png"], "threshold": 0.95, "roi": [240, 160, 80, 80], "wait_ms": 0},
            "joystick": {"center": [100, 200], "distance": 50, "duration_ms": 250, "wait_ms": 0},
            "keyframes": [{"name": "anchor", "image": "reset.png", "move": "wait", "min_score": 5}],
        }
    )
    route.bind_base_dir(tmp_path)
    taps: list[tuple[int, int]] = []
    swipes: list[tuple[int, int, int, int, int]] = []
    events: list[dict] = []
    callbacks = NavigationCallbacks(
        screencap=lambda: frame.copy(),
        tap=lambda x, y: taps.append((x, y)),
        swipe=lambda x1, y1, x2, y2, duration: swipes.append((x1, y1, x2, y2, duration)),
        sleep_ms=lambda ms: "success",
        emit=lambda level, category, message, **kwargs: events.append({"message": message, **kwargs}),
        stopped=lambda: False,
    )
    config = AppConfig(
        template_dir=tmp_path,
        debug_dir=tmp_path / "debug",
        recognition_width=320,
        recognition_height=240,
    )
    navigator = RouteNavigator(route, config, callbacks)
    navigator._fixed_step_started_at -= 2

    result = navigator.run(max_attempts=1)

    assert result.status == "failed"
    assert taps == [(290, 200)]
    assert swipes == []
    assert events[-1]["data"]["decision"] == "reset"
