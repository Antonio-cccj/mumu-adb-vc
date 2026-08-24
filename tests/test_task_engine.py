from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from settings import AppConfig
from navigation.route_config import load_navigation_route
from task_engine import TaskEngine
from task_engine import TaskDefinition, load_task_definition
from runtime_events import EventHub
from time_utils import beijing_now
from vision import MatchResult, Point, RecognitionSize, Rect, match_template


def _synthetic_maturity_screen(text: str) -> np.ndarray:
    screen = np.full((720, 1280, 3), 245, dtype=np.uint8)
    screen[120:340, 25:225] = (248, 244, 229)
    image = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 27)
    draw.text((74, 261), f"{text}成熟", fill=(90, 84, 78), font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def test_load_task_definition_parses_nodes_and_defaults(tmp_path: Path):
    task_path = tmp_path / "demo.yaml"
    task_path.write_text(
        """
start: confirm
fallback: wait_retry
max_steps: 5
nodes:
  - name: confirm
    template: confirm.png
    roi: [10, 20, 100, 80]
    threshold: 0.88
    action: click
    next: stop
    timeout_ms: 1000
    retry: 2
    on_fail: wait_retry
  - name: wait_retry
    action: wait
    wait_ms: 200
    next: confirm
  - name: stop
    action: stop
""",
        encoding="utf-8",
    )

    task = load_task_definition(task_path)

    assert isinstance(task, TaskDefinition)
    assert task.start == "confirm"
    assert task.fallback == "wait_retry"
    assert task.max_steps == 5
    assert task.nodes_by_name["confirm"].roi == [10, 20, 100, 80]
    assert task.nodes_by_name["wait_retry"].retry == 1


def test_task_definition_parses_display_name_and_terminal_status():
    task = TaskDefinition.model_validate(
        {
            "start": "stop_failed",
            "nodes": [
                {
                    "name": "stop_failed",
                    "display_name": "失败停止",
                    "action": "stop",
                    "terminal_status": "failed",
                }
            ],
        }
    )

    node = task.nodes_by_name["stop_failed"]
    assert node.display_name == "失败停止"
    assert node.terminal_status == "failed"


def test_task_definition_parses_auto_detect_start_nodes():
    task = TaskDefinition.model_validate(
        {
            "start": "home",
            "auto_detect_start": True,
            "entry_nodes": ["farm_action", "home"],
            "nodes": [
                {"name": "home", "action": "wait", "next": "farm_action"},
                {"name": "farm_action", "template": "farm.png", "action": "click"},
            ],
        }
    )

    assert task.auto_detect_start is True
    assert task.entry_nodes == ["farm_action", "home"]


def test_task_definition_parses_auto_detect_timeout():
    task = TaskDefinition.model_validate(
        {
            "start": "home",
            "auto_detect_start": True,
            "auto_detect_timeout_ms": 30000,
            "nodes": [{"name": "home", "action": "wait"}],
        }
    )

    assert task.auto_detect_timeout_ms == 30000


def test_task_definition_parses_stuck_recheck_settings():
    task = TaskDefinition.model_validate(
        {
            "start": "loop",
            "stuck_recheck_after": 5,
            "stuck_recheck_nodes": ["target", "loop"],
            "nodes": [
                {"name": "loop", "action": "wait", "next": "loop"},
                {"name": "target", "template": "target.png", "action": "click"},
            ],
        }
    )

    assert task.stuck_recheck_after == 5
    assert task.stuck_recheck_nodes == ["target", "loop"]


def test_task_definition_parses_read_maturity_time_action():
    task = TaskDefinition.model_validate(
        {
            "start": "read_time",
            "nodes": [
                {
                    "name": "read_time",
                    "action": "read_maturity_time",
                    "maturity": {
                        "max_attempts": 3,
                        "step_wait_ms": 1,
                        "anchor_template": "shovel.png",
                        "anchor_roi": [0, 120, 260, 360],
                        "anchor_threshold": 0.7,
                        "tomorrow_template": "tomorrow.png",
                        "tomorrow_threshold": 0.8,
                        "farm_templates": ["farm_scene.png", "farm_icon.png"],
                        "farm_roi": [0, 0, 1280, 720],
                        "farm_threshold": 0.75,
                        "farm_min_matches": 2,
                    },
                }
            ],
        }
    )

    node = task.nodes_by_name["read_time"]
    assert node.action == "read_maturity_time"
    assert node.maturity is not None
    assert node.maturity.max_attempts == 3
    assert node.maturity.anchor_template == "shovel.png"
    assert node.maturity.anchor_roi == [0, 120, 260, 360]
    assert node.maturity.tomorrow_template == "tomorrow.png"
    assert node.maturity.tomorrow_threshold == 0.8
    assert node.maturity.farm_templates == ["farm_scene.png", "farm_icon.png"]
    assert node.maturity.farm_roi == [0, 0, 1280, 720]
    assert node.maturity.farm_threshold == 0.75
    assert node.maturity.farm_min_matches == 2


def test_task_engine_tap_point_maps_recognition_coordinate_to_screen(tmp_path: Path):
    screen = np.zeros((1440, 2560, 3), dtype=np.uint8)

    class FakeAdb:
        def __init__(self):
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "dismiss",
            "nodes": [
                {
                    "name": "dismiss",
                    "action": "tap_point",
                    "point": [640, 620],
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(1280, 1240)]


def test_task_engine_moves_to_crop_and_emits_maturity_time(tmp_path: Path):
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    maturity = _synthetic_maturity_screen("13:04")

    class FakeAdb:
        def __init__(self):
            self.screens = [blank, blank, maturity]
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            if len(self.screens) > 1:
                return self.screens.pop(0)
            return self.screens[0].copy()

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "read_time",
            "nodes": [
                {
                    "name": "read_time",
                    "action": "read_maturity_time",
                    "maturity": {"max_attempts": 3, "step_wait_ms": 1},
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )
    events = EventHub()
    emitted = []
    events.subscribe(emitted.append)
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug", screenshot_interval_ms=50),
        task=task,
        events=events,
    )

    result = engine.run()

    assert result.status == "completed"
    assert len(fake_adb.swipes) == 2
    assert any(
        event.category == "maturity"
        and event.data.get("maturity_time") == "13:04"
        and event.data.get("attempt") == 3
        for event in emitted
    )


def test_task_engine_returns_to_reward_close_when_maturity_read_is_not_in_farm(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    rng = np.random.default_rng(123)
    farm_template = rng.integers(80, 220, size=(28, 32, 3), dtype=np.uint8)
    reward_template = rng.integers(60, 240, size=(24, 36, 3), dtype=np.uint8)
    cv2.imwrite(str(template_dir / "farm.png"), farm_template)
    cv2.imwrite(str(template_dir / "reward.png"), reward_template)

    screen = rng.integers(0, 50, size=(720, 1280, 3), dtype=np.uint8)
    screen[80:104, 520:556] = reward_template

    class FakeAdb:
        def __init__(self):
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "read_time",
            "nodes": [
                {
                    "name": "read_time",
                    "action": "read_maturity_time",
                    "maturity": {
                        "farm_templates": ["farm.png"],
                        "farm_threshold": 0.96,
                        "max_attempts": 1,
                    },
                    "on_fail": "dismiss_reward",
                },
                {
                    "name": "dismiss_reward",
                    "template": "reward.png",
                    "roi": [500, 60, 100, 80],
                    "threshold": 0.95,
                    "action": "tap_point",
                    "point": [640, 620],
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug", screenshot_interval_ms=50),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(640, 620)]


def test_wzry_harvest_task_file_is_valid():
    task_path = Path(__file__).parents[1] / "tasks" / "wzry_harvest.yaml"
    task = load_task_definition(task_path)

    assert task.start == "harvest_candidates"
    assert task.nodes_by_name["harvest_candidates"].templates
    assert task.nodes_by_name["harvest_candidates"].post_action_wait_ms == 800


def test_wzry_farm_task_file_is_valid():
    task_path = Path(__file__).parents[1] / "tasks" / "wzry_farm.yaml"
    task = load_task_definition(task_path)

    assert task.start == "go_home"
    assert task.auto_detect_start is True
    assert task.auto_detect_timeout_ms >= 30000
    assert "confirm_resource_update_restart" in task.entry_nodes
    assert "confirm_agreement_stop" in task.entry_nodes
    assert task.entry_nodes.index("confirm_resource_update_restart") < task.entry_nodes.index("close_popups")
    assert task.entry_nodes.index("confirm_agreement_stop") < task.entry_nodes.index("close_popups")
    assert "wait_farm_loading" in task.entry_nodes
    assert "wait_game_loading" in task.entry_nodes
    assert "return_to_lobby" in task.entry_nodes
    assert task.entry_nodes.index("return_to_lobby") < task.entry_nodes.index("open_wzry")
    assert task.stuck_recheck_after == 5
    assert "confirm_resource_update_restart" in task.stuck_recheck_nodes
    assert "confirm_agreement_stop" in task.stuck_recheck_nodes
    assert task.stuck_recheck_nodes.index("confirm_resource_update_restart") < task.stuck_recheck_nodes.index("close_popups")
    assert task.stuck_recheck_nodes.index("confirm_agreement_stop") < task.stuck_recheck_nodes.index("close_popups")
    assert "wait_farm_loading" in task.stuck_recheck_nodes
    assert "wait_game_loading" in task.stuck_recheck_nodes
    assert "return_to_lobby" in task.stuck_recheck_nodes
    assert task.nodes_by_name["open_wzry"].templates
    assert task.nodes_by_name["go_home"].next == "close_mumu_ad"
    assert "close_mumu_ad" in task.entry_nodes
    assert task.entry_nodes.index("close_mumu_ad") < task.entry_nodes.index("open_wzry")
    assert "close_mumu_ad" in task.stuck_recheck_nodes
    assert task.stuck_recheck_nodes.index("close_mumu_ad") < task.stuck_recheck_nodes.index("open_wzry")
    mumu_ad_node = task.nodes_by_name["close_mumu_ad"]
    assert mumu_ad_node.action == "click"
    assert mumu_ad_node.next == "open_wzry"
    assert mumu_ad_node.on_fail == "open_wzry"
    assert "wzry/mumu_ad_close.png" in mumu_ad_node.template_names()
    assert task.nodes_by_name["start_game"].timeout_ms >= 120000
    assert "wzry/start_game_202608.png" in task.nodes_by_name["start_game"].template_names()
    # 正确流程：open_wzry → wait_game_loading（初始加载）→ start_game → close_popups。
    # 等待游戏加载在点击"开始游戏"之前，而非之后。
    assert task.nodes_by_name["open_wzry"].next == "wait_game_loading"
    assert task.nodes_by_name["start_game"].next == "close_popups"
    assert task.nodes_by_name["start_game"].on_fail == "stop_failed"
    loading_node = task.nodes_by_name["wait_game_loading"]
    assert loading_node.template == "wzry/game_loading_status.png"
    assert loading_node.display_name == "等待游戏加载"
    # 加载完成（模板消失）后进入点击开始游戏节点。
    assert loading_node.on_fail == "start_game"
    return_node = task.nodes_by_name["return_to_lobby"]
    assert return_node.template == "wzry/subpage_back_button.png"
    assert return_node.action == "tap_point"
    assert return_node.point == [52, 36]
    assert return_node.next == "close_popups"
    assert task.nodes_by_name["close_popups"].on_fail == "confirm_resource_update_restart"
    assert "wzry/popup_close_duckyo_x.png" in task.nodes_by_name["close_popups"].template_names()
    resource_update_node = task.nodes_by_name["confirm_resource_update_restart"]
    assert resource_update_node.action == "tap_point"
    assert resource_update_node.template == "wzry/resource_update_confirm_dialog.png"
    assert resource_update_node.point == [632, 510]
    assert resource_update_node.next == "wait_game_loading"
    assert resource_update_node.on_fail == "confirm_agreement_stop"
    agreement_node = task.nodes_by_name["confirm_agreement_stop"]
    assert agreement_node.action == "tap_point"
    assert agreement_node.template == "wzry/agreement_confirm_dialog.png"
    assert agreement_node.point == [765, 565]
    assert agreement_node.next == "stop_keep_game_open"
    assert agreement_node.on_fail == "find_farm_entry"
    stop_keep_game_node = task.nodes_by_name["stop_keep_game_open"]
    assert stop_keep_game_node.action == "stop"
    assert stop_keep_game_node.terminal_status == "stopped"
    assert stop_keep_game_node.keep_game_open_after_run is True
    assert "wzry/farm_entry_homestead.png" in task.nodes_by_name["find_farm_entry"].templates
    assert "wzry/farm_entry_current.png" in task.nodes_by_name["find_farm_entry"].templates
    assert "wzry/farm_entry_202608.png" in task.nodes_by_name["find_farm_entry"].templates
    assert "wzry/farm_entry_lobby_20260810.png" in task.nodes_by_name["find_farm_entry"].templates
    assert "wzry/farm_entry_lobby_text_20260810.png" in task.nodes_by_name["find_farm_entry"].templates
    assert task.nodes_by_name["find_farm_entry"].next == "wait_farm_loading"
    farm_loading_node = task.nodes_by_name["wait_farm_loading"]
    assert farm_loading_node.next == "wait_farm_loading"
    assert farm_loading_node.on_fail == "wait_farm_loaded"
    assert farm_loading_node.wait_ms >= 5000
    assert farm_loading_node.no_stuck_recheck is True
    assert "wzry/farm_loading_room_202608.png" in farm_loading_node.template_names()
    assert task.nodes_by_name["wait_farm_loaded"].min_matches == 2
    assert task.nodes_by_name["wait_farm_loaded"].next == "close_farm_popups"
    assert task.nodes_by_name["wait_farm_loaded"].timeout_ms >= 300000
    farm_popup_node = task.nodes_by_name["close_farm_popups"]
    assert farm_popup_node.action == "click"
    assert farm_popup_node.on_fail == "move_to_farm_button"
    assert "wzry/popup_close_duckyo_x.png" in farm_popup_node.template_names()
    route_node = task.nodes_by_name["move_to_farm_button"]
    assert route_node.action == "route_navigate"
    assert route_node.route is not None
    assert task.nodes_by_name["move_to_farm_button"].timeout_ms >= 30000
    assert route_node.route.max_attempts >= 45
    route = load_navigation_route(Path(__file__).parents[1] / route_node.route.path)
    assert len(route.keyframes) >= 4
    assert route.keyframes[-1].move == "stop"
    assert route.joystick.center[1] <= 540
    assert len(route.recovery_moves) >= 4
    assert task.nodes_by_name["one_key_farm"].templates
    assert task.nodes_by_name["one_key_farm"].next == "wait_after_one_key_farm"
    assert task.nodes_by_name["wait_after_one_key_farm"].next == "dismiss_reward_screen"
    reward_node = task.nodes_by_name["dismiss_reward_screen"]
    assert reward_node.action == "tap_point"
    assert "wzry/reward_title.png" in reward_node.template_names()
    assert "wzry/reward_title_current.png" in reward_node.template_names()
    assert reward_node.point == [640, 620]
    assert reward_node.timeout_ms >= 25000
    assert reward_node.retry >= 25
    assert reward_node.on_fail == "read_maturity_time"
    assert reward_node.next == "read_maturity_time"
    maturity_node = task.nodes_by_name["read_maturity_time"]
    assert maturity_node.action == "read_maturity_time"
    assert maturity_node.maturity is not None
    assert maturity_node.maturity.anchor_template == "wzry/maturity_shovel_button.png"
    assert maturity_node.maturity.tomorrow_template == "wzry/maturity_tomorrow_prefix.png"
    assert maturity_node.maturity.farm_templates
    # 务农后互动菜单可能遮挡场景锚点，只需 1 个农场 UI 元素匹配即可确认在农场。
    assert maturity_node.maturity.farm_min_matches >= 1
    assert maturity_node.maturity.direction == "up_left"
    assert maturity_node.maturity.step_wait_ms >= 8000
    assert maturity_node.maturity.max_attempts >= 30
    # 仅当识别到不在农场时才回退 dismiss_reward_screen；
    # dismiss_reward_screen 设了 no_stuck_recheck=true，不会误跳回 one_key_farm。
    assert maturity_node.on_fail == "dismiss_reward_screen"
    # dismiss_reward_screen 必须设置 no_stuck_recheck=true，防止务农后农场可见时被误判跳回 one_key_farm。
    assert task.nodes_by_name["dismiss_reward_screen"].no_stuck_recheck is True
    assert maturity_node.timeout_ms >= 300000
    assert task.nodes_by_name["stop_failed"].terminal_status == "failed"


def test_wzry_reward_screen_matches_current_reward_page_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["dismiss_reward_screen"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_reward_screen_cherry.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    assert any(result.found for result in results)
    assert node.point == [640, 620]


def test_wzry_popup_close_matches_duckyo_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_popup_duckyo_close.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    for node_name in ("close_popups", "close_farm_popups"):
        node = task.nodes_by_name[node_name]
        results = []
        for template_name in node.template_names():
            template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
            assert template is not None, template_name
            results.append(
                match_template(
                    fixture,
                    template,
                    threshold=node.threshold,
                    roi=Rect.from_sequence(node.roi),
                    recognition_size=RecognitionSize(width=1280, height=720),
                    actual_size=actual_size,
                )
            )

        assert any(result.found for result in results), node_name
        assert node.action == "click"


def test_wzry_farm_entry_matches_current_lobby_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["find_farm_entry"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_farm_entry_current.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    assert any(result.found for result in results)
    assert node.action == "click"


def test_wzry_farm_entry_matches_202608_lobby_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["find_farm_entry"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_farm_entry_202608.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    assert any(result.found for result in results)
    assert node.action == "click"


def test_wzry_farm_entry_matches_20260810_lobby_fixture_with_high_confidence():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["find_farm_entry"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_farm_entry_lobby_20260810.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    best = max(results, key=lambda result: result.score)
    assert best.found
    assert best.score >= 0.95
    assert node.action == "click"


def test_wzry_farm_loading_screen_keeps_waiting_instead_of_failing_fast():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["wait_farm_loading"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_farm_loading_room_202608.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    best = max(results, key=lambda result: result.score)
    assert best.found
    assert node.action == "wait"
    assert node.next == "wait_farm_loading"
    assert node.on_fail == "wait_farm_loaded"


def test_wzry_start_game_matches_202608_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["start_game"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_start_game_202608.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    assert any(result.found for result in results)
    assert node.action == "click"


def test_mumu_launcher_ad_close_matches_current_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["close_mumu_ad"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "mumu_launcher_ad.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    results = []
    for template_name in node.template_names():
        template = cv2.imread(str(template_dir / template_name), cv2.IMREAD_COLOR)
        assert template is not None, template_name
        results.append(
            match_template(
                fixture,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=RecognitionSize(width=1280, height=720),
                actual_size=actual_size,
            )
        )

    assert any(result.found for result in results)
    assert node.next == "open_wzry"


def test_wzry_resource_update_confirm_matches_fixture():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["confirm_resource_update_restart"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_resource_update_confirm.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    template = cv2.imread(str(template_dir / node.template), cv2.IMREAD_COLOR)
    assert template is not None
    result = match_template(
        fixture,
        template,
        threshold=node.threshold,
        roi=Rect.from_sequence(node.roi),
        recognition_size=RecognitionSize(width=1280, height=720),
        actual_size=actual_size,
    )

    assert result.found
    assert result.score >= 0.95
    assert node.action == "tap_point"
    assert node.point == [632, 510]
    assert node.next == "wait_game_loading"


def test_wzry_agreement_confirm_matches_fixture_and_stops_without_closing_game():
    project_root = Path(__file__).parents[1]
    task = load_task_definition(project_root / "tasks" / "wzry_farm.yaml")
    node = task.nodes_by_name["confirm_agreement_stop"]
    fixture = cv2.imread(str(project_root / "tests" / "fixtures" / "wzry_agreement_confirm_stop.png"), cv2.IMREAD_COLOR)
    assert fixture is not None

    actual_size = RecognitionSize(width=fixture.shape[1], height=fixture.shape[0])
    template_dir = project_root / "assets" / "templates"
    template = cv2.imread(str(template_dir / node.template), cv2.IMREAD_COLOR)
    assert template is not None
    result = match_template(
        fixture,
        template,
        threshold=node.threshold,
        roi=Rect.from_sequence(node.roi),
        recognition_size=RecognitionSize(width=1280, height=720),
        actual_size=actual_size,
    )

    assert result.found
    assert result.score >= 0.95
    assert node.action == "tap_point"
    assert node.point == [765, 565]
    assert node.next == "stop_keep_game_open"
    assert task.nodes_by_name[node.next].keep_game_open_after_run is True


def test_task_engine_dry_run_completes_without_sending_tap(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template = np.zeros((30, 40, 3), dtype=np.uint8)
    template[5:25, 10:30] = (255, 255, 255)
    template[12:18, 16:24] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "confirm.png"), template)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[205:235, 310:350] = template

    class FakeAdb:
        taps: list[tuple[int, int]]

        def __init__(self):
            self.taps = []

        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "confirm",
            "nodes": [
                {
                    "name": "confirm",
                    "template": "confirm.png",
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop",
                    "timeout_ms": 1000,
                    "retry": 1,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000, "retry": 1},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=True,
    )

    result = engine.run()

    assert result.status == "completed"
    assert result.last_node == "stop"
    assert fake_adb.taps == []


def test_task_engine_auto_detects_later_start_node_from_screenshot(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    farm_template = np.zeros((24, 32, 3), dtype=np.uint8)
    farm_template[4:20, 6:26] = (255, 255, 255)
    farm_template[10:14, 13:19] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "farm.png"), farm_template)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[220:244, 500:532] = farm_template

    class FakeAdb:
        def __init__(self):
            self.keys: list[str] = []
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def keyevent(self, key):
            self.keys.append(str(key))

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "home",
            "auto_detect_start": True,
            "entry_nodes": ["farm_action", "home"],
            "nodes": [
                {
                    "name": "home",
                    "action": "key_sequence",
                    "key_sequence": [{"key": "home"}],
                    "next": "farm_action",
                },
                {
                    "name": "farm_action",
                    "template": "farm.png",
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop",
                    "timeout_ms": 1000,
                    "retry": 1,
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.keys == []
    assert fake_adb.taps == [(516, 232)]


def test_task_engine_rechecks_state_after_repeated_node_visits(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    target_template = np.zeros((24, 32, 3), dtype=np.uint8)
    target_template[4:20, 6:26] = (255, 255, 255)
    target_template[10:14, 13:19] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "target.png"), target_template)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[320:344, 600:632] = target_template

    class FakeAdb:
        def __init__(self):
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "loop",
            "max_steps": 20,
            "stuck_recheck_after": 5,
            "stuck_recheck_nodes": ["target"],
            "nodes": [
                {"name": "loop", "action": "wait", "wait_ms": 1, "next": "loop"},
                {
                    "name": "target",
                    "template": "target.png",
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(616, 332)]


def test_task_engine_rechecks_state_during_repeated_template_attempts(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    wait_template = np.zeros((24, 32, 3), dtype=np.uint8)
    wait_template[4:20, 6:26] = (255, 0, 0)
    wait_template[10:14, 13:19] = (0, 255, 0)
    popup_template = np.zeros((24, 32, 3), dtype=np.uint8)
    popup_template[4:20, 6:26] = (255, 255, 255)
    popup_template[10:14, 13:19] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "wait.png"), wait_template)
    cv2.imwrite(str(template_dir / "popup.png"), popup_template)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[100:124, 200:232] = popup_template

    class FakeAdb:
        def __init__(self):
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "wait_for_farm",
            "fallback": "stop_failed",
            "stuck_recheck_after": 5,
            "stuck_recheck_nodes": ["popup"],
            "nodes": [
                {
                    "name": "wait_for_farm",
                    "template": "wait.png",
                    "threshold": 0.95,
                    "action": "wait",
                    "wait_ms": 1,
                    "retry": 10,
                    "timeout_ms": 5000,
                    "on_fail": "stop_failed",
                },
                {
                    "name": "popup",
                    "template": "popup.png",
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop_success",
                },
                {"name": "stop_success", "action": "stop"},
                {"name": "stop_failed", "action": "stop", "terminal_status": "failed"},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug", screenshot_interval_ms=50),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(216, 112)]


def test_task_engine_auto_detect_start_waits_for_loading_screen_to_settle(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    farm_template = np.zeros((24, 32, 3), dtype=np.uint8)
    farm_template[4:20, 6:26] = (255, 255, 255)
    farm_template[10:14, 13:19] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "farm.png"), farm_template)

    loading = np.zeros((720, 1280, 3), dtype=np.uint8)
    farm = np.zeros((720, 1280, 3), dtype=np.uint8)
    farm[220:244, 500:532] = farm_template

    class FakeAdb:
        def __init__(self):
            self.screens = [loading.copy(), farm.copy(), farm.copy()]
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            if len(self.screens) > 1:
                return self.screens.pop(0)
            return self.screens[0].copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "home",
            "auto_detect_start": True,
            "auto_detect_timeout_ms": 1000,
            "entry_nodes": ["farm_action"],
            "nodes": [
                {"name": "home", "action": "wait", "next": "farm_action"},
                {
                    "name": "farm_action",
                    "template": "farm.png",
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug", screenshot_interval_ms=50),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(516, 232)]


def test_task_engine_clicks_best_match_from_multiple_templates(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    miss = np.zeros((20, 20, 3), dtype=np.uint8)
    miss[:, :10] = (255, 0, 0)
    hit = np.zeros((30, 40, 3), dtype=np.uint8)
    hit[5:25, 10:30] = (255, 255, 255)
    hit[12:18, 16:24] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "miss.png"), miss)
    cv2.imwrite(str(template_dir / "hit.png"), hit)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[205:235, 310:350] = hit

    class FakeAdb:
        def __init__(self):
            self.taps: list[tuple[int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

    task = TaskDefinition.model_validate(
        {
            "start": "collect",
            "nodes": [
                {
                    "name": "collect",
                    "templates": ["miss.png", "hit.png"],
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop",
                    "timeout_ms": 1000,
                    "retry": 1,
                    "post_action_wait_ms": 1,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000, "retry": 1},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=False,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(330, 220)]


def test_task_engine_emits_structured_runtime_events(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template = np.zeros((30, 40, 3), dtype=np.uint8)
    template[5:25, 10:30] = (255, 255, 255)
    template[12:18, 16:24] = (0, 0, 0)
    cv2.imwrite(str(template_dir / "claim.png"), template)
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[205:235, 310:350] = template

    class FakeAdb:
        def screencap_png(self):
            return screen.copy()

        def tap(self, x: int, y: int):
            return ""

    task = TaskDefinition.model_validate(
        {
            "start": "claim",
            "nodes": [
                {
                    "name": "claim",
                    "template": "claim.png",
                    "threshold": 0.95,
                    "action": "click",
                    "next": "stop",
                    "timeout_ms": 1000,
                    "retry": 1,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000, "retry": 1},
            ],
        }
    )
    hub = EventHub()
    events = []
    hub.subscribe(events.append)
    engine = TaskEngine(
        adb=FakeAdb(),  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=False,
        events=hub,
    )

    result = engine.run()

    assert result.status == "completed"
    categories = [event.category for event in events]
    assert "node" in categories
    assert "match" in categories
    assert "action" in categories
    assert "run" in categories
    assert any(event.template == "claim.png" and event.score and event.score > 0.99 for event in events)
    assert any(event.category == "node" and event.data.get("status") == "running" for event in events)
    assert any(event.category == "node" and event.data.get("status") == "success" for event in events)


def test_task_engine_stop_failed_terminal_status_returns_failed(tmp_path: Path):
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)

    class FakeAdb:
        def screencap_png(self):
            return screen.copy()

    task = TaskDefinition.model_validate(
        {
            "start": "stop_failed",
            "nodes": [
                {
                    "name": "stop_failed",
                    "display_name": "失败停止",
                    "action": "stop",
                    "terminal_status": "failed",
                    "timeout_ms": 1000,
                },
            ],
        }
    )
    hub = EventHub()
    events = []
    hub.subscribe(events.append)
    engine = TaskEngine(
        adb=FakeAdb(),  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=task,
        events=hub,
    )

    result = engine.run()

    assert result.status == "failed"
    assert result.last_node == "stop_failed"
    assert any(event.data.get("display_name") == "失败停止" for event in events)


def test_task_definition_parses_key_and_swipe_sequences():
    task = TaskDefinition.model_validate(
        {
            "start": "move",
            "nodes": [
                {
                    "name": "move",
                    "action": "key_sequence",
                    "key_sequence": [
                        {"key": "w", "duration_ms": 250, "interval_ms": 100},
                        {"key": "a", "repeat": 2, "interval_ms": 80},
                    ],
                    "next": "swipe",
                },
                {
                    "name": "swipe",
                    "action": "swipe_sequence",
                    "swipes": [
                        {"start": [180, 620], "end": [180, 520], "duration_ms": 500},
                    ],
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )

    move = task.nodes_by_name["move"]
    swipe = task.nodes_by_name["swipe"]
    assert move.key_sequence is not None
    assert move.key_sequence[0].input_count == 3
    assert move.key_sequence[1].input_count == 2
    assert swipe.swipes is not None
    assert swipe.swipes[0].duration_ms == 500


def test_task_definition_parses_joystick_route():
    task = TaskDefinition.model_validate(
        {
            "start": "move",
            "nodes": [
                {
                    "name": "move",
                    "action": "joystick_route",
                    "joystick": {
                        "center": [188, 616],
                        "distance": 96,
                        "steps": [
                            {"direction": "up", "duration_ms": 900},
                            {"direction": "up_right", "duration_ms": 450, "distance": 72},
                        ],
                    },
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )

    move = task.nodes_by_name["move"]
    assert move.joystick is not None
    assert move.joystick.center == [188, 616]
    assert move.joystick.steps[1].direction == "up_right"


def test_task_definition_parses_min_matches_and_visual_approach():
    task = TaskDefinition.model_validate(
        {
            "start": "farm_loaded",
            "nodes": [
                {
                    "name": "farm_loaded",
                    "templates": ["warehouse.png", "plant.png", "baike.png"],
                    "min_matches": 2,
                    "action": "wait",
                    "next": "approach",
                },
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["statue.png"],
                        "success_templates": ["one_key.png"],
                        "target_roi": [200, 250, 500, 300],
                        "success_roi": [650, 360, 360, 220],
                        "desired_center": [530, 407],
                        "joystick_center": [188, 616],
                        "max_attempts": 3,
                    },
                    "next": "stop",
                },
                {"name": "stop", "action": "stop"},
            ],
        }
    )

    assert task.nodes_by_name["farm_loaded"].min_matches == 2
    approach = task.nodes_by_name["approach"].approach
    assert approach is not None
    assert approach.target_templates == ["statue.png"]
    assert approach.success_templates == ["one_key.png"]


def test_task_definition_parses_route_navigate():
    task = TaskDefinition.model_validate(
        {
            "start": "navigate",
            "nodes": [
                {
                    "name": "navigate",
                    "action": "route_navigate",
                    "route": {
                        "path": "routes/demo.yaml",
                        "max_attempts": 12,
                    },
                }
            ],
        }
    )

    node = task.nodes_by_name["navigate"]
    assert node.route is not None
    assert node.route.path == "routes/demo.yaml"
    assert node.route.max_attempts == 12


def test_task_engine_wait_node_requires_minimum_match_count(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    hit_a = np.zeros((20, 24, 3), dtype=np.uint8)
    hit_a[3:17, 6:18] = (255, 255, 255)
    hit_a[8:12, 10:14] = (0, 0, 0)
    hit_b = np.zeros((18, 22, 3), dtype=np.uint8)
    hit_b[4:14, 5:17] = (0, 255, 255)
    hit_b[7:11, 9:13] = (0, 0, 255)
    miss = np.zeros((18, 22, 3), dtype=np.uint8)
    miss[:, :8] = (255, 0, 0)
    cv2.imwrite(str(template_dir / "hit_a.png"), hit_a)
    cv2.imwrite(str(template_dir / "hit_b.png"), hit_b)
    cv2.imwrite(str(template_dir / "miss.png"), miss)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[100:120, 200:224] = hit_a
    screen[220:238, 350:372] = hit_b

    class FakeAdb:
        def screencap_png(self):
            return screen.copy()

    task = TaskDefinition.model_validate(
        {
            "start": "farm_loaded",
            "nodes": [
                {
                    "name": "farm_loaded",
                    "templates": ["hit_a.png", "hit_b.png", "miss.png"],
                    "min_matches": 2,
                    "threshold": 0.95,
                    "action": "wait",
                    "wait_ms": 1,
                    "next": "stop",
                    "timeout_ms": 1000,
                    "retry": 1,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    engine = TaskEngine(
        adb=FakeAdb(),  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"


def test_task_engine_executes_key_sequence_and_swipe_sequence(tmp_path: Path):
    screen = np.zeros((720, 1280, 3), dtype=np.uint8)

    class FakeAdb:
        def __init__(self):
            self.keys: list[str] = []
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def keyevent(self, key):
            self.keys.append(str(key))

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "move",
            "nodes": [
                {
                    "name": "move",
                    "action": "key_sequence",
                    "key_sequence": [{"key": "w", "repeat": 2, "interval_ms": 1}],
                    "next": "swipe",
                    "timeout_ms": 1000,
                },
                {
                    "name": "swipe",
                    "action": "swipe_sequence",
                    "swipes": [{"start": [100, 600], "end": [120, 500], "duration_ms": 300}],
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=False,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.keys == ["w", "w"]
    assert fake_adb.swipes == [(100, 600, 120, 500, 300)]


def test_task_engine_executes_joystick_route_with_coordinate_mapping(tmp_path: Path):
    screen = np.zeros((1440, 2560, 3), dtype=np.uint8)

    class FakeAdb:
        def __init__(self):
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "move",
            "nodes": [
                {
                    "name": "move",
                    "action": "joystick_route",
                    "joystick": {
                        "center": [100, 600],
                        "distance": 80,
                        "steps": [
                            {"direction": "up", "duration_ms": 300},
                            {"direction": "right", "duration_ms": 200, "distance": 40},
                        ],
                    },
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=False,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.swipes == [
        (200, 1200, 200, 1040, 300),
        (200, 1200, 280, 1200, 200),
    ]


def test_task_engine_visual_approach_moves_until_success_template_appears(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    target = np.zeros((24, 28, 3), dtype=np.uint8)
    target[4:20, 6:22] = (255, 255, 255)
    target[10:14, 12:16] = (0, 0, 0)
    success = np.zeros((22, 26, 3), dtype=np.uint8)
    success[4:18, 5:21] = (0, 255, 0)
    success[9:13, 11:15] = (0, 0, 255)
    cv2.imwrite(str(template_dir / "statue.png"), target)
    cv2.imwrite(str(template_dir / "one_key.png"), success)

    first = np.zeros((720, 1280, 3), dtype=np.uint8)
    first[100:124, 220:248] = target
    second = np.zeros((720, 1280, 3), dtype=np.uint8)
    second[410:432, 760:786] = success

    class FakeAdb:
        def __init__(self):
            self.screens = [first.copy(), second.copy()]
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            if len(self.screens) > 1:
                return self.screens.pop(0)
            return self.screens[0].copy()

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "approach",
            "nodes": [
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["statue.png"],
                        "success_templates": ["one_key.png"],
                        "target_roi": [100, 50, 300, 160],
                        "success_roi": [700, 360, 200, 120],
                        "target_threshold": 0.95,
                        "success_threshold": 0.95,
                        "desired_center": [320, 112],
                        "tolerance": [20, 20],
                        "joystick_center": [100, 600],
                        "joystick_distance": 80,
                        "step_duration_ms": 250,
                        "max_attempts": 3,
                    },
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=False,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.swipes == [(100, 600, 20, 600, 250)]


def test_task_engine_visual_approach_can_use_detected_joystick_center(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    target = np.zeros((24, 28, 3), dtype=np.uint8)
    target[4:20, 6:22] = (255, 255, 255)
    target[10:14, 12:16] = (0, 0, 0)
    success = np.zeros((22, 26, 3), dtype=np.uint8)
    success[4:18, 5:21] = (0, 255, 0)
    success[9:13, 11:15] = (0, 0, 255)
    joystick = np.zeros((30, 30, 3), dtype=np.uint8)
    joystick[5:25, 5:25] = (200, 200, 200)
    joystick[11:19, 11:19] = (40, 40, 40)
    cv2.imwrite(str(template_dir / "statue.png"), target)
    cv2.imwrite(str(template_dir / "one_key.png"), success)
    cv2.imwrite(str(template_dir / "joystick.png"), joystick)

    first = np.zeros((720, 1280, 3), dtype=np.uint8)
    first[100:124, 220:248] = target
    first[585:615, 65:95] = joystick
    second = np.zeros((720, 1280, 3), dtype=np.uint8)
    second[410:432, 760:786] = success

    class FakeAdb:
        def __init__(self):
            self.screens = [first.copy(), second.copy()]
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            if len(self.screens) > 1:
                return self.screens.pop(0)
            return self.screens[0].copy()

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "approach",
            "nodes": [
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["statue.png"],
                        "success_templates": ["one_key.png"],
                        "joystick_templates": ["joystick.png"],
                        "joystick_roi": [0, 520, 180, 190],
                        "target_roi": [100, 50, 300, 160],
                        "success_roi": [700, 360, 200, 120],
                        "target_threshold": 0.95,
                        "success_threshold": 0.95,
                        "joystick_threshold": 0.95,
                        "desired_center": [320, 112],
                        "tolerance": [20, 20],
                        "joystick_center": [100, 600],
                        "joystick_distance": 80,
                        "step_duration_ms": 250,
                        "max_attempts": 2,
                    },
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
        dry_run=False,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.swipes == [(80, 600, 0, 600, 250)]


def test_task_engine_visual_approach_closes_interrupt_popup_before_moving(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    interrupt = np.zeros((24, 32, 3), dtype=np.uint8)
    interrupt[4:20, 6:26] = (255, 255, 255)
    interrupt[10:14, 13:19] = (0, 0, 0)
    target = np.zeros((24, 28, 3), dtype=np.uint8)
    target[4:20, 6:22] = (255, 0, 0)
    target[10:14, 12:16] = (0, 255, 0)
    success = np.zeros((22, 26, 3), dtype=np.uint8)
    success[4:18, 5:21] = (0, 255, 0)
    success[9:13, 11:15] = (0, 0, 255)
    cv2.imwrite(str(template_dir / "close.png"), interrupt)
    cv2.imwrite(str(template_dir / "statue.png"), target)
    cv2.imwrite(str(template_dir / "one_key.png"), success)

    rng = np.random.default_rng(7)
    first = rng.integers(0, 80, size=(720, 1280, 3), dtype=np.uint8)
    first[100:124, 200:232] = interrupt
    second = rng.integers(0, 80, size=(720, 1280, 3), dtype=np.uint8)
    second[410:432, 760:786] = success

    class FakeAdb:
        def __init__(self):
            self.screens = [first.copy(), second.copy()]
            self.taps: list[tuple[int, int]] = []
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            if len(self.screens) > 1:
                return self.screens.pop(0)
            return self.screens[0].copy()

        def tap(self, x: int, y: int):
            self.taps.append((x, y))

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "approach",
            "nodes": [
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["statue.png"],
                        "success_templates": ["one_key.png"],
                        "interrupt_templates": ["close.png"],
                        "interrupt_roi": [100, 60, 200, 120],
                        "success_roi": [700, 360, 200, 120],
                        "target_threshold": 0.95,
                        "success_threshold": 0.95,
                        "interrupt_threshold": 0.95,
                        "interrupt_wait_ms": 0,
                        "joystick_center": [100, 600],
                        "max_attempts": 3,
                    },
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "completed"
    assert fake_adb.taps == [(216, 112)]
    assert fake_adb.swipes == []


def test_task_engine_visual_approach_cycles_fallback_directions_when_target_missing(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    target = np.zeros((24, 28, 3), dtype=np.uint8)
    target[4:20, 6:22] = (255, 255, 255)
    success = np.zeros((22, 26, 3), dtype=np.uint8)
    success[4:18, 5:21] = (0, 255, 0)
    cv2.imwrite(str(template_dir / "statue.png"), target)
    cv2.imwrite(str(template_dir / "one_key.png"), success)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)

    class FakeAdb:
        def __init__(self):
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "approach",
            "nodes": [
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["statue.png"],
                        "success_templates": ["one_key.png"],
                        "target_threshold": 0.95,
                        "success_threshold": 0.95,
                        "joystick_center": [100, 600],
                        "joystick_distance": 80,
                        "step_duration_ms": 250,
                        "step_wait_ms": 0,
                        "max_attempts": 3,
                        "fallback_directions": ["right", "up", "down_left"],
                    },
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "failed"
    assert fake_adb.swipes == [
        (100, 600, 180, 600, 250),
        (100, 600, 100, 520, 250),
        (100, 600, 43, 657, 250),
    ]


def test_task_engine_visual_approach_moves_toward_target_relative_to_actor_center(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    target = np.zeros((24, 28, 3), dtype=np.uint8)
    target[4:20, 6:22] = (255, 255, 255)
    target[10:14, 12:16] = (0, 0, 0)
    success = np.zeros((22, 26, 3), dtype=np.uint8)
    success[4:18, 5:21] = (0, 255, 0)
    success[9:13, 11:15] = (0, 0, 255)
    cv2.imwrite(str(template_dir / "statue.png"), target)
    cv2.imwrite(str(template_dir / "one_key.png"), success)

    screen = np.zeros((720, 1280, 3), dtype=np.uint8)
    screen[408:432, 606:634] = target

    class FakeAdb:
        def __init__(self):
            self.swipes: list[tuple[int, int, int, int, int]] = []

        def screencap_png(self):
            return screen.copy()

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
            self.swipes.append((x1, y1, x2, y2, duration_ms))

    task = TaskDefinition.model_validate(
        {
            "start": "approach",
            "nodes": [
                {
                    "name": "approach",
                    "action": "visual_approach",
                    "approach": {
                        "target_templates": ["statue.png"],
                        "success_templates": ["one_key.png"],
                        "target_threshold": 0.95,
                        "success_threshold": 0.95,
                        "movement_mode": "actor_relative",
                        "actor_center": [500, 500],
                        "tolerance": [20, 20],
                        "joystick_center": [100, 600],
                        "joystick_distance": 80,
                        "step_duration_ms": 250,
                        "step_wait_ms": 0,
                        "max_attempts": 1,
                    },
                    "next": "stop",
                    "timeout_ms": 1000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )
    fake_adb = FakeAdb()
    engine = TaskEngine(
        adb=fake_adb,  # type: ignore[arg-type]
        config=AppConfig(template_dir=template_dir, debug_dir=tmp_path / "debug"),
        task=task,
    )

    result = engine.run()

    assert result.status == "failed"
    assert fake_adb.swipes == [(100, 600, 157, 543, 250)]


def test_task_engine_visual_approach_accepts_nearby_tracking_candidate_below_main_threshold():
    previous = Point(x=202, y=332)
    nearby = MatchResult(
        found=False,
        score=0.444,
        center=Point(x=235, y=340),
        rect=Rect(x=100, y=100, width=20, height=20),
        screen_center=Point(x=235, y=340),
        screen_rect=Rect(x=100, y=100, width=20, height=20),
    )
    far = MatchResult(
        found=False,
        score=0.444,
        center=Point(x=986, y=342),
        rect=Rect(x=900, y=300, width=20, height=20),
        screen_center=Point(x=986, y=342),
        screen_rect=Rect(x=900, y=300, width=20, height=20),
    )

    assert TaskEngine._accept_tracking_candidate(
        nearby,
        last_target=previous,
        tracking_threshold=0.34,
        tracking_radius=180,
    )
    assert not TaskEngine._accept_tracking_candidate(
        far,
        last_target=previous,
        tracking_threshold=0.34,
        tracking_radius=180,
    )


def _build_wait_for_harvest_task() -> TaskDefinition:
    return TaskDefinition.model_validate(
        {
            "start": "wait_for_harvest",
            "nodes": [
                {
                    "name": "wait_for_harvest",
                    "action": "wait_for_harvest",
                    "next": "stop",
                    "timeout_ms": 86400000,
                },
                {"name": "stop", "action": "stop", "timeout_ms": 1000},
            ],
        }
    )


def test_wait_for_harvest_passes_through_without_target(tmp_path: Path):
    import time as _time

    class FakeAdb:
        def screencap_png(self):
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    engine = TaskEngine(
        adb=FakeAdb(),  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=_build_wait_for_harvest_task(),
        harvest_wait_until=None,
    )

    started = _time.perf_counter()
    result = engine.run()
    elapsed = _time.perf_counter() - started

    assert result.status == "completed"
    assert elapsed < 0.5  # 无目标成熟时刻 → 直接放行，不等待


def test_wait_for_harvest_skips_when_target_in_past(tmp_path: Path):
    import time as _time
    from datetime import datetime, timedelta

    class FakeAdb:
        def screencap_png(self):
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    engine = TaskEngine(
        adb=FakeAdb(),  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=_build_wait_for_harvest_task(),
        harvest_wait_until=beijing_now() - timedelta(minutes=1),
    )

    started = _time.perf_counter()
    result = engine.run()
    elapsed = _time.perf_counter() - started

    assert result.status == "completed"
    assert elapsed < 0.5  # 目标已过 → 立即收获，不等待


def test_wait_for_harvest_waits_until_future_target(tmp_path: Path):
    import time as _time
    from datetime import datetime, timedelta

    class FakeAdb:
        def screencap_png(self):
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    hub = EventHub()
    events = []
    hub.subscribe(events.append)
    engine = TaskEngine(
        adb=FakeAdb(),  # type: ignore[arg-type]
        config=AppConfig(debug_dir=tmp_path / "debug"),
        task=_build_wait_for_harvest_task(),
        events=hub,
        harvest_wait_until=beijing_now() + timedelta(milliseconds=400),
    )

    started = _time.perf_counter()
    result = engine.run()
    elapsed = _time.perf_counter() - started

    assert result.status == "completed"
    assert elapsed >= 0.35  # 到成熟时刻前在按钮前等待
    assert any(
        event.action == "wait_for_harvest" and "等待作物成熟" in event.message
        for event in events
    )
