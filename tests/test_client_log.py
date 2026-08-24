from datetime import datetime

from client import (
    client_log_tag_for_event,
    format_client_log_event,
    load_client_state,
    save_client_state,
    should_arm_manual_schedule,
    should_close_game_after_run,
    should_trigger_schedule,
)
from runtime_events import EventLevel, RunEvent
from settings import AppConfig
from task_engine import EngineRunResult, TaskDefinition


def _event(category: str, message: str, **kwargs) -> RunEvent:
    return RunEvent(
        level=EventLevel.INFO,
        category=category,
        message=message,
        timestamp=datetime(2026, 6, 13, 8, 30, 0),
        **kwargs,
    )


def test_client_log_shows_only_current_step_for_running_node():
    event = _event(
        "node",
        "raw engine message",
        node="one_key_farm",
        data={"status": "running", "display_name": "点击一键务农"},
    )

    assert format_client_log_event(event) == "08:30:00 当前步骤：点击一键务农"


def test_client_log_hides_verbose_match_events():
    event = _event("match", "模板匹配完成", node="one_key_farm", score=0.99)

    assert format_client_log_event(event) is None


def test_client_log_shows_maturity_and_schedule_events():
    maturity = _event("maturity", "作物成熟时间：13:04")
    schedule = _event("schedule", "下次定时启动：12:54")

    assert format_client_log_event(maturity) == "08:30:00 作物成熟时间：13:04"
    assert format_client_log_event(schedule) == "08:30:00 下次定时启动：12:54"


def test_client_state_round_trips_task_options(tmp_path):
    state_path = tmp_path / "client_state.yaml"
    save_client_state(
        {
            "wake_on_launch": True,
            "farm_task_enabled": True,
            "schedule_enabled": False,
        },
        state_path,
    )

    assert load_client_state(state_path) == {
        "wake_on_launch": True,
        "farm_task_enabled": True,
        "schedule_enabled": False,
    }


def test_client_log_tags_successful_node_as_green():
    event = _event(
        "node",
        "raw engine message",
        node="one_key_farm",
        data={"status": "success", "display_name": "one key farm"},
    )

    assert client_log_tag_for_event(event) == "success"


def test_client_log_tags_failed_node_as_red():
    event = _event(
        "node",
        "raw engine message",
        node="one_key_farm",
        data={"status": "failed", "display_name": "one key farm"},
    )

    assert client_log_tag_for_event(event) == "error"


def test_client_log_tags_error_event_as_red():
    event = RunEvent(
        level=EventLevel.ERROR,
        category="run",
        message="failed",
        timestamp=datetime(2026, 6, 13, 8, 30, 0),
    )

    assert client_log_tag_for_event(event) == "error"


def test_manual_schedule_can_be_armed_from_text_without_apply_button():
    assert should_arm_manual_schedule(
        schedule_enabled=True,
        next_run_at=None,
        custom_next_start="12:54",
    ) is True


def test_schedule_triggers_when_due_and_worker_idle():
    now = datetime(2026, 6, 16, 12, 54, 0)

    assert should_trigger_schedule(
        now=now,
        next_run_at=datetime(2026, 6, 16, 12, 54, 0),
        worker_alive=False,
    ) is True


def test_client_keeps_game_open_when_terminal_node_requests_it():
    task = TaskDefinition.model_validate(
        {
            "start": "stop_keep_game_open",
            "nodes": [
                {
                    "name": "stop_keep_game_open",
                    "action": "stop",
                    "terminal_status": "stopped",
                    "keep_game_open_after_run": True,
                }
            ],
        }
    )
    result = EngineRunResult(
        status="stopped",
        steps=1,
        last_node="stop_keep_game_open",
        reason="Stop node reached",
    )

    assert (
        should_close_game_after_run(
            config=AppConfig(close_game_after_run=True),
            task=task,
            result=result,
        )
        is False
    )


def test_client_closes_game_for_normal_finished_run_when_enabled():
    task = TaskDefinition.model_validate(
        {
            "start": "stop_success",
            "nodes": [{"name": "stop_success", "action": "stop"}],
        }
    )
    result = EngineRunResult(
        status="completed",
        steps=1,
        last_node="stop_success",
        reason="Stop node reached",
    )

    assert (
        should_close_game_after_run(
            config=AppConfig(close_game_after_run=True),
            task=task,
            result=result,
        )
        is True
    )


def test_client_respects_global_close_game_disabled():
    assert (
        should_close_game_after_run(
            config=AppConfig(close_game_after_run=False),
            task=None,
            result=None,
        )
        is False
    )
