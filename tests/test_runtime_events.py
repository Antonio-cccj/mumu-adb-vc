import json
from pathlib import Path

from runtime_events import EventHub, EventLevel, RunEvent, RunEventRecorder


def test_run_event_has_stable_ui_text():
    event = RunEvent(
        level=EventLevel.INFO,
        category="match",
        message="识别到领取按钮",
        node="harvest",
        template="claim.png",
        score=0.9234,
    )

    assert "[INFO]" in event.to_display_text()
    assert "识别到领取按钮" in event.to_display_text()
    assert "node=harvest" in event.to_display_text()
    assert "score=0.9234" in event.to_display_text()


def test_event_hub_delivers_events_to_subscribers():
    hub = EventHub()
    received: list[RunEvent] = []
    hub.subscribe(received.append)

    event = hub.emit("INFO", "task", "开始任务", node="start")

    assert received == [event]
    assert event.level == EventLevel.INFO
    assert event.node == "start"


def test_run_event_recorder_writes_jsonl_and_text_log(tmp_path: Path):
    recorder = RunEventRecorder(log_dir=tmp_path, run_name="demo")

    event = recorder.emit("WARN", "adb", "连接失败", data={"serial": "127.0.0.1:16384"})
    recorder.close()

    assert recorder.jsonl_path.exists()
    assert recorder.text_path.exists()
    json_record = json.loads(recorder.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    text_record = recorder.text_path.read_text(encoding="utf-8")
    assert json_record["level"] == "WARN"
    assert json_record["message"] == "连接失败"
    assert json_record["data"]["serial"] == "127.0.0.1:16384"
    assert event.to_display_text() in text_record
