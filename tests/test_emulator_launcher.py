from pathlib import Path

import pytest

from adb_controller import AdbCommandError
from emulator_launcher import ensure_emulator_ready, find_emulator_executable, launcher_command
from runtime_events import EventHub
from settings import AppConfig


def test_find_emulator_executable_accepts_configured_file(tmp_path: Path):
    executable = tmp_path / "MuMuNxMain.exe"
    executable.write_bytes(b"exe")

    assert find_emulator_executable(executable) == executable


def test_find_emulator_executable_searches_mumu_install_directory(tmp_path: Path):
    executable = tmp_path / "x_main" / "MuMuNxMain.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"exe")

    assert find_emulator_executable(tmp_path) == executable


def test_find_emulator_executable_recovers_from_stale_child_path(tmp_path: Path):
    executable = tmp_path / "nx_main" / "MuMuManager.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"exe")
    stale_path = tmp_path / "x_main" / "MuMuNxMain.exe"

    assert find_emulator_executable(stale_path) == executable


def test_launcher_command_uses_mumu_manager_when_available(tmp_path: Path):
    main_executable = tmp_path / "x_main" / "MuMuNxMain.exe"
    manager = tmp_path / "x_main" / "MuMuManager.exe"
    main_executable.parent.mkdir()
    main_executable.write_bytes(b"exe")
    manager.write_bytes(b"exe")

    assert launcher_command(main_executable) == [str(manager), "api", "-v", "1", "launch_player"]


def test_ensure_emulator_ready_does_not_launch_when_adb_probe_succeeds():
    class FakeAdb:
        def __init__(self):
            self.connect_count = 0

        def connect(self, timeout=None):
            self.connect_count += 1
            return "already connected"

        def screencap_png(self, timeout=None):
            return object()

    launches = []
    ensure_emulator_ready(
        config=AppConfig(auto_launch_emulator=True),
        adb=FakeAdb(),  # type: ignore[arg-type]
        launcher=lambda path: launches.append(path),
    )

    assert launches == []


def test_ensure_emulator_ready_launches_and_retries_until_adb_probe_succeeds(tmp_path: Path):
    executable = tmp_path / "MuMuNxMain.exe"
    executable.write_bytes(b"exe")

    class FakeAdb:
        def __init__(self):
            self.probes = 0

        def connect(self, timeout=None):
            self.probes += 1
            if self.probes < 2:
                raise AdbCommandError("offline", command=["adb"])
            return "connected"

        def screencap_png(self, timeout=None):
            return object()

    launches = []
    events = EventHub()
    messages = []
    events.subscribe(lambda event: messages.append(event.message))

    ensure_emulator_ready(
        config=AppConfig(
            emulator_path=executable,
            auto_launch_emulator=True,
            emulator_launch_timeout_sec=5,
        ),
        adb=FakeAdb(),  # type: ignore[arg-type]
        events=events,
        launcher=lambda path: launches.append(path),
        sleep=lambda seconds: None,
    )

    assert launches == [executable]
    assert any("启动 MuMu" in message for message in messages)


def test_ensure_emulator_ready_raises_when_auto_launch_disabled():
    class FakeAdb:
        def connect(self, timeout=None):
            raise AdbCommandError("offline", command=["adb"])

    with pytest.raises(AdbCommandError):
        ensure_emulator_ready(
            config=AppConfig(auto_launch_emulator=False),
            adb=FakeAdb(),  # type: ignore[arg-type]
        )
