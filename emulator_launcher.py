from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from adb_controller import AdbCommandError, AdbController
from runtime_events import EventHub
from settings import AppConfig


Launcher = Callable[[Path], object]
Sleeper = Callable[[float], None]


DEFAULT_EMULATOR_CANDIDATES = [
    Path("C:/Program Files/NetEase/MuMu Player 12/nx_main/MuMuManager.exe"),
    Path("C:/Program Files/NetEase/MuMu Player 12/nx_main/MuMuNxMain.exe"),
    Path("C:/Program Files/NetEase/MuMu Player 12/x_main/MuMuNxMain.exe"),
    Path("C:/Program Files/NetEase/MuMu Player 12/nx_device/12.0/shell/NemuShell.exe"),
    Path("C:/Program Files/NetEase/MuMu Player 12/shell/NemuShell.exe"),
    Path("C:/Program Files/Netease/MuMu Player 12/nx_main/MuMuManager.exe"),
    Path("C:/Program Files/Netease/MuMu Player 12/nx_main/MuMuNxMain.exe"),
    Path("C:/Program Files/Netease/MuMu Player 12/x_main/MuMuNxMain.exe"),
    Path("C:/Program Files/Netease/MuMu Player 12/nx_device/12.0/shell/NemuShell.exe"),
    Path("C:/Program Files/Netease/MuMu Player 12/shell/NemuShell.exe"),
]

INSTALL_DIR_RELATIVE_CANDIDATES = [
    Path("nx_main/MuMuManager.exe"),
    Path("nx_main/MuMuNxMain.exe"),
    Path("x_main/MuMuNxMain.exe"),
    Path("nx_device/12.0/shell/NemuShell.exe"),
    Path("shell/NemuShell.exe"),
]


def _normalize_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def find_emulator_executable(configured_path: str | Path | None = None) -> Path:
    if configured_path:
        configured = _normalize_path(configured_path)
        if configured.is_file():
            return configured
        if configured.is_dir():
            found = _find_in_install_dir(configured)
            if found:
                return found
        for parent in [configured.parent, configured.parent.parent]:
            if parent.exists() and parent.is_dir():
                found = _find_in_install_dir(parent)
                if found:
                    return found
        raise FileNotFoundError(f"MuMu executable not found: {configured}")

    for candidate in DEFAULT_EMULATOR_CANDIDATES:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("MuMu executable not found. Set emulator_path in config.yaml.")


def _find_in_install_dir(directory: Path) -> Path | None:
    for relative in INSTALL_DIR_RELATIVE_CANDIDATES:
        candidate = directory / relative
        if candidate.is_file():
            return candidate
    return None


def launch_emulator(executable: Path) -> subprocess.Popen[bytes]:
    command = launcher_command(executable)
    kwargs: dict[str, object] = {
        "cwd": str(executable.parent),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(command, **kwargs)


def launcher_command(executable: Path) -> list[str]:
    manager = executable if executable.name.lower() == "mumumanager.exe" else executable.parent / "MuMuManager.exe"
    if manager.is_file():
        return [str(manager), "api", "-v", "1", "launch_player"]
    return [str(executable)]


def _probe_adb_screenshot(adb: AdbController, timeout: float) -> None:
    adb.connect(timeout=timeout)
    adb.screencap_png(timeout=timeout)


def ensure_emulator_ready(
    *,
    config: AppConfig,
    adb: AdbController,
    events: EventHub | None = None,
    launcher: Launcher = launch_emulator,
    sleep: Sleeper = time.sleep,
) -> None:
    probe_timeout = min(max(config.command_timeout_sec, 1.0), 8.0)
    try:
        _probe_adb_screenshot(adb, timeout=probe_timeout)
        if events:
            events.emit("INFO", "emulator", "MuMu/ADB 已就绪")
        return
    except Exception as first_error:
        if not config.auto_launch_emulator:
            raise first_error
        last_error: Exception = first_error

    executable = find_emulator_executable(config.emulator_path)
    if events:
        events.emit("INFO", "emulator", f"启动 MuMu：{executable}")
    launcher(executable)

    deadline = time.monotonic() + config.emulator_launch_timeout_sec
    while time.monotonic() < deadline:
        sleep(3.0)
        try:
            _probe_adb_screenshot(adb, timeout=probe_timeout)
            if events:
                events.emit("INFO", "emulator", "MuMu/ADB 已就绪")
            return
        except Exception as error:
            last_error = error
            if events:
                events.emit("DEBUG", "emulator", f"等待 MuMu ADB 可截图：{error}")

    raise AdbCommandError(
        f"MuMu did not become ready within {config.emulator_launch_timeout_sec:.0f}s: {last_error}",
        command=[str(executable)],
        stderr=str(last_error),
    )
