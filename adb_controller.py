from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vision import RecognitionSize


class AdbError(RuntimeError):
    pass


class AdbNotFoundError(AdbError):
    pass


class AdbCommandError(AdbError):
    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str],
        returncode: int | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr


class AdbScreenshotDecodeError(AdbError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    state: str
    description: str = ""


def find_adb_path(configured_path: str | None = None) -> str:
    if configured_path:
        expanded = Path(os.path.expandvars(configured_path)).expanduser()
        if expanded.is_file():
            return str(expanded)
        resolved = shutil.which(configured_path)
        if resolved:
            return resolved
        raise AdbNotFoundError(f"Configured adb_path does not exist: {configured_path}")

    resolved = shutil.which("adb")
    if resolved:
        return resolved

    candidates = [
        Path("C:/Program Files/NetEase/MuMu Player 12/nx_device/12.0/shell/adb.exe"),
        Path("C:/Program Files/Netease/MuMu Player 12/shell/adb.exe"),
        Path("C:/Program Files/Netease/MuMuPlayer-12.0/shell/adb.exe"),
        Path("C:/Program Files (x86)/Netease/MuMu/shell/adb.exe"),
        Path("C:/Android/platform-tools/adb.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise AdbNotFoundError("Could not find adb. Set adb_path in config.yaml or add adb to PATH.")


def decode_screencap_png(raw_png: bytes) -> np.ndarray:
    candidates = [
        raw_png,
        raw_png.replace(b"\r\n", b"\n"),
        raw_png.replace(b"\r\r\n", b"\n"),
    ]
    for candidate in candidates:
        data = np.frombuffer(candidate, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise AdbScreenshotDecodeError("Failed to decode PNG data returned by adb screencap")


def is_adb_connect_failure(output: str) -> bool:
    lowered = output.lower()
    failure_markers = [
        "cannot connect",
        "failed to connect",
        "unable to connect",
        "connection refused",
        "actively refused",
    ]
    return any(marker in lowered for marker in failure_markers)


KEYCODE_ALIASES: dict[str, str] = {
    "home": "HOME",
    "back": "BACK",
    "enter": "ENTER",
    "w": "W",
    "a": "A",
    "s": "S",
    "d": "D",
    "up": "DPAD_UP",
    "down": "DPAD_DOWN",
    "left": "DPAD_LEFT",
    "right": "DPAD_RIGHT",
}


def normalize_keycode(key: str | int) -> str:
    if isinstance(key, int):
        return str(key)
    normalized = key.strip().lower()
    if not normalized:
        raise ValueError("Key must not be empty")
    return "KEYCODE_" + KEYCODE_ALIASES.get(normalized, normalized.upper()).removeprefix("KEYCODE_")


def hidden_subprocess_kwargs(timeout: float) -> dict:
    kwargs = {
        "capture_output": True,
        "check": False,
        "timeout": timeout,
    }
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["startupinfo"] = None
        kwargs["creationflags"] = 0
    return kwargs


def _escape_input_text(text: str) -> str:
    replacements = {
        " ": "%s",
        "\\": "\\\\",
        "&": "\\&",
        "|": "\\|",
        "<": "\\<",
        ">": "\\>",
        ";": "\\;",
        "(": "\\(",
        ")": "\\)",
        "$": "\\$",
        "`": "\\`",
        '"': '\\"',
        "'": "\\'",
    }
    return "".join(replacements.get(char, char) for char in text)


class AdbController:
    def __init__(
        self,
        *,
        adb_path: str | None = None,
        device_serial: str | None = None,
        default_timeout: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.adb_path = find_adb_path(adb_path)
        self.device_serial = device_serial
        self.default_timeout = default_timeout
        self.logger = logger or logging.getLogger(__name__)

    def connect(self, serial: str | None = None, timeout: float | None = None) -> str:
        target = serial or self.device_serial
        if not target:
            self.logger.info("No device_serial configured; skipping adb connect.")
            return ""
        if ":" not in target:
            self.logger.info("Using non-TCP adb serial %s; skipping adb connect.", target)
            return ""
        output = self._run(["connect", target], timeout=timeout)
        if is_adb_connect_failure(output):
            self.logger.error("adb connect %s failed: %s", target, output.strip())
            raise AdbCommandError(
                f"ADB connect failed for {target}: {output.strip()}",
                command=[self.adb_path, "connect", target],
                stderr=output,
            )
        self.logger.info("adb connect %s: %s", target, output.strip())
        return output

    def list_devices(self, timeout: float | None = None) -> list[DeviceInfo]:
        output = self._run(["devices", "-l"], timeout=timeout)
        devices: list[DeviceInfo] = []
        for line in output.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=2)
            serial = parts[0]
            state = parts[1] if len(parts) > 1 else "unknown"
            description = parts[2] if len(parts) > 2 else ""
            devices.append(DeviceInfo(serial=serial, state=state, description=description))
        return devices

    def screencap_png(self, timeout: float | None = None) -> np.ndarray:
        raw = self._run(
            ["exec-out", "screencap", "-p"],
            serial=self.device_serial,
            timeout=timeout,
            binary=True,
        )
        assert isinstance(raw, bytes)
        return decode_screencap_png(raw)

    def tap(self, x: int, y: int, timeout: float | None = None) -> str:
        return self._run(["shell", "input", "tap", str(int(x)), str(int(y))], serial=self.device_serial, timeout=timeout)

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 500,
        timeout: float | None = None,
    ) -> str:
        return self._run(
            [
                "shell",
                "input",
                "swipe",
                str(int(x1)),
                str(int(y1)),
                str(int(x2)),
                str(int(y2)),
                str(int(duration_ms)),
            ],
            serial=self.device_serial,
            timeout=timeout,
        )

    def input_text(self, text: str, timeout: float | None = None) -> str:
        return self._run(
            ["shell", "input", "text", _escape_input_text(text)],
            serial=self.device_serial,
            timeout=timeout,
        )

    def keyevent(self, key: str | int, timeout: float | None = None) -> str:
        return self._run(
            ["shell", "input", "keyevent", normalize_keycode(key)],
            serial=self.device_serial,
            timeout=timeout,
        )

    def force_stop_package(self, package_name: str, timeout: float | None = None) -> str:
        return self._run(
            ["shell", "am", "force-stop", package_name],
            serial=self.device_serial,
            timeout=timeout,
        )

    def get_screen_size(self, timeout: float | None = None) -> RecognitionSize:
        output = self._run(["shell", "wm", "size"], serial=self.device_serial, timeout=timeout)
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            screenshot = self.screencap_png(timeout=timeout)
            return RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
        width, height = matches[-1]
        return RecognitionSize(width=int(width), height=int(height))

    def _run(
        self,
        args: Sequence[str],
        *,
        serial: str | None = None,
        timeout: float | None = None,
        binary: bool = False,
    ) -> str | bytes:
        command = [self.adb_path]
        if serial:
            command.extend(["-s", serial])
        command.extend(args)
        effective_timeout = timeout or self.default_timeout
        try:
            completed = subprocess.run(
                command,
                **hidden_subprocess_kwargs(effective_timeout),
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.error("ADB command timed out after %.1fs: %s", effective_timeout, " ".join(command))
            raise AdbCommandError(
                f"ADB command timed out after {effective_timeout:.1f}s",
                command=command,
            ) from exc
        except OSError as exc:
            self.logger.error("ADB command failed to start: %s", " ".join(command))
            raise AdbCommandError(
                f"ADB command failed to start: {exc}",
                command=command,
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            stdout = completed.stdout.decode("utf-8", errors="replace")
            details = stderr.strip() or stdout.strip() or f"exit code {completed.returncode}"
            self.logger.error("ADB command failed: %s\n%s", " ".join(command), details)
            raise AdbCommandError(
                f"ADB command failed: {details}",
                command=command,
                returncode=completed.returncode,
                stderr=stderr,
            )
        if binary:
            return completed.stdout
        return completed.stdout.decode("utf-8", errors="replace")
