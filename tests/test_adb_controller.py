import cv2
import numpy as np

from adb_controller import (
    AdbController,
    decode_screencap_png,
    hidden_subprocess_kwargs,
    is_adb_connect_failure,
    normalize_keycode,
)


def test_decode_screencap_png_repairs_crlf_polluted_stream():
    original = np.zeros((12, 16, 3), dtype=np.uint8)
    original[2:8, 3:10] = (10, 120, 240)
    ok, encoded = cv2.imencode(".png", original)
    assert ok

    polluted = encoded.tobytes().replace(b"\n", b"\r\n")
    decoded = decode_screencap_png(polluted)

    assert decoded.shape == original.shape
    assert decoded[4, 5].tolist() == [10, 120, 240]


def test_is_adb_connect_failure_detects_text_failures():
    assert is_adb_connect_failure("cannot connect to 127.0.0.1:16384")
    assert is_adb_connect_failure("failed to connect to 127.0.0.1:16384")
    assert not is_adb_connect_failure("connected to 127.0.0.1:16384")
    assert not is_adb_connect_failure("already connected to 127.0.0.1:16384")


def test_normalize_keycode_accepts_aliases_and_numeric_codes():
    assert normalize_keycode("home") == "KEYCODE_HOME"
    assert normalize_keycode("w") == "KEYCODE_W"
    assert normalize_keycode("KEYCODE_D") == "KEYCODE_D"
    assert normalize_keycode(51) == "51"


def test_hidden_subprocess_kwargs_hide_windows_console():
    kwargs = hidden_subprocess_kwargs(timeout=3.0)

    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 3.0
    assert kwargs["creationflags"] != 0
    assert kwargs["startupinfo"] is not None


def test_adb_run_uses_hidden_subprocess_kwargs(monkeypatch):
    captured = {}

    class Completed:
        returncode = 0
        stdout = b"ok"
        stderr = b""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr("adb_controller.subprocess.run", fake_run)
    controller = AdbController(adb_path="python", default_timeout=7.0)

    assert controller._run(["version"]) == "ok"
    assert captured["kwargs"]["creationflags"] != 0
    assert captured["kwargs"]["startupinfo"] is not None


def test_force_stop_package_uses_standard_android_command(monkeypatch):
    captured = {}

    def fake_run(self, args, *, serial=None, timeout=None, binary=False):
        captured["args"] = args
        captured["serial"] = serial
        captured["timeout"] = timeout
        captured["binary"] = binary
        return ""

    monkeypatch.setattr(AdbController, "_run", fake_run)
    controller = AdbController(adb_path="python", device_serial="127.0.0.1:16384", default_timeout=7.0)

    controller.force_stop_package("com.tencent.tmgp.sgame", timeout=3.0)

    assert captured == {
        "args": ["shell", "am", "force-stop", "com.tencent.tmgp.sgame"],
        "serial": "127.0.0.1:16384",
        "timeout": 3.0,
        "binary": False,
    }
