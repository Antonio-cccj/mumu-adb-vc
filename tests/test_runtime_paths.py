import os
import sys
from pathlib import Path

from runtime_paths import app_base_dir, ensure_app_working_dir


def test_app_base_dir_uses_source_directory_when_not_frozen():
    assert app_base_dir() == Path(__file__).parents[1]


def test_app_base_dir_uses_executable_parent_when_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/tools/MuMuADBVC/MuMuADBVC.exe")

    assert app_base_dir() == Path("C:/tools/MuMuADBVC")


def test_ensure_app_working_dir_changes_to_base_dir():
    original_cwd = Path.cwd()
    try:
        base_dir = ensure_app_working_dir()

        assert Path.cwd() == base_dir
    finally:
        os.chdir(original_cwd)
