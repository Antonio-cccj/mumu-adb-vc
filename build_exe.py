from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "MuMuADBVC"
RELEASE_ROOT = Path("release")
STAGING_ROOT = Path("build") / "release_stage"
SPEC_DIR = Path("build") / "pyinstaller"
RUNTIME_DIRS = ["debug", "logs"]
EXCLUDED_MODULES = [
    "black",
    "IPython",
    "jedi",
    "matplotlib",
    "notebook",
    "pytest",
    "scipy",
]


def build_pyinstaller_args() -> list[str]:
    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--specpath",
        str(SPEC_DIR),
        "--distpath",
        str(STAGING_ROOT),
        "client_launcher.py",
    ]
    for module_name in EXCLUDED_MODULES:
        args.extend(["--exclude-module", module_name])
    return args


def release_dir(project_root: Path) -> Path:
    return project_root / RELEASE_ROOT / APP_NAME


def staging_release_dir(project_root: Path) -> Path:
    return project_root / STAGING_ROOT / APP_NAME


def distribution_files(project_root: Path) -> dict[str, Path]:
    return {
        "config.yaml": project_root / "config.yaml",
        "README.md": project_root / "README.md",
        "tasks": project_root / "tasks",
        "routes": project_root / "routes",
        "assets": project_root / "assets",
    }


def copy_runtime_files(project_root: Path, dist_dir: Path) -> None:
    for relative_name, source in distribution_files(project_root).items():
        target = dist_dir / relative_name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for relative_name in RUNTIME_DIRS:
        (dist_dir / relative_name).mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def publish_app_files(stage_dir: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for target in list(final_dir.iterdir()):
        if target.name in RUNTIME_DIRS:
            continue
        _remove_path(target)
    for source in stage_dir.iterdir():
        target = final_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def build_exe() -> Path:
    project_root = Path(__file__).resolve().parent
    args = [sys.executable, "-m", "PyInstaller", *build_pyinstaller_args()]
    staging_root = project_root / STAGING_ROOT
    if staging_root.exists():
        shutil.rmtree(staging_root)
    subprocess.run(args, cwd=project_root, check=True)
    dist_dir = release_dir(project_root)
    publish_app_files(staging_release_dir(project_root), dist_dir)
    copy_runtime_files(project_root, dist_dir)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    return dist_dir / f"{APP_NAME}.exe"


def main() -> int:
    exe_path = build_exe()
    print(f"Built: {exe_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
