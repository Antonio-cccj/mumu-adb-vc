from pathlib import Path

from build_exe import (
    APP_NAME,
    RELEASE_ROOT,
    STAGING_ROOT,
    build_pyinstaller_args,
    copy_runtime_files,
    distribution_files,
    publish_app_files,
    release_dir,
)


def test_build_pyinstaller_args_targets_windowed_client_launcher():
    args = build_pyinstaller_args()

    assert "client_launcher.py" in args
    assert "--windowed" in args
    assert "--onedir" in args
    assert "--name" in args
    assert "MuMuADBVC" in args
    assert "--distpath" in args
    assert args[args.index("--distpath") + 1] == str(STAGING_ROOT)


def test_distribution_files_include_editable_runtime_assets():
    files = distribution_files(Path("C:/work/project"))

    assert files["config.yaml"] == Path("C:/work/project/config.yaml")
    assert files["tasks"] == Path("C:/work/project/tasks")
    assert files["assets"] == Path("C:/work/project/assets")
    assert files["README.md"] == Path("C:/work/project/README.md")


def test_release_dir_is_stable_project_output_location():
    assert release_dir(Path("C:/work/project")) == Path("C:/work/project") / "release" / APP_NAME


def test_copy_runtime_files_creates_runtime_output_directories(tmp_path: Path):
    project_root = tmp_path / "project"
    dist_dir = tmp_path / "release" / APP_NAME
    project_root.mkdir()
    (project_root / "config.yaml").write_text("debug_dir: debug\nlog_dir: logs\n", encoding="utf-8")
    (project_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project_root / "tasks").mkdir()
    (project_root / "assets").mkdir()

    copy_runtime_files(project_root, dist_dir)

    assert (dist_dir / "config.yaml").is_file()
    assert (dist_dir / "README.md").is_file()
    assert (dist_dir / "tasks").is_dir()
    assert (dist_dir / "assets").is_dir()
    assert (dist_dir / "debug").is_dir()
    assert (dist_dir / "logs").is_dir()


def test_publish_app_files_preserves_runtime_dirs_and_replaces_app_files(tmp_path: Path):
    stage_dir = tmp_path / "stage" / APP_NAME
    final_dir = tmp_path / "release" / APP_NAME
    stage_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    (stage_dir / "MuMuADBVC.exe").write_text("new exe", encoding="utf-8")
    (stage_dir / "_internal").mkdir()
    (stage_dir / "_internal" / "library.pyd").write_text("new lib", encoding="utf-8")
    (final_dir / "MuMuADBVC.exe").write_text("old exe", encoding="utf-8")
    (final_dir / "old.txt").write_text("remove me", encoding="utf-8")
    (final_dir / "debug").mkdir()
    (final_dir / "debug" / "snapshot.png").write_text("keep", encoding="utf-8")
    (final_dir / "logs").mkdir()
    (final_dir / "logs" / "run.log").write_text("keep", encoding="utf-8")

    publish_app_files(stage_dir, final_dir)

    assert (final_dir / "MuMuADBVC.exe").read_text(encoding="utf-8") == "new exe"
    assert (final_dir / "_internal" / "library.pyd").read_text(encoding="utf-8") == "new lib"
    assert not (final_dir / "old.txt").exists()
    assert (final_dir / "debug" / "snapshot.png").read_text(encoding="utf-8") == "keep"
    assert (final_dir / "logs" / "run.log").read_text(encoding="utf-8") == "keep"
