from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from adb_controller import AdbController, AdbError
from emulator_launcher import ensure_emulator_ready
from runtime_events import RunEventRecorder
from runtime_paths import ensure_app_working_dir
from settings import AppConfig, load_config
from task_engine import TaskEngine, load_task_definition
from task_preflight import format_missing_template_groups, missing_template_groups
from vision import RecognitionSize, Rect, match_template, resize_to_recognition


console = Console()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=False)],
    )


def build_controller(config: AppConfig) -> AdbController:
    return AdbController(
        adb_path=config.adb_path,
        device_serial=config.device_serial,
        default_timeout=config.command_timeout_sec,
        logger=logging.getLogger("adb"),
    )


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def command_devices(config: AppConfig) -> int:
    adb = build_controller(config)
    devices = adb.list_devices()
    table = Table(title="ADB Devices")
    table.add_column("Serial")
    table.add_column("State")
    table.add_column("Description")
    for device in devices:
        table.add_row(device.serial, device.state, device.description)
    if not devices:
        table.add_row("-", "none", "No adb devices reported")
    console.print(table)
    return 0


def command_capture(config: AppConfig, args: argparse.Namespace) -> int:
    adb = build_controller(config)
    adb.connect()
    screenshot = adb.screencap_png()
    output = Path(args.output) if args.output else Path(config.debug_dir) / f"capture_{timestamp()}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image = resize_to_recognition(screenshot, config.recognition_size) if args.recognition else screenshot
    cv2.imwrite(str(output), image)
    console.print(f"Saved screenshot: [bold]{output}[/bold] ({image.shape[1]}x{image.shape[0]})")
    return 0


def command_match(config: AppConfig, args: argparse.Namespace) -> int:
    adb = build_controller(config)
    adb.connect()
    screenshot = adb.screencap_png()
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        console.print(f"[red]Template not found or unreadable:[/red] {args.template}")
        return 2

    roi = Rect.from_sequence(args.roi)
    debug_path = Path(config.debug_dir) / f"match_{timestamp()}.png"
    actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
    result = match_template(
        screenshot,
        template,
        threshold=args.threshold,
        roi=roi,
        recognition_size=config.recognition_size,
        actual_size=actual_size,
        debug_path=debug_path,
    )
    table = Table(title="Template Match")
    table.add_column("Found")
    table.add_column("Score")
    table.add_column("Recognition Center")
    table.add_column("Screen Center")
    table.add_column("Recognition Rect")
    table.add_row(
        str(result.found),
        f"{result.score:.4f}",
        f"({result.center.x}, {result.center.y})",
        f"({result.screen_center.x}, {result.screen_center.y})",
        f"({result.rect.x}, {result.rect.y}, {result.rect.width}, {result.rect.height})",
    )
    console.print(table)
    console.print(f"Debug image: [bold]{debug_path}[/bold]")
    return 0 if result.found else 1


def start_manual_control(engine: TaskEngine) -> threading.Thread | None:
    if not sys.stdin.isatty():
        return None

    def control_loop() -> None:
        console.print("Controls: p=pause/resume, r=resume, s=stop, Ctrl+C=stop")
        while not engine.stopped:
            try:
                command = input().strip().lower()
            except EOFError:
                return
            if command in {"p", "pause"}:
                engine.toggle_pause()
            elif command in {"r", "resume"}:
                engine.resume()
            elif command in {"s", "stop", "q", "quit"}:
                engine.request_stop()
                return

    thread = threading.Thread(target=control_loop, name="manual-control", daemon=True)
    thread.start()
    return thread


def command_run(config: AppConfig, args: argparse.Namespace) -> int:
    task_path = Path(args.task)
    recorder = RunEventRecorder(log_dir=config.log_dir, run_name=task_path.stem)
    console.print(f"Run log: [bold]{recorder.text_path}[/bold]")
    try:
        recorder.emit("INFO", "run", "准备启动任务", data={"task": str(task_path)})
        task = load_task_definition(task_path)
        missing_groups = missing_template_groups(task, config.template_dir)
        if missing_groups:
            message = format_missing_template_groups(missing_groups)
            recorder.emit("ERROR", "preflight", message)
            console.print(f"[red]{message}[/red]")
            return 2
        adb = build_controller(config)
        recorder.emit(
            "INFO",
            "emulator",
            "准备 MuMu/ADB",
            data={
                "serial": config.device_serial,
                "adb_path": config.adb_path,
                "emulator_path": str(config.emulator_path or ""),
            },
        )
        ensure_emulator_ready(config=config, adb=adb, events=recorder)
        engine = TaskEngine(
            adb=adb,
            config=config,
            task=task,
            dry_run=False,
            logger=logging.getLogger("task"),
            events=recorder,
        )
        start_manual_control(engine)
        try:
            result = engine.run()
        except KeyboardInterrupt:
            engine.request_stop()
            recorder.emit("WARN", "run", "用户通过 Ctrl+C 停止")
            console.print("[yellow]Stop requested by Ctrl+C.[/yellow]")
            return 130

        recorder.emit(
            "INFO" if result.status == "completed" else "WARN",
            "run",
            f"任务结束：{result.status}",
            node=result.last_node,
            data={"steps": result.steps, "reason": result.reason},
        )
        color = "green" if result.status == "completed" else "yellow" if result.status in {"stopped", "max_steps"} else "red"
        console.print(
            f"[{color}]Run {result.status}[/{color}]: steps={result.steps}, "
            f"last_node={result.last_node}, reason={result.reason}"
        )
        return 0 if result.status in {"completed", "stopped", "max_steps"} else 1
    except Exception as exc:
        recorder.emit("ERROR", "run", f"任务异常：{exc}")
        raise
    finally:
        console.print(f"Structured log: [bold]{recorder.jsonl_path}[/bold]")
        recorder.close()


def command_client() -> int:
    from client import main as client_main

    ensure_app_working_dir()
    client_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local MuMu emulator UI automation MVP using ADB screenshots.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("devices", help="List adb devices")

    capture = subparsers.add_parser("capture", help="Capture current emulator screenshot")
    capture.add_argument("--output", help="Output PNG path")
    capture.add_argument("--recognition", action="store_true", help="Save resized 1280x720 recognition image")

    match = subparsers.add_parser("match", help="Match a template against current screenshot")
    match.add_argument("--template", required=True, help="Template image path")
    match.add_argument("--threshold", type=float, default=0.8, help="Template match threshold")
    match.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="ROI in recognition coordinates")

    run = subparsers.add_parser("run", help="Run a task YAML pipeline")
    run.add_argument("--task", required=True, help="Task YAML path")

    subparsers.add_parser("client", help="Launch the local desktop client")

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "devices":
            return command_devices(config)
        if args.command == "capture":
            return command_capture(config, args)
        if args.command == "match":
            return command_match(config, args)
        if args.command == "run":
            return command_run(config, args)
        if args.command == "client":
            return command_client()
    except (AdbError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
