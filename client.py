from __future__ import annotations

import os
import queue
import threading
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, messagebox
from tkinter import ttk
import tkinter as tk

import cv2
import yaml

from adb_controller import AdbController
from cache_manager import clear_screenshot_cache
from emulator_launcher import ensure_emulator_ready
from runtime_events import RunEvent, RunEventRecorder
from run_scheduler import (
    build_crop_optimization_plan,
    compute_crop_next_run,
    compute_manual_next_run,
    compute_next_run,
    format_crop_optimization_report,
    format_duration,
    remaining_until_clock_time,
)
from settings import AppConfig, load_config
from task_engine import EngineRunResult, TaskDefinition, TaskEngine, load_task_definition
from task_preflight import format_missing_template_groups, missing_template_groups
from template_capture import CAPTURE_STAGES, capture_recognition_source, stage_id_from_display
from time_utils import beijing_now
from vision import resize_to_recognition


CLIENT_STATE_PATH = Path("client_state.yaml")
MANUAL_NEXT_RUN_STATE_KEY = "manual_next_run_at"


def load_client_state(path: Path = CLIENT_STATE_PATH) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def save_client_state(state: dict[str, object], path: Path = CLIENT_STATE_PATH) -> None:
    path.write_text(yaml.safe_dump(state, allow_unicode=True, sort_keys=True), encoding="utf-8")


def format_client_log_event(event: RunEvent) -> str | None:
    time_text = event.timestamp.strftime("%H:%M:%S")
    if event.category == "node":
        status = str(event.data.get("status") or "").lower()
        display_name = str(event.data.get("display_name") or event.node or "")
        if status == "running":
            return f"{time_text} 当前步骤：{display_name}"
        if status in {"success", "completed"}:
            return f"{time_text} 完成：{display_name}"
        if status == "failed":
            return f"{time_text} 失败：{display_name}"
        if status == "stopped":
            return f"{time_text} 已停止：{display_name}"
        return None

    if event.category in {"match", "action", "capture", "template", "route", "state"}:
        if event.level.value not in {"WARN", "ERROR"}:
            return None

    if event.category in {"run", "emulator", "preflight", "maturity", "schedule", "cache"} or event.level.value in {"WARN", "ERROR"}:
        return f"{time_text} {event.message}"
    return None


def client_log_tag_for_event(event: RunEvent) -> str:
    if event.level.value == "ERROR":
        return "error"
    if event.level.value == "WARN":
        return "warning"
    if event.category == "node":
        status = str(event.data.get("status") or "").lower()
        if status in {"success", "completed"}:
            return "success"
        if status in {"failed", "stopped"}:
            return "error"
    if event.category == "run":
        status = str(event.data.get("status") or event.data.get("terminal_status") or "").lower()
        if status == "completed":
            return "success"
        if status in {"failed", "stopped"}:
            return "error"
    if event.category == "cache" and event.level.value == "INFO":
        return "success"
    return "normal"


def _coerce_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _coerce_timedelta_minutes(value: object) -> timedelta | None:
    if isinstance(value, timedelta):
        return value
    if isinstance(value, int | float):
        return timedelta(minutes=float(value))
    if isinstance(value, str):
        try:
            return timedelta(minutes=float(value))
        except ValueError:
            return None
    return None


def _duration_text(value: timedelta) -> str:
    return format_duration(value)


def _bytes_text(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / 1024 / 1024:.1f} MB"
    if value >= 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value} B"


def should_arm_manual_schedule(
    *,
    schedule_enabled: bool,
    next_run_at: datetime | None,
    custom_next_start: str,
) -> bool:
    return schedule_enabled and next_run_at is None and bool(custom_next_start.strip())


def should_trigger_schedule(
    *,
    now: datetime,
    next_run_at: datetime | None,
    worker_alive: bool,
) -> bool:
    return next_run_at is not None and now >= next_run_at and not worker_alive


def should_close_game_after_run(
    *,
    config: AppConfig,
    task: TaskDefinition | None,
    result: EngineRunResult | None,
) -> bool:
    if not config.close_game_after_run:
        return False
    if task is not None and result is not None and result.last_node:
        node = task.nodes_by_name.get(result.last_node)
        if node is not None and node.keep_game_open_after_run:
            return False
    return True


class AutomationClient(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MuMu ADB VC")
        self.geometry("1320x760")
        self.minsize(1120, 660)

        self.config_model = load_config()
        self.client_state = load_client_state()
        self.event_queue: queue.Queue[RunEvent | tuple[str, object]] = queue.Queue()
        self.engine: TaskEngine | None = None
        self.worker: threading.Thread | None = None
        self.paused = False

        self.wake_enabled = BooleanVar(value=bool(self.client_state.get("wake_on_launch", False)))
        self.task_enabled = BooleanVar(value=bool(self.client_state.get("farm_task_enabled", True)))
        self.reward_enabled = BooleanVar(value=False)
        self.steal_enabled = BooleanVar(value=False)
        self.schedule_enabled = BooleanVar(value=bool(self.client_state.get("schedule_enabled", False)))
        self.serial = StringVar(value=self.config_model.device_serial or "")
        self.adb_path = StringVar(value=self.config_model.adb_path or "")
        self.emulator_path = StringVar(value=str(self.config_model.emulator_path or ""))
        self.template_stage = StringVar(value=CAPTURE_STAGES[0].display_text)
        self.status = StringVar(value="就绪")
        self.current_step = StringVar(value="当前步骤：未开始")
        self.schedule_status = StringVar(value="下次启动：未开启")
        self.custom_next_start = StringVar(value="")
        self.crop_type = StringVar(value=str(self.client_state.get("crop_type", "32h")))
        self.settings_title = StringVar(value="一键务农设置")
        self.settings_hint = StringVar(value="启动游戏、关弹窗、进农场、移动到互动点、点击一键务农并读取成熟时间。")
        self.next_run_at: datetime | None = (
            _coerce_datetime(self.client_state.get(MANUAL_NEXT_RUN_STATE_KEY))
            if self.schedule_enabled.get()
            else None
        )
        self.progress_items: dict[str, str] = {}
        self._snapshot_link_counter = 0

        self._build_ui()
        self._load_progress_steps()
        self._attach_state_traces()
        self._poll_events()
        self._poll_schedule()
        self.after(800, self._auto_start_if_enabled)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=2)
        self.rowconfigure(0, weight=1)

        task_frame = ttk.LabelFrame(self, text="一键任务", padding=12)
        task_frame.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        task_frame.columnconfigure(0, weight=1)

        task_list = ttk.Frame(task_frame)
        task_list.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        task_list.columnconfigure(0, weight=1)
        self._add_task_option_row(
            task_list,
            0,
            variable=self.wake_enabled,
            title="开始唤醒",
            description="打开客户端后自动执行已勾选任务。",
            settings_key="wake",
        )
        self._add_task_option_row(
            task_list,
            1,
            variable=self.task_enabled,
            title="一键务农",
            description="启动游戏、关弹窗、进农场、点击一键务农。",
            settings_key="farm",
        )
        self._add_task_option_row(
            task_list,
            2,
            variable=self.reward_enabled,
            title="领取奖励",
            description="预留功能，后续接入。",
            settings_key="reward",
            enabled=False,
        )
        self._add_task_option_row(
            task_list,
            3,
            variable=self.steal_enabled,
            title="自动偷菜",
            description="预留功能，后续接入。",
            settings_key="steal",
            enabled=False,
        )

        ttk.Label(task_frame, textvariable=self.current_step, foreground="#1666b1", wraplength=260).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )
        self.progress_tree = ttk.Treeview(
            task_frame,
            columns=("status", "step"),
            show="headings",
            height=12,
        )
        self.progress_tree.heading("status", text="状态")
        self.progress_tree.heading("step", text="流程")
        self.progress_tree.column("status", width=64, anchor="center", stretch=False)
        self.progress_tree.column("step", width=190, anchor="w", stretch=True)
        self.progress_tree.tag_configure("running", foreground="#1666b1")
        self.progress_tree.tag_configure("success", foreground="#1f7a1f")
        self.progress_tree.tag_configure("completed", foreground="#1f7a1f")
        self.progress_tree.tag_configure("failed", foreground="#b42318")
        self.progress_tree.tag_configure("stopped", foreground="#8a5a00")
        self.progress_tree.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        task_frame.rowconfigure(2, weight=1)

        control_frame = ttk.LabelFrame(self, text="详细设置", padding=12)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=16)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, textvariable=self.settings_title, foreground="#1666b1").grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 4),
        )
        ttk.Label(control_frame, textvariable=self.settings_hint, foreground="#666666", wraplength=360, justify="left").grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 12),
        )
        ttk.Separator(control_frame).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(control_frame, text="ADB 地址").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(control_frame, textvariable=self.serial).grid(row=3, column=1, sticky="ew", pady=6)
        ttk.Label(control_frame, text="ADB 路径").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(control_frame, textvariable=self.adb_path).grid(row=4, column=1, sticky="ew", pady=6)
        ttk.Label(control_frame, text="MuMu 路径").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(control_frame, textvariable=self.emulator_path).grid(row=5, column=1, sticky="ew", pady=6)
        schedule_row = ttk.Frame(control_frame)
        schedule_row.grid(row=6, column=1, sticky="ew", pady=(8, 16))
        schedule_row.columnconfigure(4, weight=1)
        ttk.Checkbutton(schedule_row, text="定时启动", variable=self.schedule_enabled).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(schedule_row, text="下次启动").grid(row=0, column=1, sticky="w", padx=(18, 4))
        ttk.Entry(schedule_row, textvariable=self.custom_next_start, width=8).grid(row=0, column=2, sticky="w")
        ttk.Label(schedule_row, text="HH:MM").grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Button(schedule_row, text="应用", command=self.apply_custom_schedule).grid(
            row=0,
            column=4,
            sticky="w",
            padx=(8, 0),
        )

        ttk.Label(schedule_row, text="作物类型").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            schedule_row,
            textvariable=self.crop_type,
            values=["1h", "8h", "16h", "32h"],
            state="readonly",
            width=8,
        ).grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(18, 0))

        button_row = ttk.Frame(control_frame)
        button_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=8)
        for index in range(4):
            button_row.columnconfigure(index, weight=1)
        ttk.Button(button_row, text="截图", command=self.capture).grid(row=0, column=0, sticky="ew", padx=3)
        ttk.Button(button_row, text="开始", command=self.start).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(button_row, text="暂停", command=self.toggle_pause).grid(row=0, column=2, sticky="ew", padx=3)
        ttk.Button(button_row, text="停止", command=self.stop).grid(row=0, column=3, sticky="ew", padx=3)

        folder_row = ttk.Frame(control_frame)
        folder_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        folder_row.columnconfigure(0, weight=1)
        folder_row.columnconfigure(1, weight=1)
        folder_row.columnconfigure(2, weight=1)
        ttk.Button(folder_row, text="打开模板目录", command=self.open_template_dir).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=3,
        )
        ttk.Button(folder_row, text="清理截图缓存", command=self.clear_debug_cache).grid(
            row=0,
            column=2,
            sticky="ew",
            padx=3,
        )
        ttk.Button(folder_row, text="打开截图目录", command=self.open_debug_dir).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=3,
        )

        capture_frame = ttk.LabelFrame(control_frame, text="模板采集", padding=10)
        capture_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        capture_frame.columnconfigure(0, weight=1)
        capture_frame.columnconfigure(1, weight=0)
        ttk.Combobox(
            capture_frame,
            textvariable=self.template_stage,
            values=[stage.display_text for stage in CAPTURE_STAGES],
            state="readonly",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(capture_frame, text="采集当前屏幕", command=self.capture_template_source).grid(
            row=0,
            column=1,
            sticky="ew",
        )
        ttk.Button(capture_frame, text="打开素材目录", command=self.open_template_source_dir).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )

        ttk.Label(control_frame, textvariable=self.status, foreground="#1666b1").grid(
            row=10,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(16, 0),
        )
        ttk.Label(control_frame, textvariable=self.schedule_status, foreground="#666666").grid(
            row=11,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        help_text = (
            "模板流程：先让模拟器停到目标画面，选阶段，点“采集当前屏幕”。\n"
            "运行流程：点击开始后会启动/连接 MuMu，执行一键务农，读取成熟时间，并按定时启动设置安排下一轮。"
        )
        ttk.Label(control_frame, text=help_text, foreground="#666666", wraplength=360, justify="left").grid(
            row=12,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(24, 0),
        )

        log_frame = ttk.LabelFrame(self, text="运行日志", padding=12)
        log_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 16), pady=16)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word", height=28, relief="flat", bg="#fbfbfb")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_configure("normal", foreground="#222222")
        self.log_text.tag_configure("success", foreground="#16833a")
        self.log_text.tag_configure("warning", foreground="#8a5a00")
        self.log_text.tag_configure("error", foreground="#b42318")
        self.log_text.tag_configure("snapshot", foreground="#1666b1", underline=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _add_task_option_row(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        variable: BooleanVar,
        title: str,
        description: str,
        settings_key: str,
        enabled: bool = True,
    ) -> None:
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row_frame.columnconfigure(0, weight=1)
        state = "normal" if enabled else "disabled"
        ttk.Checkbutton(row_frame, text=title, variable=variable, state=state).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Button(row_frame, text="⚙", width=3, command=lambda: self._show_task_settings(settings_key)).grid(
            row=0,
            column=1,
            sticky="e",
        )
        ttk.Label(row_frame, text=description, foreground="#666666", wraplength=250).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            padx=(22, 0),
        )

    def _show_task_settings(self, key: str) -> None:
        settings = {
            "wake": (
                "开始唤醒设置",
                "勾选后会保存为本地选项。下次打开客户端时，程序会自动启动已勾选的一键任务。",
            ),
            "farm": (
                "一键务农设置",
                "启动游戏、关闭弹窗、进入农场、移动到互动点、点击一键务农、关闭收获页并读取成熟时间。",
            ),
            "reward": (
                "领取奖励设置",
                "预留功能：后续会在这里配置日常奖励领取流程。",
            ),
            "steal": (
                "自动偷菜设置",
                "预留功能：后续会在这里配置好友农场访问和收取策略。",
            ),
        }
        title, hint = settings.get(key, settings["farm"])
        self.settings_title.set(title)
        self.settings_hint.set(hint)

    def _attach_state_traces(self) -> None:
        for variable in (self.wake_enabled, self.task_enabled, self.schedule_enabled, self.crop_type):
            variable.trace_add("write", lambda *_: self._persist_client_state())

    def _persist_client_state(self) -> None:
        self.client_state.update(
            {
                "wake_on_launch": bool(self.wake_enabled.get()),
                "farm_task_enabled": bool(self.task_enabled.get()),
                "schedule_enabled": bool(self.schedule_enabled.get()),
                "crop_type": self.crop_type.get(),
            }
        )
        save_client_state(self.client_state)

    def _auto_start_if_enabled(self) -> None:
        if self.wake_enabled.get() and self.task_enabled.get() and not (self.worker and self.worker.is_alive()):
            self._append_log_line("开始唤醒：自动执行一键务农")
            self.start(scheduled=True)

    def start(self, scheduled: bool = False) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "当前已有任务在运行。")
            return
        if not self.task_enabled.get():
            messagebox.showwarning("未选择任务", "请至少选择一个任务。")
            return
        self.status.set("正在启动")
        self.paused = False
        self._reset_progress_steps()
        self._append_log_line("定时启动：开始执行一键务农" if scheduled else "开始执行一键务农")
        self.worker = threading.Thread(target=self._run_task, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        if self.engine:
            self.engine.request_stop()
            self.status.set("正在停止")
        else:
            self._append_log_line("当前没有运行中的任务")

    def toggle_pause(self) -> None:
        if not self.engine:
            self._append_log_line("当前没有运行中的任务")
            return
        if self.paused:
            self.engine.resume()
            self.status.set("运行中")
            self.paused = False
        else:
            self.engine.pause()
            self.status.set("已暂停")
            self.paused = True

    def capture(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "任务运行时暂不执行单独截图。")
            return
        self.status.set("正在截图")
        self.worker = threading.Thread(target=self._capture_screenshot, daemon=True)
        self.worker.start()

    def capture_template_source(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "任务运行时暂不执行模板采集。")
            return
        self.status.set("正在采集模板素材")
        self.worker = threading.Thread(target=self._capture_template_source, daemon=True)
        self.worker.start()

    def open_template_dir(self) -> None:
        path = Path(self.config_model.template_dir) / "wzry"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)
        self._append_log_line(f"模板目录：{path.resolve()}")

    def open_template_source_dir(self) -> None:
        path = Path(self.config_model.debug_dir) / "template_sources"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)
        self._append_log_line(f"模板素材目录：{path.resolve()}")

    def open_debug_dir(self) -> None:
        path = Path(self.config_model.debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)
        self._append_log_line(f"截图目录：{path.resolve()}")

    def clear_debug_cache(self) -> None:
        result = clear_screenshot_cache(self.config_model.debug_dir)
        message = f"已清理截图缓存：{result.removed_files} 个文件，释放 {_bytes_text(result.removed_bytes)}"
        self._append_log_line(f"{beijing_now().strftime('%H:%M:%S')} {message}", tag="success")
        self.status.set("缓存已清理")

    def apply_custom_schedule(self) -> None:
        if not self.custom_next_start.get().strip():
            messagebox.showinfo("下次启动时间", "请填写 HH:MM，例如 12:54。")
            return
        self.schedule_enabled.set(True)
        self._schedule_manual_next_run(show_error=True)

    def _run_task(self) -> None:
        recorder = RunEventRecorder(log_dir=self.config_model.log_dir, run_name="client_wzry_farm")
        maturity_time: dict[str, str | None] = {"value": None}
        runtime_config: AppConfig | None = None
        adb: AdbController | None = None
        task: TaskDefinition | None = None
        result: EngineRunResult | None = None

        def handle_event(event: RunEvent) -> None:
            if event.category == "maturity":
                value = event.data.get("maturity_time")
                if isinstance(value, str):
                    maturity_time["value"] = value
            self.event_queue.put(event)

        recorder.subscribe(handle_event)
        try:
            runtime_config = self._runtime_config()
            recorder.emit("INFO", "run", "客户端启动任务")
            task = load_task_definition("tasks/wzry_farm.yaml")
            missing_groups = missing_template_groups(task, runtime_config.template_dir)
            if missing_groups:
                recorder.emit("ERROR", "preflight", format_missing_template_groups(missing_groups))
                return
            adb = AdbController(
                adb_path=runtime_config.adb_path,
                device_serial=runtime_config.device_serial,
                default_timeout=runtime_config.command_timeout_sec,
            )
            recorder.emit(
                "INFO",
                "emulator",
                "准备 MuMu/ADB",
                data={
                    "serial": runtime_config.device_serial,
                    "adb_path": runtime_config.adb_path,
                    "emulator_path": str(runtime_config.emulator_path or ""),
                },
            )
            ensure_emulator_ready(config=runtime_config, adb=adb, events=recorder)
            # 自然成熟收获：上一轮调度记录了目标成熟时刻，本轮到达按钮后等到点再收获，规避偷菜。
            harvest_wait_until = _coerce_datetime(self.client_state.get("crop_harvest_at"))
            if harvest_wait_until is not None and harvest_wait_until > beijing_now():
                recorder.emit(
                    "INFO",
                    "run",
                    f"本轮为自然成熟收获，将提前到按钮前等待至 {harvest_wait_until.strftime('%H:%M')} 收获",
                    data={"harvest_at": harvest_wait_until.isoformat(timespec="seconds")},
                )
            self.engine = TaskEngine(
                adb=adb,
                config=runtime_config,
                task=task,
                dry_run=False,
                events=recorder,
                harvest_wait_until=harvest_wait_until,
            )
            result = self.engine.run()
            recorder.emit(
                "INFO" if result.status == "completed" else "WARN",
                "run",
                f"任务结束：{result.status}",
                node=result.last_node,
                data={"steps": result.steps, "reason": result.reason, "status": result.status},
            )
            if result.status != "completed" and runtime_config is not None:
                self._save_failure_snapshot(
                    adb=adb,
                    config=runtime_config,
                    recorder=recorder,
                    reason=result.status,
                )
            if result.status == "completed":
                self.event_queue.put(
                    (
                        "run_completed",
                        {
                            "maturity_time": maturity_time["value"],
                            "finished_at": beijing_now(),
                        },
                    )
                )
        except Exception as exc:
            recorder.emit("ERROR", "run", f"任务异常：{exc}")
            if adb is not None and runtime_config is not None:
                self._save_failure_snapshot(
                    adb=adb,
                    config=runtime_config,
                    recorder=recorder,
                    reason="exception",
                )
        finally:
            if (
                adb is not None
                and runtime_config is not None
                and should_close_game_after_run(config=runtime_config, task=task, result=result)
            ):
                self._close_game_after_run(adb=adb, config=runtime_config, recorder=recorder)
            elif adb is not None and runtime_config is not None and result is not None and result.last_node:
                node = task.nodes_by_name.get(result.last_node) if task is not None else None
                if node is not None and node.keep_game_open_after_run:
                    recorder.emit("INFO", "emulator", "已按弹窗要求停止脚本，保留王者荣耀运行")
            if runtime_config is not None:
                cleanup = clear_screenshot_cache(runtime_config.debug_dir)
                if cleanup.removed_files:
                    recorder.emit(
                        "INFO",
                        "cache",
                        f"已清理截图缓存：{cleanup.removed_files} 个文件，释放 {_bytes_text(cleanup.removed_bytes)}",
                    )
            recorder.close()
            self.engine = None
            self.event_queue.put(("done", None))

    def _save_failure_snapshot(
        self,
        *,
        adb: AdbController,
        config: AppConfig,
        recorder: RunEventRecorder,
        reason: str,
    ) -> Path | None:
        try:
            snapshot = adb.screencap_png(timeout=config.command_timeout_sec)
            output_dir = Path(config.debug_dir) / "failure_snapshots"
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_reason = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in reason)
            output = output_dir / f"{beijing_now().strftime('%Y%m%d-%H%M%S')}_{safe_reason}.png"
            cv2.imwrite(str(output), snapshot)
        except Exception as exc:
            recorder.emit("WARN", "snapshot", f"保存失败快照失败：{exc}")
            return None
        level = "ERROR" if reason in {"failed", "max_steps", "exception"} else "WARN"
        recorder.emit(
            level,
            "snapshot",
            f"已保存卡死/失败截图：{output}",
            data={"snapshot_path": str(output.resolve()), "reason": reason},
        )
        return output

    def _close_game_after_run(
        self,
        *,
        adb: AdbController,
        config: AppConfig,
        recorder: RunEventRecorder,
    ) -> None:
        try:
            adb.force_stop_package(config.game_package, timeout=config.command_timeout_sec)
        except Exception as exc:
            recorder.emit("WARN", "emulator", f"关闭王者荣耀后台失败：{exc}")
            return
        recorder.emit("INFO", "emulator", "已关闭王者荣耀后台")

    def _capture_screenshot(self) -> None:
        recorder = RunEventRecorder(log_dir=self.config_model.log_dir, run_name="client_capture")
        recorder.subscribe(lambda event: self.event_queue.put(event))
        try:
            runtime_config = self._runtime_config()
            adb = AdbController(
                adb_path=runtime_config.adb_path,
                device_serial=runtime_config.device_serial,
                default_timeout=runtime_config.command_timeout_sec,
            )
            recorder.emit("INFO", "emulator", "准备 MuMu/ADB", data={"serial": runtime_config.device_serial})
            ensure_emulator_ready(config=runtime_config, adb=adb, events=recorder)
            screenshot = adb.screencap_png()
            output = Path(runtime_config.debug_dir) / "client_capture.png"
            recognition_output = Path(runtime_config.debug_dir) / "client_capture_recognition.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), screenshot)
            cv2.imwrite(str(recognition_output), resize_to_recognition(screenshot, runtime_config.recognition_size))
            recorder.emit("INFO", "capture", f"截图已保存：{output}")
            recorder.emit("INFO", "capture", f"识别图已保存：{recognition_output}")
        except Exception as exc:
            recorder.emit("ERROR", "capture", f"截图失败：{exc}")
        finally:
            recorder.close()
            self.event_queue.put(("done", None))

    def _capture_template_source(self) -> None:
        recorder = RunEventRecorder(log_dir=self.config_model.log_dir, run_name="client_template_capture")
        recorder.subscribe(lambda event: self.event_queue.put(event))
        try:
            runtime_config = self._runtime_config()
            adb = AdbController(
                adb_path=runtime_config.adb_path,
                device_serial=runtime_config.device_serial,
                default_timeout=runtime_config.command_timeout_sec,
            )
            recorder.emit("INFO", "emulator", "准备 MuMu/ADB", data={"serial": runtime_config.device_serial})
            ensure_emulator_ready(config=runtime_config, adb=adb, events=recorder)
            stage_id = stage_id_from_display(self.template_stage.get())
            output = capture_recognition_source(
                adb=adb,
                source_dir=Path(runtime_config.debug_dir) / "template_sources",
                stage_id=stage_id,
                recognition_size=runtime_config.recognition_size,
            )
            recorder.emit("INFO", "capture", f"模板素材已保存：{output}")
        except Exception as exc:
            recorder.emit("ERROR", "capture", f"模板采集失败：{exc}")
        finally:
            recorder.close()
            self.event_queue.put(("done", None))

    def _runtime_config(self) -> AppConfig:
        return self.config_model.model_copy(
            update={
                "adb_path": self.adb_path.get().strip() or None,
                "device_serial": self.serial.get().strip() or None,
                "emulator_path": self.emulator_path.get().strip() or None,
            }
        )

    def _poll_events(self) -> None:
        while True:
            try:
                item = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, RunEvent):
                display_text = format_client_log_event(item)
                if display_text:
                    self._append_log_line(display_text, tag=client_log_tag_for_event(item))
                snapshot_path = item.data.get("snapshot_path")
                if isinstance(snapshot_path, str):
                    self._append_snapshot_link(snapshot_path)
                # 读取成熟时间成功时，显示可点击的作物信息卡截图链接。
                card_snapshot_path = item.data.get("card_snapshot_path")
                if isinstance(card_snapshot_path, str):
                    self._append_card_snapshot_link(card_snapshot_path)
                self._update_progress_from_event(item)
                if item.level.value == "ERROR":
                    self.status.set("出错")
                elif item.category == "run":
                    self.status.set(item.message)
            elif isinstance(item, tuple) and item[0] == "run_completed":
                payload = item[1] if isinstance(item[1], dict) else {}
                self._handle_run_completed(payload)
            elif isinstance(item, tuple) and item[0] == "done":
                if self.status.get() not in {"出错"}:
                    self.status.set("已停止")
        self.after(100, self._poll_events)

    def _handle_run_completed(self, payload: dict[str, object]) -> None:
        maturity_time = payload.get("maturity_time")
        if not isinstance(maturity_time, str):
            maturity_time = None
        finished_at = payload.get("finished_at")
        now = finished_at if isinstance(finished_at, datetime) else beijing_now()
        self._append_crop_optimization_report(now=now, maturity_time=maturity_time)
        self._update_crop_schedule_state(now=now, maturity_time=maturity_time)
        if not self.schedule_enabled.get():
            self.next_run_at = None
            self.schedule_status.set("下次启动：未开启")
            return
        if self._schedule_manual_next_run(show_error=False):
            return
        try:
            crop_plan = self._compute_crop_schedule_plan(now=now, maturity_time=maturity_time)
        except ValueError:
            fallback_plan = compute_next_run(now=now, maturity_time=maturity_time)
            self.next_run_at = fallback_plan.next_run
            message = f"下次定时启动：{fallback_plan.next_run.strftime('%Y-%m-%d %H:%M')}"
            self.schedule_status.set(message)
            self._append_log_line(f"{beijing_now().strftime('%H:%M:%S')} {message}", tag="warning")
            return
        self.next_run_at = crop_plan.next_run
        message = (
            f"下次定时启动：{crop_plan.next_run.strftime('%Y-%m-%d %H:%M')}"
            f"（{crop_plan.crop_hours}h作物，{_duration_text(crop_plan.interval)}后）"
        )
        self.schedule_status.set(message)
        self._append_log_line(f"{beijing_now().strftime('%H:%M:%S')} {message}", tag="success")

    def _append_crop_optimization_report(self, *, now: datetime, maturity_time: str | None) -> None:
        try:
            plan = build_crop_optimization_plan(
                now=now,
                crop_type=self.crop_type.get(),
                maturity_time=maturity_time,
                last_run_at=_coerce_datetime(self.client_state.get("crop_last_run_at")),
                last_remaining_after_run=_coerce_timedelta_minutes(
                    self.client_state.get("crop_remaining_after_run_minutes")
                ),
            )
        except ValueError as exc:
            self._append_log_line(f"{beijing_now().strftime('%H:%M:%S')} 浇水最优解计算失败：{exc}", tag="warning")
            return
        report = format_crop_optimization_report(plan)
        prefix = beijing_now().strftime("%H:%M:%S")
        for index, line in enumerate(report.splitlines()):
            self._append_log_line(f"{prefix} {line}" if index == 0 else f"         {line}", tag="success")

    def _compute_crop_schedule_plan(self, *, now: datetime, maturity_time: str | None):
        observed_remaining = remaining_until_clock_time(now=now, clock_time=maturity_time) if maturity_time else None
        return compute_crop_next_run(
            now=now,
            crop_type=self.crop_type.get(),
            last_run_at=_coerce_datetime(self.client_state.get("crop_last_run_at")),
            last_remaining_after_run=_coerce_timedelta_minutes(self.client_state.get("crop_remaining_after_run_minutes")),
            observed_remaining=observed_remaining,
        )

    def _update_crop_schedule_state(self, *, now: datetime, maturity_time: str | None) -> None:
        try:
            plan = self._compute_crop_schedule_plan(now=now, maturity_time=maturity_time)
        except ValueError:
            return
        # 自然成熟收获：记录目标成熟墙钟时刻，供下一次运行在按钮前等待到点再收获。
        # 其它阶段清空，避免把过期目标误用到非自然成熟的运行上。
        harvest_at = (
            plan.harvest_at.isoformat(timespec="seconds")
            if plan.phase == "natural_harvest" and plan.harvest_at is not None
            else None
        )
        self.client_state.update(
            {
                "crop_type": self.crop_type.get(),
                "crop_last_run_at": now.isoformat(timespec="seconds"),
                "crop_remaining_after_run_minutes": round(plan.remaining_after_run.total_seconds() / 60, 3),
                "crop_harvest_at": harvest_at,
            }
        )
        save_client_state(self.client_state)

    def _schedule_manual_next_run(self, *, show_error: bool) -> bool:
        start_time = self.custom_next_start.get().strip()
        if not start_time:
            return False
        try:
            plan = compute_manual_next_run(now=beijing_now(), start_time=start_time)
        except ValueError:
            message = "自定义启动时间格式错误，请使用 HH:MM"
            if show_error:
                self.schedule_status.set(message)
                self._append_log_line(f"{beijing_now().strftime('%H:%M:%S')} {message}", tag="warning")
                messagebox.showwarning("下次启动时间", message)
            return False
        self.next_run_at = plan.next_run
        self.client_state[MANUAL_NEXT_RUN_STATE_KEY] = plan.next_run.isoformat(timespec="seconds")
        save_client_state(self.client_state)
        message = f"下次定时启动：{plan.next_run.strftime('%Y-%m-%d %H:%M')}（手动）"
        self.schedule_status.set(message)
        self._append_log_line(f"{beijing_now().strftime('%H:%M:%S')} {message}", tag="success")
        return True

    def _poll_schedule(self) -> None:
        if not self.schedule_enabled.get():
            self.next_run_at = None
            self.schedule_status.set("下次启动：未开启")
            if self.client_state.pop(MANUAL_NEXT_RUN_STATE_KEY, None) is not None:
                save_client_state(self.client_state)
        else:
            now = beijing_now()
            worker_alive = bool(self.worker and self.worker.is_alive())
            if self.next_run_at is None:
                persisted = _coerce_datetime(self.client_state.get(MANUAL_NEXT_RUN_STATE_KEY))
                if persisted is not None:
                    self.next_run_at = persisted
                    if persisted > now:
                        self.schedule_status.set(f"下次定时启动：{persisted.strftime('%Y-%m-%d %H:%M')}（手动）")
                elif should_arm_manual_schedule(
                    schedule_enabled=True,
                    next_run_at=self.next_run_at,
                    custom_next_start=self.custom_next_start.get(),
                ):
                    self._schedule_manual_next_run(show_error=False)
            if should_trigger_schedule(now=now, next_run_at=self.next_run_at, worker_alive=worker_alive):
                self.next_run_at = None
                self.custom_next_start.set("")
                self.client_state.pop(MANUAL_NEXT_RUN_STATE_KEY, None)
                save_client_state(self.client_state)
                self.schedule_status.set("定时启动：正在执行")
                self.start(scheduled=True)
        self.after(1000, self._poll_schedule)

    def _append_log_line(self, text: str, tag: str = "normal") -> None:
        self.log_text.insert("end", text + "\n", tag)
        self.log_text.see("end")

    def _append_snapshot_link(self, path: str) -> None:
        snapshot_path = Path(path).resolve()
        self._snapshot_link_counter += 1
        tag_name = f"snapshot_{self._snapshot_link_counter}"
        self.log_text.insert("end", "    查看卡死截图\n", ("snapshot", tag_name))
        self.log_text.tag_bind(tag_name, "<Button-1>", lambda _event, target=snapshot_path: os.startfile(target))
        self.log_text.tag_bind(tag_name, "<Enter>", lambda _event: self.log_text.configure(cursor="hand2"))
        self.log_text.tag_bind(tag_name, "<Leave>", lambda _event: self.log_text.configure(cursor=""))
        self.log_text.see("end")

    def _append_card_snapshot_link(self, path: str) -> None:
        """在日志中插入可点击的作物信息卡截图链接，点击后用系统默认图片查看器打开。"""
        snapshot_path = Path(path).resolve()
        self._snapshot_link_counter += 1
        tag_name = f"snapshot_{self._snapshot_link_counter}"
        self.log_text.insert("end", "    查看作物信息卡截图\n", ("snapshot", tag_name))
        self.log_text.tag_bind(tag_name, "<Button-1>", lambda _event, target=snapshot_path: os.startfile(target))
        self.log_text.tag_bind(tag_name, "<Enter>", lambda _event: self.log_text.configure(cursor="hand2"))
        self.log_text.tag_bind(tag_name, "<Leave>", lambda _event: self.log_text.configure(cursor=""))
        self.log_text.see("end")

    def _load_progress_steps(self) -> None:
        self.progress_items.clear()
        for item_id in self.progress_tree.get_children():
            self.progress_tree.delete(item_id)
        try:
            task = load_task_definition("tasks/wzry_farm.yaml")
        except Exception:
            return
        for node in task.nodes:
            if node.name.endswith("_example"):
                continue
            item_id = self.progress_tree.insert("", "end", iid=node.name, values=("等待", node.label))
            self.progress_items[node.name] = item_id

    def _reset_progress_steps(self) -> None:
        self.current_step.set("当前步骤：准备启动")
        for item_id in self.progress_items.values():
            values = self.progress_tree.item(item_id, "values")
            step_name = values[1] if len(values) > 1 else ""
            self.progress_tree.item(item_id, values=("等待", step_name), tags=())

    def _update_progress_from_event(self, event: RunEvent) -> None:
        if event.category != "node" or not event.node:
            return
        item_id = self.progress_items.get(event.node)
        if item_id is None:
            return
        status = str(event.data.get("status") or "").lower()
        display_name = str(event.data.get("display_name") or event.node)
        status_text = {
            "running": "执行中",
            "success": "完成",
            "completed": "完成",
            "failed": "失败",
            "stopped": "停止",
        }.get(status, "进行中")
        self.progress_tree.item(item_id, values=(status_text, display_name), tags=(status,))
        self.progress_tree.see(item_id)
        if status == "running":
            self.current_step.set(f"当前步骤：{display_name}")


def main() -> None:
    app = AutomationClient()
    app.mainloop()


if __name__ == "__main__":
    main()
