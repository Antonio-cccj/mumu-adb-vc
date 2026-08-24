from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adb_controller import AdbController
from maturity_reader import DEFAULT_CARD_ROI, read_maturity_time
from navigation.route_config import load_navigation_route
from navigation.route_navigator import NavigationCallbacks, RouteNavigator
from runtime_events import EventHub
from settings import AppConfig
from time_utils import beijing_now
from vision import MatchResult, Point, RecognitionSize, Rect, crop_roi, map_point_to_screen, match_template, resize_to_recognition


ActionName = Literal[
    "click",
    "swipe",
    "swipe_sequence",
    "joystick_route",
    "visual_approach",
    "route_navigate",
    "read_maturity_time",
    "key_sequence",
    "tap_point",
    "wait",
    "wait_for_harvest",
    "stop",
]
JoystickDirection = Literal[
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
]
MovementMode = Literal["target_to_desired", "actor_relative"]
TerminalStatus = Literal["completed", "failed", "stopped"]


class SwipeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: list[int] = Field(..., min_length=2, max_length=2)
    end: list[int] = Field(..., min_length=2, max_length=2)
    duration_ms: int = Field(default=500, ge=1)


class JoystickStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: JoystickDirection
    duration_ms: int = Field(default=500, ge=1)
    distance: int | None = Field(default=None, ge=1)


class JoystickRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center: list[int] = Field(..., min_length=2, max_length=2)
    distance: int = Field(default=90, ge=1)
    steps: list[JoystickStep] = Field(..., min_length=1)


class VisualApproachSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_templates: list[str] = Field(..., min_length=1)
    success_templates: list[str] = Field(..., min_length=1)
    interrupt_templates: list[str] | None = None
    joystick_templates: list[str] | None = None
    target_roi: list[int] | None = None
    success_roi: list[int] | None = None
    interrupt_roi: list[int] | None = None
    joystick_roi: list[int] | None = None
    target_threshold: float = Field(default=0.76, ge=-1.0, le=1.0)
    tracking_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    tracking_radius: int = Field(default=180, ge=1)
    success_threshold: float = Field(default=0.82, ge=-1.0, le=1.0)
    interrupt_threshold: float = Field(default=0.80, ge=-1.0, le=1.0)
    joystick_threshold: float = Field(default=0.80, ge=-1.0, le=1.0)
    movement_mode: MovementMode = "target_to_desired"
    actor_center: list[int] | None = Field(default=None, min_length=2, max_length=2)
    desired_center: list[int] | None = Field(default=None, min_length=2, max_length=2)
    tolerance: list[int] = Field(default_factory=lambda: [60, 60], min_length=2, max_length=2)
    joystick_center: list[int] = Field(default_factory=lambda: [188, 616], min_length=2, max_length=2)
    joystick_distance: int = Field(default=88, ge=1)
    step_duration_ms: int = Field(default=350, ge=1)
    step_wait_ms: int = Field(default=500, ge=0)
    interrupt_wait_ms: int = Field(default=500, ge=0)
    max_attempts: int = Field(default=8, ge=1)
    invert_x: bool = False
    invert_y: bool = False
    fallback_direction: JoystickDirection = "left"
    fallback_directions: list[JoystickDirection] = Field(default_factory=list)

    @field_validator("target_roi", "success_roi", "interrupt_roi", "joystick_roi")
    @classmethod
    def validate_optional_roi(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 4:
            raise ValueError("roi must be [x, y, width, height]")
        return value


class RouteNavigateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    max_attempts: int = Field(default=45, ge=1)


class MaturityReadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roi: list[int] = Field(
        default_factory=lambda: [
            DEFAULT_CARD_ROI.x,
            DEFAULT_CARD_ROI.y,
            DEFAULT_CARD_ROI.width,
            DEFAULT_CARD_ROI.height,
        ]
    )
    anchor_template: str | None = None
    anchor_threshold: float = Field(default=0.72, ge=-1.0, le=1.0)
    anchor_roi: list[int] | None = None
    tomorrow_template: str | None = None
    tomorrow_threshold: float = Field(default=0.78, ge=-1.0, le=1.0)
    farm_templates: list[str] = Field(default_factory=list)
    farm_roi: list[int] | None = None
    farm_threshold: float = Field(default=0.78, ge=-1.0, le=1.0)
    farm_min_matches: int = Field(default=1, ge=1)
    joystick_center: list[int] = Field(default_factory=lambda: [184, 505], min_length=2, max_length=2)
    direction: JoystickDirection = "up_left"
    distance: int = Field(default=72, ge=1)
    duration_ms: int = Field(default=280, ge=1)
    step_wait_ms: int = Field(default=8000, ge=0)
    max_attempts: int = Field(default=60, ge=1)

    @field_validator("roi", "anchor_roi", "farm_roi")
    @classmethod
    def validate_roi(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 4:
            raise ValueError("roi must be [x, y, width, height]")
        return value


class KeyStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    duration_ms: int = Field(default=0, ge=0)
    repeat: int | None = Field(default=None, ge=1)
    interval_ms: int = Field(default=100, ge=1)

    @property
    def input_count(self) -> int:
        if self.repeat is not None:
            return self.repeat
        if self.duration_ms <= 0:
            return 1
        return max(1, math.ceil(self.duration_ms / self.interval_ms))


class TaskNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str | None = None
    roi: list[int] | None = None
    template: str | None = None
    templates: list[str] | None = None
    threshold: float = Field(default=0.8, ge=-1.0, le=1.0)
    min_matches: int = Field(default=1, ge=1)
    action: ActionName = "wait"
    terminal_status: TerminalStatus = "completed"
    next: str | None = None
    timeout_ms: int = Field(default=5000, ge=1)
    retry: int = Field(default=1, ge=1)
    on_fail: str | None = None
    offset: list[int] | None = None
    swipe: SwipeSpec | None = None
    swipes: list[SwipeSpec] | None = None
    joystick: JoystickRoute | None = None
    approach: VisualApproachSpec | None = None
    route: RouteNavigateSpec | None = None
    maturity: MaturityReadSpec | None = None
    key_sequence: list[KeyStep] | None = None
    point: list[int] | None = None
    wait_ms: int | None = Field(default=None, ge=1)
    post_action_wait_ms: int | None = Field(default=None, ge=0)
    # 设为 true 时，stuck_recheck 机制不会在此节点上触发页面重判跳转，
    # 适合允许长时间重试但不希望被误判为"卡死"的节点（如 dismiss_reward_screen）。
    no_stuck_recheck: bool = False
    # 设为 true 时，客户端任务收尾阶段会保留游戏运行。适合需要用户接手的合规弹窗。
    keep_game_open_after_run: bool = False

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 4:
            raise ValueError("roi must be [x, y, width, height]")
        return value

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 2:
            raise ValueError("offset must be [dx, dy]")
        return value

    @field_validator("point")
    @classmethod
    def validate_point(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 2:
            raise ValueError("point must be [x, y]")
        return value

    @field_validator("joystick")
    @classmethod
    def validate_joystick(cls, value: JoystickRoute | None) -> JoystickRoute | None:
        if value is not None and len(value.center) != 2:
            raise ValueError("joystick.center must be [x, y]")
        return value

    def template_names(self) -> list[str]:
        names: list[str] = []
        if self.template:
            names.append(self.template)
        if self.templates:
            names.extend(self.templates)
        return names

    @property
    def label(self) -> str:
        return self.display_name or self.name


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    fallback: str | None = None
    max_steps: int = Field(default=100, ge=1)
    emergency_stop: bool = False
    auto_detect_start: bool = False
    auto_detect_timeout_ms: int = Field(default=5000, ge=0)
    entry_nodes: list[str] = Field(default_factory=list)
    stuck_recheck_after: int = Field(default=0, ge=0)
    stuck_recheck_nodes: list[str] = Field(default_factory=list)
    nodes: list[TaskNode] = Field(..., min_length=1)

    @property
    def nodes_by_name(self) -> dict[str, TaskNode]:
        return {node.name: node for node in self.nodes}

    @model_validator(mode="after")
    def validate_references(self) -> "TaskDefinition":
        names = [node.name for node in self.nodes]
        duplicate_names = {name for name in names if names.count(name) > 1}
        if duplicate_names:
            raise ValueError(f"Duplicate task node names: {sorted(duplicate_names)}")
        known = set(names)
        references = [self.start, self.fallback, *self.entry_nodes, *self.stuck_recheck_nodes]
        for node in self.nodes:
            references.extend([node.next, node.on_fail])
        missing = sorted({ref for ref in references if ref and ref not in known})
        if missing:
            raise ValueError(f"Task references unknown node(s): {missing}")
        return self


@dataclass(frozen=True)
class EngineRunResult:
    status: Literal["completed", "stopped", "failed", "max_steps"]
    steps: int
    last_node: str | None
    reason: str


@dataclass(frozen=True)
class NodePresence:
    matched_count: int
    min_matches: int
    best_template: str | None
    score: float | None
    found: bool


def load_task_definition(path: str | Path) -> TaskDefinition:
    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Task file must contain a YAML mapping: {task_path}")
    return TaskDefinition.model_validate(raw)


class TaskEngine:
    def __init__(
        self,
        *,
        adb: AdbController,
        config: AppConfig,
        task: TaskDefinition,
        dry_run: bool = False,
        logger: logging.Logger | None = None,
        events: EventHub | None = None,
        harvest_wait_until: datetime | None = None,
    ) -> None:
        self.adb = adb
        self.config = config
        self.task = task
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)
        self.events = events or EventHub()
        # 自然成熟收获时的目标成熟墙钟时刻：脚本提前到达按钮后，在此时刻前等待，
        # 到点再点击一键务农，避免成熟后未收获被偷菜。None 表示本次无需等待。
        self.harvest_wait_until = harvest_wait_until
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._redirect_node: str | None = None

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def paused(self) -> bool:
        return self._pause_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()
        self._emit("WARN", "run", "收到停止请求")

    def pause(self) -> None:
        self._pause_event.set()
        self.logger.info("Task engine paused.")
        self._emit("INFO", "run", "已暂停")

    def resume(self) -> None:
        self._pause_event.clear()
        self.logger.info("Task engine resumed.")
        self._emit("INFO", "run", "已继续")

    def toggle_pause(self) -> None:
        if self.paused:
            self.resume()
        else:
            self.pause()

    def run(self) -> EngineRunResult:
        if self.task.emergency_stop:
            self._emit("WARN", "run", "任务文件 emergency_stop 为 true，停止运行")
            return EngineRunResult(
                status="stopped",
                steps=0,
                last_node=None,
                reason="Task file emergency_stop is true",
            )

        current_name: str | None = self._select_initial_node()
        steps = 0
        node_visit_counts: dict[str, int] = {}
        while current_name and not self.stopped:
            if steps >= self.task.max_steps:
                return EngineRunResult(
                    status="max_steps",
                    steps=steps,
                    last_node=current_name,
                    reason=f"Reached max_steps={self.task.max_steps}",
                )

            node = self.task.nodes_by_name[current_name]
            visit_count = node_visit_counts.get(current_name, 0) + 1
            node_visit_counts[current_name] = visit_count
            rechecked, detected_node_name = self._maybe_recheck_stuck_node(node, visit_count)
            if rechecked:
                node_visit_counts[current_name] = 0
                if detected_node_name and detected_node_name != current_name:
                    current_name = detected_node_name
                    continue

            steps += 1
            self.logger.info("Node %s started (%s/%s).", node.name, steps, self.task.max_steps)
            self._emit_node_status(node, "running", f"开始步骤：{node.label}", step=steps)
            status = self._run_node(node)
            if status == "redirect":
                redirect_node = self._redirect_node
                self._redirect_node = None
                if redirect_node:
                    self._emit_node_status(node, "failed", f"页面重判跳转：{node.label}", step=steps)
                    current_name = redirect_node
                    continue
                status = "failed"
            if status == "stopped":
                self._emit("WARN", "run", "任务已停止", node=node.name)
                return EngineRunResult(status="stopped", steps=steps, last_node=node.name, reason="Stopped")
            if status == "success":
                if node.action == "stop":
                    terminal_status = node.terminal_status
                    if terminal_status == "completed":
                        level = "INFO"
                        message = "任务完成"
                    elif terminal_status == "failed":
                        level = "ERROR"
                        message = "任务失败"
                    else:
                        level = "WARN"
                        message = "任务停止"
                    self._emit_node_status(node, terminal_status, f"{message}：{node.label}", step=steps)
                    self._emit(
                        level,
                        "run",
                        message,
                        node=node.name,
                        data={"display_name": node.label, "terminal_status": terminal_status},
                    )
                    return EngineRunResult(
                        status=terminal_status,
                        steps=steps,
                        last_node=node.name,
                        reason="Stop node reached",
                    )
                self._emit_node_status(node, "success", f"步骤完成：{node.label}", step=steps)
                current_name = node.next
                if current_name is None:
                    self._emit("INFO", "run", "任务完成：没有下一节点", node=node.name)
                    return EngineRunResult(status="completed", steps=steps, last_node=node.name, reason="No next node")
                continue

            next_on_fail = node.on_fail or self.task.fallback
            if not next_on_fail:
                self._emit_node_status(node, "failed", f"步骤失败：{node.label}", step=steps)
                self._emit("ERROR", "run", "节点失败且没有失败出口", node=node.name)
                return EngineRunResult(status="failed", steps=steps, last_node=node.name, reason="Node failed without on_fail")
            self.logger.warning("Node %s failed; switching to %s.", node.name, next_on_fail)
            self._emit_node_status(node, "failed", f"步骤失败：{node.label}", step=steps)
            self._emit("WARN", "node", f"节点失败，切换到 {next_on_fail}", node=node.name)
            current_name = next_on_fail

        self._emit("WARN", "run", "停止请求已处理", node=current_name)
        return EngineRunResult(status="stopped", steps=steps, last_node=current_name, reason="Stop requested")

    def _select_initial_node(self) -> str:
        if not self.task.auto_detect_start or not self.task.entry_nodes:
            return self.task.start

        self._emit(
            "INFO",
            "state",
            "启动状态识别：检查当前画面",
            data={
                "entry_nodes": self.task.entry_nodes,
                "timeout_ms": self.task.auto_detect_timeout_ms,
            },
        )
        deadline = time.monotonic() + self.task.auto_detect_timeout_ms / 1000
        attempt = 0
        while True:
            attempt += 1
            try:
                screenshot = self.adb.screencap_png()
            except Exception as exc:
                self._emit("WARN", "state", f"启动状态识别失败，使用默认起点：{exc}")
                return self.task.start

            actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
            for node_name in self.task.entry_nodes:
                node = self.task.nodes_by_name[node_name]
                presence = self._detect_node_presence(screenshot, node, actual_size, purpose="entry")
                self._emit(
                    "INFO",
                    "state",
                    f"入口识别：{node.label} 命中 {presence.matched_count}/{presence.min_matches}",
                    node=node.name,
                    template=presence.best_template,
                    score=presence.score,
                    data={
                        "attempt": attempt,
                        "matched_count": presence.matched_count,
                        "min_matches": presence.min_matches,
                        "found": presence.found,
                    },
                )
                if presence.found:
                    self._emit(
                        "INFO",
                        "state",
                        f"入口识别命中：从 {node.label} 继续",
                        node=node.name,
                        template=presence.best_template,
                        score=presence.score,
                        data={"attempt": attempt},
                    )
                    return node.name

            if time.monotonic() >= deadline:
                break
            if self._sleep_interruptible(self.config.screenshot_interval_ms) == "stopped":
                return self.task.start

        self._emit("INFO", "state", f"入口识别未命中，使用默认起点：{self.task.start}")
        return self.task.start

    def _maybe_recheck_stuck_node(self, node: TaskNode, visit_count: int) -> tuple[bool, str | None]:
        if self.task.stuck_recheck_after <= 0:
            return False, None
        if visit_count <= self.task.stuck_recheck_after:
            return False, None
        # 节点设置了 no_stuck_recheck=true 时，跳过页面重判，让正常的 on_fail / next 链处理。
        if node.no_stuck_recheck:
            return False, None

        candidates = self.task.stuck_recheck_nodes or self.task.entry_nodes
        if not candidates:
            return False, None

        self._emit(
            "WARN",
            "state",
            f"节点连续循环超过 {self.task.stuck_recheck_after} 次，重新识别当前页面",
            node=node.name,
            data={
                "visit_count": visit_count,
                "stuck_recheck_after": self.task.stuck_recheck_after,
                "candidates": candidates,
            },
        )
        detected_node_name = self._detect_current_state_node(candidates, purpose="stuck")
        if detected_node_name:
            detected_node = self.task.nodes_by_name[detected_node_name]
            self._emit(
                "WARN",
                "state",
                f"页面重判命中：切换到 {detected_node.label}",
                node=detected_node_name,
                data={"from_node": node.name, "to_node": detected_node_name},
            )
        else:
            self._emit(
                "WARN",
                "state",
                "页面重判未命中候选节点，继续当前步骤",
                node=node.name,
                data={"from_node": node.name},
            )
        return True, detected_node_name

    def _maybe_redirect_from_attempt_recheck(self, node: TaskNode, attempt: int) -> bool:
        if self.task.stuck_recheck_after <= 0:
            return False
        if attempt < self.task.stuck_recheck_after or attempt % self.task.stuck_recheck_after != 0:
            return False
        # 节点设置了 no_stuck_recheck=true 时，跳过页面重判，避免在长时间等待的节点
        # 中误触发跳转（如 dismiss_reward_screen 等待收获页出现期间检测到 one_key_farm）。
        if node.no_stuck_recheck:
            return False

        candidates = self.task.stuck_recheck_nodes or self.task.entry_nodes
        if not candidates:
            return False

        self._emit(
            "WARN",
            "state",
            f"节点识别连续 {attempt} 次未完成，重新识别当前页面",
            node=node.name,
            data={
                "attempt": attempt,
                "stuck_recheck_after": self.task.stuck_recheck_after,
                "candidates": candidates,
            },
        )
        detected_node_name = self._detect_current_state_node(candidates, purpose="stuck_attempt")
        if not detected_node_name:
            self._emit("WARN", "state", "页面重判未命中候选节点，继续当前步骤", node=node.name)
            return False
        if detected_node_name == node.name:
            self._emit("INFO", "state", "页面重判仍是当前步骤，继续等待", node=node.name)
            return False

        detected_node = self.task.nodes_by_name[detected_node_name]
        self._redirect_node = detected_node_name
        self._emit(
            "WARN",
            "state",
            f"页面重判命中：切换到 {detected_node.label}",
            node=detected_node_name,
            data={"from_node": node.name, "to_node": detected_node_name, "attempt": attempt},
        )
        return True

    def _detect_current_state_node(self, candidate_nodes: list[str], *, purpose: str) -> str | None:
        try:
            screenshot = self.adb.screencap_png()
        except Exception as exc:
            self._emit("WARN", "state", f"页面状态识别截图失败：{exc}")
            return None

        actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
        for node_name in candidate_nodes:
            node = self.task.nodes_by_name[node_name]
            if not node.template_names():
                continue
            presence = self._detect_node_presence(screenshot, node, actual_size, purpose=purpose)
            self._emit(
                "INFO",
                "state",
                f"页面状态识别：{node.label} 命中 {presence.matched_count}/{presence.min_matches}",
                node=node.name,
                template=presence.best_template,
                score=presence.score,
                data={
                    "matched_count": presence.matched_count,
                    "min_matches": presence.min_matches,
                    "found": presence.found,
                    "purpose": purpose,
                },
            )
            if presence.found:
                return node.name
        return None

    def _detect_node_presence(
        self,
        screenshot,
        node: TaskNode,
        actual_size: RecognitionSize,
        *,
        purpose: str,
    ) -> NodePresence:
        template_names = node.template_names()
        if not template_names:
            return NodePresence(
                matched_count=0,
                min_matches=node.min_matches,
                best_template=None,
                score=None,
                found=False,
            )

        best_result = None
        best_template_name: str | None = None
        matched_count = 0
        for template_name in template_names:
            template_path = self._resolve_template_path(template_name)
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                self._emit("WARN", "template", "模板不可读", node=node.name, template=template_name)
                continue
            debug_path = Path(self.config.debug_dir) / (
                f"{node.name}_{purpose}_{Path(template_name).stem}_{int(time.time() * 1000)}.png"
            )
            result = match_template(
                screenshot,
                template,
                threshold=node.threshold,
                roi=Rect.from_sequence(node.roi),
                recognition_size=self.config.recognition_size,
                actual_size=actual_size,
                debug_path=debug_path,
            )
            if best_result is None or result.score > best_result.score:
                best_result = result
                best_template_name = template_name
            if result.found:
                matched_count += 1

        if best_result is None:
            return NodePresence(
                matched_count=0,
                min_matches=node.min_matches,
                best_template=None,
                score=None,
                found=False,
            )
        return NodePresence(
            matched_count=matched_count,
            min_matches=node.min_matches,
            best_template=best_template_name,
            score=best_result.score,
            found=matched_count >= node.min_matches,
        )

    def _run_node(self, node: TaskNode) -> Literal["success", "failed", "stopped", "redirect"]:
        deadline = time.monotonic() + node.timeout_ms / 1000
        for attempt in range(1, node.retry + 1):
            if self._wait_while_paused_or_stopped() == "stopped":
                return "stopped"
            if time.monotonic() > deadline:
                self.logger.warning("Node %s timed out before attempt %s.", node.name, attempt)
                self._emit("WARN", "node", "节点执行超时", node=node.name)
                return "failed"

            template_names = node.template_names()
            if not template_names and node.action in {
                "visual_approach",
                "route_navigate",
                "read_maturity_time",
                "key_sequence",
                "wait",
                "wait_for_harvest",
                "stop",
            }:
                self.logger.info("Node %s has no template; executing %s without pre-capture.", node.name, node.action)
                self._emit("INFO", "action", f"鎵ц鍔ㄤ綔 {node.action}", node=node.name, action=node.action)
                return self._execute_action(node, None, self.config.recognition_size)

            started = time.perf_counter()
            screenshot = self.adb.screencap_png()
            actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
            self.logger.info("Node %s attempt %s/%s captured screenshot.", node.name, attempt, node.retry)
            self._emit("DEBUG", "capture", "截图完成", node=node.name, data={"attempt": attempt, "retry": node.retry})

            if not template_names:
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.logger.info("Node %s has no template; executing %s after %.1fms.", node.name, node.action, elapsed_ms)
                self._emit("INFO", "action", f"执行动作 {node.action}", node=node.name, action=node.action)
                return self._execute_action(node, None, actual_size)

            best_result = None
            best_template_name: str | None = None
            matched_count = 0
            for template_name in template_names:
                template_path = self._resolve_template_path(template_name)
                template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                if template is None:
                    self.logger.warning("Template not found or unreadable for node %s: %s", node.name, template_path)
                    self._emit("WARN", "template", "模板不可读", node=node.name, template=template_name)
                    continue

                debug_path = Path(self.config.debug_dir) / (
                    f"{node.name}_{Path(template_name).stem}_{int(time.time() * 1000)}.png"
                )
                result = match_template(
                    screenshot,
                    template,
                    threshold=node.threshold,
                    roi=Rect.from_sequence(node.roi),
                    recognition_size=self.config.recognition_size,
                    actual_size=actual_size,
                    debug_path=debug_path,
                )
                self.logger.info(
                    "Node %s template=%s score=%.4f found=%s.",
                    node.name,
                    template_name,
                    result.score,
                    result.found,
                )
                self._emit(
                    "INFO",
                    "match",
                    "模板匹配完成",
                    node=node.name,
                    template=template_name,
                    score=result.score,
                    data={
                        "found": result.found,
                        "center": [result.center.x, result.center.y],
                        "screen_center": [result.screen_center.x, result.screen_center.y],
                    },
                )
                if best_result is None or result.score > best_result.score:
                    best_result = result
                    best_template_name = template_name
                if result.found:
                    matched_count += 1

            if best_result is None:
                self.logger.warning("Node %s had no readable templates.", node.name)
                self._emit("WARN", "template", "没有可读取的模板", node=node.name)
                if self._maybe_redirect_from_attempt_recheck(node, attempt):
                    return "redirect"
                self._sleep_interruptible(self.config.screenshot_interval_ms)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000
            self.logger.info(
                "Node %s best_template=%s score=%.4f found=%s center=(%s,%s) screen=(%s,%s) elapsed=%.1fms.",
                node.name,
                best_template_name,
                best_result.score,
                best_result.found,
                best_result.center.x,
                best_result.center.y,
                best_result.screen_center.x,
                best_result.screen_center.y,
                elapsed_ms,
            )
            self._emit(
                "INFO",
                "match",
                f"{node.label}：模板组命中 {matched_count}/{node.min_matches}",
                node=node.name,
                template=best_template_name,
                score=best_result.score,
                data={"matched_count": matched_count, "min_matches": node.min_matches},
            )
            if matched_count >= node.min_matches:
                self._emit(
                    "INFO",
                    "match",
                    "选择最高分模板",
                    node=node.name,
                    template=best_template_name,
                    score=best_result.score,
                    data={"matched_count": matched_count, "min_matches": node.min_matches},
                )
                return self._execute_action(node, best_result, actual_size)

            if self._maybe_redirect_from_attempt_recheck(node, attempt):
                return "redirect"
            self._sleep_interruptible(self.config.screenshot_interval_ms)
            if time.monotonic() > deadline:
                self.logger.warning("Node %s timed out after %.1fms.", node.name, node.timeout_ms)
                self._emit("WARN", "node", "节点执行超时", node=node.name)
                return "failed"

        return "failed"

    def _execute_action(
        self,
        node: TaskNode,
        result,
        actual_size: RecognitionSize,
    ) -> Literal["success", "failed", "stopped"]:
        if node.action == "stop":
            self.request_stop()
            self.logger.info("Stop action reached.")
            self._emit("INFO", "action", "执行停止动作", node=node.name, action="stop")
            return "success"

        if node.action == "wait":
            wait_ms = node.wait_ms or self.config.screenshot_interval_ms
            self.logger.info("Waiting %sms.", wait_ms)
            self._emit("INFO", "action", f"等待 {wait_ms}ms", node=node.name, action="wait")
            return self._sleep_interruptible(wait_ms)

        if node.action == "wait_for_harvest":
            return self._execute_wait_for_harvest(node)

        if node.action == "click":
            if result is None:
                self.logger.error("Click action requires a successful template match.")
                return "failed"
            target = result.center
            if node.offset:
                target = Point(x=target.x + int(node.offset[0]), y=target.y + int(node.offset[1]))
            screen_target = map_point_to_screen(target, self.config.recognition_size, actual_size)
            self.logger.info(
                "%s tap at recognition=(%s,%s), screen=(%s,%s).",
                "Dry-run would" if self.dry_run else "Executing",
                target.x,
                target.y,
                screen_target.x,
                screen_target.y,
            )
            self._emit(
                "INFO",
                "action",
                "Dry-run 点击" if self.dry_run else "执行点击",
                node=node.name,
                action="click",
                data={
                    "recognition": [target.x, target.y],
                    "screen": [screen_target.x, screen_target.y],
                    "dry_run": self.dry_run,
                },
            )
            if not self.dry_run:
                self.adb.tap(screen_target.x, screen_target.y)
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"

        if node.action == "tap_point":
            if node.point is None:
                self.logger.error("Tap point action requires point.")
                return "failed"
            target = Point(x=int(node.point[0]), y=int(node.point[1]))
            screen_target = map_point_to_screen(target, self.config.recognition_size, actual_size)
            self._emit(
                "INFO",
                "action",
                "Dry-run 定点点击" if self.dry_run else "执行定点点击",
                node=node.name,
                action="tap_point",
                data={
                    "recognition": [target.x, target.y],
                    "screen": [screen_target.x, screen_target.y],
                    "dry_run": self.dry_run,
                },
            )
            if not self.dry_run:
                self.adb.tap(screen_target.x, screen_target.y)
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"

        if node.action == "swipe":
            if node.swipe is None:
                self.logger.error("Swipe action requires swipe.start and swipe.end.")
                return "failed"
            start = map_point_to_screen(
                Point(x=int(node.swipe.start[0]), y=int(node.swipe.start[1])),
                self.config.recognition_size,
                actual_size,
            )
            end = map_point_to_screen(
                Point(x=int(node.swipe.end[0]), y=int(node.swipe.end[1])),
                self.config.recognition_size,
                actual_size,
            )
            self.logger.info(
                "%s swipe screen=(%s,%s)->(%s,%s), duration=%sms.",
                "Dry-run would" if self.dry_run else "Executing",
                start.x,
                start.y,
                end.x,
                end.y,
                node.swipe.duration_ms,
            )
            self._emit(
                "INFO",
                "action",
                "Dry-run 滑动" if self.dry_run else "执行滑动",
                node=node.name,
                action="swipe",
                data={
                    "start": [start.x, start.y],
                    "end": [end.x, end.y],
                    "duration_ms": node.swipe.duration_ms,
                    "dry_run": self.dry_run,
                },
            )
            if not self.dry_run:
                self.adb.swipe(start.x, start.y, end.x, end.y, node.swipe.duration_ms)
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"

        if node.action == "swipe_sequence":
            if not node.swipes:
                self.logger.error("Swipe sequence action requires swipes.")
                return "failed"
            for index, swipe in enumerate(node.swipes, start=1):
                start = map_point_to_screen(
                    Point(x=int(swipe.start[0]), y=int(swipe.start[1])),
                    self.config.recognition_size,
                    actual_size,
                )
                end = map_point_to_screen(
                    Point(x=int(swipe.end[0]), y=int(swipe.end[1])),
                    self.config.recognition_size,
                    actual_size,
                )
                self._emit(
                    "INFO",
                    "action",
                    "Dry-run 摇杆滑动" if self.dry_run else "执行摇杆滑动",
                    node=node.name,
                    action="swipe_sequence",
                    data={
                        "index": index,
                        "start": [start.x, start.y],
                        "end": [end.x, end.y],
                        "duration_ms": swipe.duration_ms,
                        "dry_run": self.dry_run,
                    },
                )
                if not self.dry_run:
                    self.adb.swipe(start.x, start.y, end.x, end.y, swipe.duration_ms)
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"

        if node.action == "joystick_route":
            if node.joystick is None:
                self.logger.error("Joystick route action requires joystick.")
                return "failed"
            center = Point(x=int(node.joystick.center[0]), y=int(node.joystick.center[1]))
            for index, step in enumerate(node.joystick.steps, start=1):
                if self._wait_while_paused_or_stopped() == "stopped":
                    return "stopped"
                distance = step.distance or node.joystick.distance
                end = self._joystick_end(center, step.direction, distance)
                screen_start = map_point_to_screen(center, self.config.recognition_size, actual_size)
                screen_end = map_point_to_screen(end, self.config.recognition_size, actual_size)
                self._emit(
                    "INFO",
                    "action",
                    "Dry-run 摇杆移动" if self.dry_run else "执行摇杆移动",
                    node=node.name,
                    action="joystick_route",
                    data={
                        "index": index,
                        "direction": step.direction,
                        "recognition_start": [center.x, center.y],
                        "recognition_end": [end.x, end.y],
                        "screen_start": [screen_start.x, screen_start.y],
                        "screen_end": [screen_end.x, screen_end.y],
                        "duration_ms": step.duration_ms,
                        "dry_run": self.dry_run,
                    },
                )
                if not self.dry_run:
                    self.adb.swipe(
                        screen_start.x,
                        screen_start.y,
                        screen_end.x,
                        screen_end.y,
                        step.duration_ms,
                    )
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"

        if node.action == "visual_approach":
            if node.approach is None:
                self.logger.error("Visual approach action requires approach.")
                return "failed"
            return self._execute_visual_approach(node, actual_size)

        if node.action == "route_navigate":
            if node.route is None:
                self.logger.error("Route navigation action requires route.")
                return "failed"
            return self._execute_route_navigate(node)

        if node.action == "read_maturity_time":
            return self._execute_read_maturity_time(node, actual_size)

        if node.action == "key_sequence":
            if not node.key_sequence:
                self.logger.error("Key sequence action requires key_sequence.")
                return "failed"
            for step in node.key_sequence:
                for index in range(step.input_count):
                    self._emit(
                        "INFO",
                        "action",
                        "Dry-run 按键" if self.dry_run else "执行按键",
                        node=node.name,
                        action="key_sequence",
                        data={
                            "key": step.key,
                            "index": index + 1,
                            "count": step.input_count,
                            "dry_run": self.dry_run,
                        },
                    )
                    if not self.dry_run:
                        self.adb.keyevent(step.key)
                    if index + 1 < step.input_count:
                        sleep_status = self._sleep_interruptible(step.interval_ms)
                        if sleep_status == "stopped":
                            return "stopped"
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"

        self.logger.error("Unsupported action: %s", node.action)
        self._emit("ERROR", "action", f"不支持的动作 {node.action}", node=node.name, action=node.action)
        return "failed"

    def _execute_wait_for_harvest(self, node: TaskNode) -> Literal["success", "failed", "stopped"]:
        """到达“一键务农”按钮后，若本次为自然成熟收获，则在此原地等待到成熟时刻再继续点击，
        以覆盖脚本启动与导航耗时，规避作物成熟后未收获被偷菜的窗口。"""
        target = self.harvest_wait_until
        if target is None:
            # 本次不是自然成熟收获（如浇水/终浇即收），无需等待，直接放行。
            self._emit("INFO", "action", "无需提前等待，直接点击", node=node.name, action="wait_for_harvest")
            return "success"

        wait_seconds = (target - beijing_now()).total_seconds()
        if wait_seconds <= 0:
            # 已到/已过成熟时刻：立刻收获即可。
            self._emit(
                "INFO",
                "action",
                "已到作物成熟时刻，立即收获",
                node=node.name,
                action="wait_for_harvest",
                data={"harvest_at": target.isoformat(timespec="seconds")},
            )
            return "success"

        self._emit(
            "INFO",
            "action",
            f"已到按钮前，等待作物成熟（{target.strftime('%H:%M')}）后收获，约 {int(round(wait_seconds))} 秒",
            node=node.name,
            action="wait_for_harvest",
            data={
                "harvest_at": target.isoformat(timespec="seconds"),
                "wait_seconds": round(wait_seconds, 1),
            },
        )
        # 可被暂停/停止中断的等待；到点后由下一节点点击一键务农完成收获。
        status = self._sleep_interruptible(int(round(wait_seconds * 1000)))
        if status == "stopped":
            return "stopped"
        if node.post_action_wait_ms:
            return self._sleep_interruptible(node.post_action_wait_ms)
        return "success"

    def _execute_read_maturity_time(
        self,
        node: TaskNode,
        actual_size: RecognitionSize,
    ) -> Literal["success", "failed", "stopped"]:
        spec = node.maturity or MaturityReadSpec()
        card_roi = Rect.from_sequence(spec.roi)
        anchor_roi = Rect.from_sequence(spec.anchor_roi)
        anchor_template = None
        if spec.anchor_template:
            anchor_path = self._resolve_template_path(spec.anchor_template)
            anchor_template = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
            if anchor_template is None:
                self._emit(
                    "WARN",
                    "template",
                    "成熟时间锚点模板不可读，使用固定区域兜底",
                    node=node.name,
                    template=spec.anchor_template,
                )
        tomorrow_template = None
        if spec.tomorrow_template:
            tomorrow_path = self._resolve_template_path(spec.tomorrow_template)
            tomorrow_template = cv2.imread(str(tomorrow_path), cv2.IMREAD_COLOR)
            if tomorrow_template is None:
                self._emit(
                    "WARN",
                    "template",
                    "成熟时间明天前缀模板不可读，将只读取当天 HH:MM",
                    node=node.name,
                    template=spec.tomorrow_template,
                )
        farm_templates = []
        for template_name in spec.farm_templates:
            template_path = self._resolve_template_path(template_name)
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                self._emit(
                    "WARN",
                    "template",
                    "农场页面守卫模板不可读，跳过该模板",
                    node=node.name,
                    template=template_name,
                )
                continue
            farm_templates.append((template_name, template))
        center = Point(x=int(spec.joystick_center[0]), y=int(spec.joystick_center[1]))
        for attempt in range(1, spec.max_attempts + 1):
            if self._wait_while_paused_or_stopped() == "stopped":
                return "stopped"
            screenshot = self.adb.screencap_png()
            actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
            if farm_templates and not self._maturity_farm_present(screenshot, actual_size, spec, farm_templates):
                self._emit(
                    "WARN",
                    "maturity",
                    "读取成熟时间前未确认在农场，返回关闭务农收获页",
                    node=node.name,
                    action="read_maturity_time",
                    data={
                        "attempt": attempt,
                        "required_count": spec.farm_min_matches,
                        "template_count": len(farm_templates),
                    },
                )
                return "failed"
            result = read_maturity_time(
                screenshot,
                self.config.recognition_size,
                card_roi=card_roi,
                anchor_template=anchor_template,
                anchor_threshold=spec.anchor_threshold,
                anchor_roi=anchor_roi,
                tomorrow_template=tomorrow_template,
                tomorrow_threshold=spec.tomorrow_threshold,
            )
            if result.found and result.time_text:
                # 保存作物信息卡截图供用户查看（识别尺寸的完整画面，信息卡在左上角）。
                card_snapshot_str: str | None = None
                try:
                    recognition = resize_to_recognition(screenshot, self.config.recognition_size)
                    card_dir = Path(self.config.debug_dir) / "card_snapshots"
                    card_dir.mkdir(parents=True, exist_ok=True)
                    card_path = card_dir / f"{beijing_now().strftime('%Y%m%d-%H%M%S')}_card.png"
                    cv2.imwrite(str(card_path), recognition)
                    card_snapshot_str = str(card_path.resolve())
                except Exception as snap_exc:
                    self._emit(
                        "WARN",
                        "maturity",
                        f"保存信息卡截图失败：{snap_exc}",
                        node=node.name,
                        action="read_maturity_time",
                    )

                event_data: dict = {
                    "maturity_time": result.time_text,
                    "score": result.score,
                    "attempt": attempt,
                    "day_offset": result.day_offset,
                    "anchor_found": result.anchor_found,
                }
                if card_snapshot_str:
                    event_data["card_snapshot_path"] = card_snapshot_str

                self._emit(
                    "INFO",
                    "maturity",
                    f"作物成熟时间：{result.time_text}",
                    node=node.name,
                    action="read_maturity_time",
                    data=event_data,
                )
                if node.post_action_wait_ms:
                    return self._sleep_interruptible(node.post_action_wait_ms)
                return "success"

            if attempt >= spec.max_attempts:
                break

            end = self._joystick_end(center, spec.direction, spec.distance)
            screen_start = map_point_to_screen(center, self.config.recognition_size, actual_size)
            screen_end = map_point_to_screen(end, self.config.recognition_size, actual_size)
            self._emit(
                "INFO",
                "action",
                "移动到作物上读取成熟时间",
                node=node.name,
                action="read_maturity_time",
                data={
                    "attempt": attempt,
                    "direction": spec.direction,
                    "screen_start": [screen_start.x, screen_start.y],
                    "screen_end": [screen_end.x, screen_end.y],
                    "duration_ms": spec.duration_ms,
                    "dry_run": self.dry_run,
                },
            )
            if not self.dry_run:
                self.adb.swipe(screen_start.x, screen_start.y, screen_end.x, screen_end.y, spec.duration_ms)
            if spec.step_wait_ms:
                sleep_status = self._sleep_interruptible(spec.step_wait_ms)
                if sleep_status == "stopped":
                    return "stopped"

        self._emit(
            "WARN",
            "maturity",
            "未读取到作物成熟时间，将使用 5 小时默认定时",
            node=node.name,
            action="read_maturity_time",
            data={"max_attempts": spec.max_attempts},
        )
        return "success"

    def _maturity_farm_present(
        self,
        screenshot,
        actual_size: RecognitionSize,
        spec: MaturityReadSpec,
        farm_templates,
    ) -> bool:
        matched_count = 0
        best_score: float | None = None
        best_template: str | None = None
        for template_name, template in farm_templates:
            result = match_template(
                screenshot,
                template,
                threshold=spec.farm_threshold,
                roi=Rect.from_sequence(spec.farm_roi),
                recognition_size=self.config.recognition_size,
                actual_size=actual_size,
            )
            if best_score is None or result.score > best_score:
                best_score = result.score
                best_template = template_name
            if result.found:
                matched_count += 1
        self._emit(
            "INFO",
            "state",
            f"成熟时间读取前确认农场：命中 {matched_count}/{spec.farm_min_matches}",
            template=best_template,
            score=best_score,
            data={
                "matched_count": matched_count,
                "min_matches": spec.farm_min_matches,
                "found": matched_count >= spec.farm_min_matches,
            },
        )
        return matched_count >= spec.farm_min_matches

    def _execute_route_navigate(self, node: TaskNode) -> Literal["success", "failed", "stopped"]:
        spec = node.route
        if spec is None:
            return "failed"
        try:
            route = load_navigation_route(spec.path)
        except Exception as exc:
            self.logger.exception("Failed to load route file %s.", spec.path)
            self._emit(
                "ERROR",
                "route",
                f"Route file load failed: {exc}",
                node=node.name,
                action="route_navigate",
                data={"path": spec.path},
            )
            return "failed"

        def emit(level: str, category: str, message: str, **kwargs) -> None:
            self._emit(level, category, message, node=node.name, **kwargs)

        def tap(x: int, y: int) -> None:
            self._emit(
                "INFO",
                "action",
                "Dry-run route tap" if self.dry_run else "Route tap",
                node=node.name,
                action="route_navigate",
                data={"screen": [x, y], "dry_run": self.dry_run},
            )
            if not self.dry_run:
                self.adb.tap(x, y)

        def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
            self._emit(
                "INFO",
                "action",
                "Dry-run route joystick" if self.dry_run else "Route joystick",
                node=node.name,
                action="route_navigate",
                data={
                    "screen_start": [x1, y1],
                    "screen_end": [x2, y2],
                    "duration_ms": duration_ms,
                    "dry_run": self.dry_run,
                },
            )
            if not self.dry_run:
                self.adb.swipe(x1, y1, x2, y2, duration_ms)

        navigator = RouteNavigator(
            route,
            self.config,
            NavigationCallbacks(
                screencap=self.adb.screencap_png,
                tap=tap,
                swipe=swipe,
                sleep_ms=self._sleep_interruptible,
                emit=emit,
                stopped=lambda: self.stopped,
            ),
        )
        result = navigator.run(max_attempts=spec.max_attempts)
        self._emit(
            "INFO" if result.status == "success" else "WARN",
            "route",
            f"route navigation result: {result.status} ({result.reason})",
            node=node.name,
            action="route_navigate",
            data={"attempts": result.attempts, "status": result.status, "reason": result.reason},
        )
        if result.status == "success":
            if node.post_action_wait_ms:
                return self._sleep_interruptible(node.post_action_wait_ms)
            return "success"
        if result.status == "stopped":
            return "stopped"
        return "failed"

    def _execute_visual_approach(
        self,
        node: TaskNode,
        actual_size: RecognitionSize,
    ) -> Literal["success", "failed", "stopped"]:
        spec = node.approach
        if spec is None:
            return "failed"

        last_target_center: Point | None = None
        for attempt in range(1, spec.max_attempts + 1):
            if self._wait_while_paused_or_stopped() == "stopped":
                return "stopped"

            screenshot = self.adb.screencap_png()
            actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
            if spec.interrupt_templates:
                interrupt_name, interrupt_result = self._match_template_candidates(
                    screenshot,
                    spec.interrupt_templates,
                    threshold=spec.interrupt_threshold,
                    roi=Rect.from_sequence(spec.interrupt_roi),
                    actual_size=actual_size,
                    node_name=node.name,
                    purpose="interrupt",
                )
                if interrupt_result is not None and interrupt_result.found:
                    self._emit(
                        "INFO",
                        "action",
                        "Dry-run close interrupt popup" if self.dry_run else "Close interrupt popup",
                        node=node.name,
                        template=interrupt_name,
                        score=interrupt_result.score,
                        action="visual_approach",
                        data={
                            "attempt": attempt,
                            "interrupt_center": [interrupt_result.center.x, interrupt_result.center.y],
                            "screen_center": [
                                interrupt_result.screen_center.x,
                                interrupt_result.screen_center.y,
                            ],
                            "dry_run": self.dry_run,
                        },
                    )
                    if not self.dry_run:
                        self.adb.tap(interrupt_result.screen_center.x, interrupt_result.screen_center.y)
                    if spec.interrupt_wait_ms:
                        sleep_status = self._sleep_interruptible(spec.interrupt_wait_ms)
                        if sleep_status == "stopped":
                            return "stopped"
                    continue

            success_name, success_result = self._match_template_candidates(
                screenshot,
                spec.success_templates,
                threshold=spec.success_threshold,
                roi=Rect.from_sequence(spec.success_roi),
                actual_size=actual_size,
                node_name=node.name,
                purpose="success",
            )
            if success_result is not None and success_result.found:
                self._emit(
                    "INFO",
                    "action",
                    "视觉走位完成：已出现目标按钮",
                    node=node.name,
                    template=success_name,
                    score=success_result.score,
                    action="visual_approach",
                    data={
                        "attempt": attempt,
                        "center": [success_result.center.x, success_result.center.y],
                    },
                )
                if node.post_action_wait_ms:
                    return self._sleep_interruptible(node.post_action_wait_ms)
                return "success"

            target_name, target_result = self._match_template_candidates(
                screenshot,
                spec.target_templates,
                threshold=spec.target_threshold,
                roi=Rect.from_sequence(spec.target_roi),
                actual_size=actual_size,
                node_name=node.name,
                purpose="target",
            )
            tracking_used = False
            target_locked = target_result is not None and target_result.found
            if not target_locked and spec.tracking_threshold is not None:
                tracking_used = self._accept_tracking_candidate(
                    target_result,
                    last_target=last_target_center,
                    tracking_threshold=spec.tracking_threshold,
                    tracking_radius=spec.tracking_radius,
                )
                target_locked = tracking_used

            if target_result is not None and target_locked:
                direction, navigation_reference, target_distance = self._visual_target_direction(
                    spec=spec,
                    target=target_result.center,
                )
                last_target_center = target_result.center
            else:
                target_name = None
                target_result = None
                direction = self._fallback_approach_direction(spec, attempt)
                navigation_reference = None
                target_distance = None
                tracking_used = False

            center, joystick_source, joystick_name, joystick_score = self._resolve_joystick_center(
                screenshot,
                spec,
                actual_size=actual_size,
                node_name=node.name,
            )
            end = self._joystick_end(center, direction, spec.joystick_distance)
            screen_start = map_point_to_screen(center, self.config.recognition_size, actual_size)
            screen_end = map_point_to_screen(end, self.config.recognition_size, actual_size)
            target_status = (
                f"target_distance={target_distance}"
                if target_distance is not None
                else "target_lost=search"
            )
            approach_prefix = "dry-run " if self.dry_run else ""
            self._emit(
                "INFO",
                "action",
                "Dry-run 视觉走位" if self.dry_run else "执行视觉走位",
                node=node.name,
                template=target_name,
                score=target_result.score if target_result is not None else None,
                action="visual_approach",
                data={
                    "attempt": attempt,
                    "direction": direction,
                    "movement_mode": spec.movement_mode,
                    "target_found": target_locked,
                    "tracking_used": tracking_used,
                    "target_center": (
                        [target_result.center.x, target_result.center.y] if target_result is not None else None
                    ),
                    "navigation_reference": (
                        [navigation_reference.x, navigation_reference.y] if navigation_reference is not None else None
                    ),
                    "target_distance": target_distance,
                    "joystick_source": joystick_source,
                    "joystick_template": joystick_name,
                    "joystick_score": joystick_score,
                    "recognition_start": [center.x, center.y],
                    "recognition_end": [end.x, end.y],
                    "screen_start": [screen_start.x, screen_start.y],
                    "screen_end": [screen_end.x, screen_end.y],
                    "duration_ms": spec.step_duration_ms,
                    "dry_run": self.dry_run,
                    "log_summary": (
                        f"{approach_prefix}visual_approach {attempt}/{spec.max_attempts}: "
                        f"mode={spec.movement_mode}, direction={direction}, {target_status}"
                    ),
                },
            )
            if not self.dry_run:
                self.adb.swipe(
                    screen_start.x,
                    screen_start.y,
                    screen_end.x,
                    screen_end.y,
                    spec.step_duration_ms,
                )
            if spec.step_wait_ms:
                sleep_status = self._sleep_interruptible(spec.step_wait_ms)
                if sleep_status == "stopped":
                    return "stopped"

        self._emit(
            "WARN",
            "action",
            "视觉走位达到最大尝试次数",
            node=node.name,
            action="visual_approach",
            data={"max_attempts": spec.max_attempts},
        )
        return "failed"

    def _resolve_joystick_center(
        self,
        screenshot,
        spec: VisualApproachSpec,
        *,
        actual_size: RecognitionSize,
        node_name: str,
    ) -> tuple[Point, str, str | None, float | None]:
        configured = Point(x=int(spec.joystick_center[0]), y=int(spec.joystick_center[1]))
        if not spec.joystick_templates:
            return configured, "configured", None, None

        joystick_name, joystick_result = self._match_template_candidates(
            screenshot,
            spec.joystick_templates,
            threshold=spec.joystick_threshold,
            roi=Rect.from_sequence(spec.joystick_roi),
            actual_size=actual_size,
            node_name=node_name,
            purpose="joystick",
        )
        if joystick_result is not None and joystick_result.found:
            self._emit(
                "INFO",
                "state",
                "摇杆位置识别成功",
                node=node_name,
                template=joystick_name,
                score=joystick_result.score,
                data={"center": [joystick_result.center.x, joystick_result.center.y]},
            )
            return joystick_result.center, "detected", joystick_name, joystick_result.score

        self._emit(
            "WARN",
            "state",
            "摇杆模板未命中，使用配置中心",
            node=node_name,
            template=joystick_name,
            score=joystick_result.score if joystick_result is not None else None,
            data={"center": [configured.x, configured.y]},
        )
        return configured, "configured", joystick_name, joystick_result.score if joystick_result is not None else None

    def _match_template_candidates(
        self,
        screenshot,
        template_names: list[str],
        *,
        threshold: float,
        roi: Rect | None,
        actual_size: RecognitionSize,
        node_name: str,
        purpose: str,
    ):
        best_result = None
        best_template_name: str | None = None
        for template_name in template_names:
            template_path = self._resolve_template_path(template_name)
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                self._emit("WARN", "template", "模板不可读", node=node_name, template=template_name)
                continue
            debug_path = Path(self.config.debug_dir) / (
                f"{node_name}_{purpose}_{Path(template_name).stem}_{int(time.time() * 1000)}.png"
            )
            result = match_template(
                screenshot,
                template,
                threshold=threshold,
                roi=roi,
                recognition_size=self.config.recognition_size,
                actual_size=actual_size,
                debug_path=debug_path,
            )
            self._emit(
                "INFO",
                "match",
                f"视觉走位{purpose}模板匹配完成",
                node=node_name,
                template=template_name,
                score=result.score,
                data={
                    "found": result.found,
                    "center": [result.center.x, result.center.y],
                    "screen_center": [result.screen_center.x, result.screen_center.y],
                },
            )
            if best_result is None or result.score > best_result.score:
                best_result = result
                best_template_name = template_name
        return best_template_name, best_result

    @staticmethod
    def _fallback_approach_direction(spec: VisualApproachSpec, attempt: int) -> JoystickDirection:
        directions = spec.fallback_directions or [spec.fallback_direction]
        return directions[(attempt - 1) % len(directions)]

    @staticmethod
    def _accept_tracking_candidate(
        candidate: MatchResult | None,
        *,
        last_target: Point | None,
        tracking_threshold: float,
        tracking_radius: int,
    ) -> bool:
        if candidate is None or last_target is None:
            return False
        if candidate.score < tracking_threshold:
            return False
        distance = math.hypot(candidate.center.x - last_target.x, candidate.center.y - last_target.y)
        return distance <= tracking_radius

    def _visual_target_direction(
        self,
        *,
        spec: VisualApproachSpec,
        target: Point,
    ) -> tuple[JoystickDirection, Point, float]:
        if spec.movement_mode == "actor_relative":
            reference = self._point_from_optional_pair(
                spec.actor_center,
                fallback=self._point_from_optional_pair(
                    spec.desired_center,
                    fallback=Point(
                        x=self.config.recognition_size.width // 2,
                        y=self.config.recognition_size.height // 2,
                    ),
                ),
            )
            direction = self._screen_vector_direction(
                origin=reference,
                target=target,
                tolerance=spec.tolerance,
                fallback=spec.fallback_direction,
                invert_x=spec.invert_x,
                invert_y=spec.invert_y,
            )
        else:
            reference = self._point_from_optional_pair(
                spec.desired_center,
                fallback=Point(
                    x=self.config.recognition_size.width // 2,
                    y=self.config.recognition_size.height // 2,
                ),
            )
            direction = self._approach_direction(
                current=target,
                desired=reference,
                tolerance=spec.tolerance,
                fallback=spec.fallback_direction,
                invert_x=spec.invert_x,
                invert_y=spec.invert_y,
            )
        distance = math.hypot(target.x - reference.x, target.y - reference.y)
        return direction, reference, round(distance, 2)

    @staticmethod
    def _point_from_optional_pair(value: list[int] | None, *, fallback: Point) -> Point:
        if value is None:
            return fallback
        return Point(x=int(value[0]), y=int(value[1]))

    def _resolve_template_path(self, template: str) -> Path:
        template_path = Path(template)
        if template_path.is_absolute() or template_path.exists():
            return template_path
        return Path(self.config.template_dir) / template_path

    @staticmethod
    def _approach_direction(
        *,
        current: Point,
        desired: Point,
        tolerance: list[int],
        fallback: JoystickDirection,
        invert_x: bool,
        invert_y: bool,
    ) -> JoystickDirection:
        horizontal: str | None = None
        vertical: str | None = None
        tolerance_x = int(tolerance[0])
        tolerance_y = int(tolerance[1])
        if current.x < desired.x - tolerance_x:
            horizontal = "left"
        elif current.x > desired.x + tolerance_x:
            horizontal = "right"
        if current.y < desired.y - tolerance_y:
            vertical = "up"
        elif current.y > desired.y + tolerance_y:
            vertical = "down"

        if invert_x and horizontal == "left":
            horizontal = "right"
        elif invert_x and horizontal == "right":
            horizontal = "left"
        if invert_y and vertical == "up":
            vertical = "down"
        elif invert_y and vertical == "down":
            vertical = "up"

        if vertical and horizontal:
            return f"{vertical}_{horizontal}"  # type: ignore[return-value]
        if vertical:
            return vertical  # type: ignore[return-value]
        if horizontal:
            return horizontal  # type: ignore[return-value]
        return fallback

    @staticmethod
    def _screen_vector_direction(
        *,
        origin: Point,
        target: Point,
        tolerance: list[int],
        fallback: JoystickDirection,
        invert_x: bool,
        invert_y: bool,
    ) -> JoystickDirection:
        horizontal: str | None = None
        vertical: str | None = None
        tolerance_x = int(tolerance[0])
        tolerance_y = int(tolerance[1])
        if target.x < origin.x - tolerance_x:
            horizontal = "left"
        elif target.x > origin.x + tolerance_x:
            horizontal = "right"
        if target.y < origin.y - tolerance_y:
            vertical = "up"
        elif target.y > origin.y + tolerance_y:
            vertical = "down"

        if invert_x and horizontal == "left":
            horizontal = "right"
        elif invert_x and horizontal == "right":
            horizontal = "left"
        if invert_y and vertical == "up":
            vertical = "down"
        elif invert_y and vertical == "down":
            vertical = "up"

        if vertical and horizontal:
            return f"{vertical}_{horizontal}"  # type: ignore[return-value]
        if vertical:
            return vertical  # type: ignore[return-value]
        if horizontal:
            return horizontal  # type: ignore[return-value]
        return fallback

    @staticmethod
    def _joystick_end(center: Point, direction: JoystickDirection, distance: int) -> Point:
        diagonal = int(round(distance / math.sqrt(2)))
        offsets: dict[str, tuple[int, int]] = {
            "up": (0, -distance),
            "down": (0, distance),
            "left": (-distance, 0),
            "right": (distance, 0),
            "up_left": (-diagonal, -diagonal),
            "up_right": (diagonal, -diagonal),
            "down_left": (-diagonal, diagonal),
            "down_right": (diagonal, diagonal),
        }
        dx, dy = offsets[direction]
        return Point(x=center.x + dx, y=center.y + dy)

    def _wait_while_paused_or_stopped(self) -> Literal["success", "stopped"]:
        logged = False
        while self.paused and not self.stopped:
            if not logged:
                self.logger.info("Paused. Type r to resume or s to stop.")
                logged = True
            time.sleep(0.1)
        if self.stopped:
            return "stopped"
        return "success"

    def _sleep_interruptible(self, duration_ms: int) -> Literal["success", "stopped"]:
        deadline = time.monotonic() + duration_ms / 1000
        while time.monotonic() < deadline:
            if self._wait_while_paused_or_stopped() == "stopped":
                return "stopped"
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return "success"

    def _emit(
        self,
        level: str,
        category: str,
        message: str,
        *,
        node: str | None = None,
        template: str | None = None,
        score: float | None = None,
        action: str | None = None,
        data: dict | None = None,
    ) -> None:
        if data is not None and isinstance(data.get("log_summary"), str):
            message = f"{message} [{data['log_summary']}]"
        self.events.emit(
            level,
            category,
            message,
            node=node,
            template=template,
            score=score,
            action=action,
            data=data,
        )

    def _emit_node_status(
        self,
        node: TaskNode,
        status: str,
        message: str,
        *,
        step: int | None = None,
    ) -> None:
        data = {
            "display_name": node.label,
            "status": status,
        }
        if step is not None:
            data["step"] = step
        self._emit("INFO" if status in {"running", "success", "completed"} else "WARN", "node", message, node=node.name, data=data)
