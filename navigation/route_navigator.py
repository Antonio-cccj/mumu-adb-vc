from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np

from navigation.feature_localizer import FeatureLocalizer, LocalizationResult
from navigation.route_config import NavigationRoute, RouteMove, TemplateGroup
from settings import AppConfig
from vision import Point, RecognitionSize, Rect, map_point_to_screen, match_template


@dataclass(frozen=True)
class NavigationCallbacks:
    screencap: Callable[[], object]
    tap: Callable[[int, int], None]
    swipe: Callable[[int, int, int, int, int], None]
    sleep_ms: Callable[[int], Literal["success", "stopped"]]
    emit: Callable[..., None]
    stopped: Callable[[], bool]


@dataclass(frozen=True)
class NavigationRunResult:
    status: Literal["success", "failed", "stopped", "running"]
    attempts: int
    reason: str


@dataclass(frozen=True)
class _MoveDecision:
    move: RouteMove
    decision: str
    active_index: int | None
    progress_index: int
    same_keyframe_count: int


class RouteNavigator:
    def __init__(
        self,
        route: NavigationRoute,
        config: AppConfig,
        callbacks: NavigationCallbacks,
    ) -> None:
        self.route = route
        self.config = config
        self.callbacks = callbacks
        self.localizer = FeatureLocalizer(route, config.recognition_size)
        self._progress_index = -1
        self._last_keyframe_index: int | None = None
        self._same_keyframe_count = 0
        self._correction_offset = 0
        self._recovery_offset = 0
        self._fixed_step_started_at = time.monotonic()
        self._template_cache: dict[str, np.ndarray | None] = {}

    def run(self, *, max_attempts: int) -> NavigationRunResult:
        for attempt in range(1, max_attempts + 1):
            if self.callbacks.stopped():
                return NavigationRunResult("stopped", attempt - 1, "stop requested")
            screenshot = self.callbacks.screencap()
            actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])

            if self._handle_template_group(
                screenshot,
                actual_size,
                self.route.interrupt,
                purpose="interrupt",
                attempt=attempt,
                click=True,
            ):
                continue

            if self._handle_template_group(
                screenshot,
                actual_size,
                self.route.success,
                purpose="success",
                attempt=attempt,
                click=False,
            ):
                return NavigationRunResult("success", attempt, "success template found")

            if self.route.mode == "fixed_step":
                fixed_result = self._run_fixed_step_attempt(screenshot, actual_size, attempt, max_attempts)
                if fixed_result is not None:
                    return fixed_result
                continue

            localization = self.localizer.locate(screenshot)
            decision = self._select_move(localization)
            move = decision.move
            self.callbacks.emit(
                "INFO",
                "navigation",
                (
                    f"route_navigate {attempt}/{max_attempts}: "
                    f"keyframe={localization.keyframe_name}, score={localization.score}, "
                    f"decision={decision.decision}, move={move}"
                ),
                action="route_navigate",
                data={
                    "attempt": attempt,
                    "keyframe": localization.keyframe_name,
                    "keyframe_index": localization.index,
                    "active_index": decision.active_index,
                    "progress_index": decision.progress_index,
                    "same_keyframe_count": decision.same_keyframe_count,
                    "score": localization.score,
                    "inliers": localization.inliers,
                    "inlier_ratio": localization.inlier_ratio,
                    "correlation": localization.correlation,
                    "found": localization.found,
                    "decision": decision.decision,
                    "move": move,
                    "top_candidates": [
                        {
                            "name": candidate.keyframe_name,
                            "index": candidate.index,
                            "score": candidate.score,
                            "found": candidate.found,
                        }
                        for candidate in localization.candidates
                    ],
                },
            )
            if move == "stop":
                return NavigationRunResult("success", attempt, "terminal keyframe reached")
            if move != "wait":
                self._move_joystick(move, actual_size)
            if self.route.joystick.wait_ms:
                sleep_status = self.callbacks.sleep_ms(self.route.joystick.wait_ms)
                if sleep_status == "stopped":
                    return NavigationRunResult("stopped", attempt, "stop requested")
        return NavigationRunResult("failed", max_attempts, "max attempts reached")

    def _run_fixed_step_attempt(
        self,
        screenshot,
        actual_size: RecognitionSize,
        attempt: int,
        max_attempts: int,
    ) -> NavigationRunResult | None:
        elapsed_ms = int((time.monotonic() - self._fixed_step_started_at) * 1000)
        if elapsed_ms >= self.route.fixed_step.reset_after_ms:
            reset_clicked = self._handle_template_group(
                screenshot,
                actual_size,
                self.route.reset,
                purpose="reset",
                attempt=attempt,
                click=True,
            )
            self.callbacks.emit(
                "INFO",
                "navigation",
                (
                    f"fixed_step {attempt}/{max_attempts}: "
                    f"elapsed_ms={elapsed_ms}, decision={'reset' if reset_clicked else 'reset_missed'}"
                ),
                action="route_navigate",
                data={
                    "attempt": attempt,
                    "elapsed_ms": elapsed_ms,
                    "reset_after_ms": self.route.fixed_step.reset_after_ms,
                    "decision": "reset" if reset_clicked else "reset_missed",
                    "move": None,
                },
            )
            if reset_clicked:
                self._fixed_step_started_at = time.monotonic()
                return None

        move = self.route.fixed_step.direction
        self.callbacks.emit(
            "INFO",
            "navigation",
            (
                f"fixed_step {attempt}/{max_attempts}: "
                f"decision=fixed_step, move={move}, wait_ms={self.route.fixed_step.step_wait_ms}, "
                f"settle_wait_ms={self.route.fixed_step.settle_wait_ms}, "
                f"poll_interval_ms={self.route.fixed_step.success_poll_interval_ms}"
            ),
            action="route_navigate",
            data={
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
                "reset_after_ms": self.route.fixed_step.reset_after_ms,
                "decision": "fixed_step",
                "move": move,
                "wait_ms": self.route.fixed_step.step_wait_ms,
                "settle_wait_ms": self.route.fixed_step.settle_wait_ms,
                "poll_interval_ms": self.route.fixed_step.success_poll_interval_ms,
            },
        )
        self._move_joystick(move, actual_size)
        return self._wait_fixed_step_window(attempt)

    def _wait_fixed_step_window(self, attempt: int) -> NavigationRunResult | None:
        total_wait_ms = self.route.fixed_step.step_wait_ms
        if total_wait_ms <= 0:
            return self._poll_fixed_step_templates(attempt, 0)

        should_poll = bool(self.route.success.templates or self.route.interrupt.templates)
        if not should_poll:
            sleep_status = self.callbacks.sleep_ms(total_wait_ms)
            if sleep_status == "stopped":
                return NavigationRunResult("stopped", attempt, "stop requested")
            return None

        settle_wait_ms = min(self.route.fixed_step.settle_wait_ms, total_wait_ms)
        if settle_wait_ms:
            sleep_status = self.callbacks.sleep_ms(settle_wait_ms)
            if sleep_status == "stopped":
                return NavigationRunResult("stopped", attempt, "stop requested")
        return self._poll_fixed_step_templates(attempt, total_wait_ms - settle_wait_ms)

    def _poll_fixed_step_templates(self, attempt: int, window_ms: int) -> NavigationRunResult | None:
        deadline = time.monotonic() + max(window_ms, 0) / 1000
        first_check = True
        while first_check or time.monotonic() < deadline:
            first_check = False
            if self.callbacks.stopped():
                return NavigationRunResult("stopped", attempt, "stop requested")
            screenshot = self.callbacks.screencap()
            actual_size = RecognitionSize(width=screenshot.shape[1], height=screenshot.shape[0])
            if self._handle_template_group(
                screenshot,
                actual_size,
                self.route.interrupt,
                purpose="interrupt",
                attempt=attempt,
                click=True,
            ):
                continue
            if self._handle_template_group(
                screenshot,
                actual_size,
                self.route.success,
                purpose="success",
                attempt=attempt,
                click=False,
            ):
                return NavigationRunResult("success", attempt, "success template found")
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break
            sleep_ms = min(self.route.fixed_step.success_poll_interval_ms, remaining_ms)
            sleep_status = self.callbacks.sleep_ms(sleep_ms)
            if sleep_status == "stopped":
                return NavigationRunResult("stopped", attempt, "stop requested")
        return None

    def _handle_template_group(
        self,
        screenshot,
        actual_size: RecognitionSize,
        group: TemplateGroup,
        *,
        purpose: str,
        attempt: int,
        click: bool,
    ) -> bool:
        if not group.templates:
            return False
        best_name = None
        best_result = None
        for template_name in group.templates:
            template = self._load_template(template_name)
            if template is None:
                continue
            result = match_template(
                screenshot,
                template,
                threshold=group.threshold,
                roi=Rect.from_sequence(group.roi),
                recognition_size=self.config.recognition_size,
                actual_size=actual_size,
            )
            if best_result is None or result.score > best_result.score:
                best_name = template_name
                best_result = result
        if best_result is None or not best_result.found:
            return False
        self.callbacks.emit(
            "INFO",
            "navigation",
            f"route_navigate {purpose} matched",
            template=best_name,
            score=best_result.score,
            action="route_navigate",
            data={
                "attempt": attempt,
                "purpose": purpose,
                "center": [best_result.center.x, best_result.center.y],
                "screen_center": [best_result.screen_center.x, best_result.screen_center.y],
            },
        )
        if click:
            self.callbacks.tap(best_result.screen_center.x, best_result.screen_center.y)
        if group.wait_ms:
            self.callbacks.sleep_ms(group.wait_ms)
        return True

    def _load_template(self, template_name: str) -> np.ndarray | None:
        if template_name not in self._template_cache:
            template_path = self._resolve_template_path(template_name)
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            self._template_cache[template_name] = template
            if template is None:
                self.callbacks.emit(
                    "WARN",
                    "template",
                    f"route template unreadable: {template_name}",
                    template=template_name,
                )
        return self._template_cache[template_name]

    def _select_move(self, localization: LocalizationResult) -> _MoveDecision:
        if localization.found and localization.index is not None:
            active_index = localization.index
            if self._last_keyframe_index == active_index:
                self._same_keyframe_count += 1
            else:
                self._same_keyframe_count = 1
                self._correction_offset = 0
            self._last_keyframe_index = active_index

            if active_index > self._progress_index:
                self._progress_index = active_index
                decision = "advance"
            elif active_index == self._progress_index:
                decision = "follow"
            else:
                decision = "backtrack_seen"

            keyframe = self.route.keyframes[active_index]
            stuck_after = keyframe.stuck_after or self.route.stuck_after
            if self._same_keyframe_count > self.route.max_stuck_attempts and self.route.recovery_moves:
                move = self._next_recovery_move()
                decision = "route_recovery"
            elif self._same_keyframe_count > stuck_after and keyframe.correction_moves:
                move = keyframe.correction_moves[self._correction_offset % len(keyframe.correction_moves)]
                self._correction_offset += 1
                decision = "correction"
            else:
                move = keyframe.move
            return _MoveDecision(
                move=move,
                decision=decision,
                active_index=active_index,
                progress_index=self._progress_index,
                same_keyframe_count=self._same_keyframe_count,
            )

        self._last_keyframe_index = None
        self._same_keyframe_count = 0
        self._correction_offset = 0
        if not self.route.recovery_moves:
            return _MoveDecision("wait", "lost_wait", None, self._progress_index, 0)
        return _MoveDecision(
            self._next_recovery_move(),
            "lost_recovery",
            None,
            self._progress_index,
            0,
        )

    def _next_recovery_move(self) -> RouteMove:
        move = self.route.recovery_moves[self._recovery_offset % len(self.route.recovery_moves)]
        self._recovery_offset += 1
        return move

    def _move_joystick(self, direction: RouteMove, actual_size: RecognitionSize) -> None:
        center = Point(x=int(self.route.joystick.center[0]), y=int(self.route.joystick.center[1]))
        end = self._joystick_end(center, direction, self.route.joystick.distance)
        screen_start = map_point_to_screen(center, self.config.recognition_size, actual_size)
        screen_end = map_point_to_screen(end, self.config.recognition_size, actual_size)
        self.callbacks.swipe(
            screen_start.x,
            screen_start.y,
            screen_end.x,
            screen_end.y,
            self.route.joystick.duration_ms,
        )

    @staticmethod
    def _joystick_end(center: Point, direction: RouteMove, distance: int) -> Point:
        if direction in {"wait", "stop"}:
            return center
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

    def _resolve_template_path(self, template: str):
        path = self.route.resolve_path(template)
        if path.exists():
            return path
        path = self.config.template_dir / template
        return path
