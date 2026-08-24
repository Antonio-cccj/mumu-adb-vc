from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

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
RouteMove = JoystickDirection | Literal["wait", "stop"]
RouteMode = Literal["keyframe", "fixed_step"]


class TemplateGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    templates: list[str] = Field(default_factory=list)
    threshold: float = Field(default=0.8, ge=-1.0, le=1.0)
    roi: list[int] | None = None
    wait_ms: int = Field(default=500, ge=0)

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 4:
            raise ValueError("roi must be [x, y, width, height]")
        return value


class RouteJoystick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center: list[int] = Field(default_factory=lambda: [184, 505], min_length=2, max_length=2)
    distance: int = Field(default=72, ge=1)
    duration_ms: int = Field(default=280, ge=1)
    wait_ms: int = Field(default=650, ge=0)


class FixedStepRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: JoystickDirection = "up_left"
    step_wait_ms: int = Field(default=1800, ge=0)
    settle_wait_ms: int = Field(default=500, ge=0)
    success_poll_interval_ms: int = Field(default=500, ge=1)
    reset_after_ms: int = Field(default=300000, ge=1)


class RouteKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    image: str
    move: RouteMove = "wait"
    correction_moves: list[JoystickDirection] = Field(default_factory=list)
    stuck_after: int | None = Field(default=None, ge=1)
    min_score: float = Field(default=12.0, ge=0.0)
    roi: list[int] | None = None

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 4:
            raise ValueError("roi must be [x, y, width, height]")
        return value


class NavigationRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mode: RouteMode = "keyframe"
    scene_roi: list[int] | None = None
    joystick: RouteJoystick = Field(default_factory=RouteJoystick)
    fixed_step: FixedStepRoute = Field(default_factory=FixedStepRoute)
    success: TemplateGroup = Field(default_factory=TemplateGroup)
    interrupt: TemplateGroup = Field(default_factory=TemplateGroup)
    reset: TemplateGroup = Field(default_factory=TemplateGroup)
    stuck_after: int = Field(default=2, ge=1)
    max_stuck_attempts: int = Field(default=8, ge=1)
    recovery_moves: list[JoystickDirection] = Field(default_factory=lambda: ["left", "up_left", "up", "right"])
    keyframes: list[RouteKeyframe] = Field(..., min_length=1)

    _base_dir: Path = PrivateAttr(default=Path("."))

    @field_validator("scene_roi")
    @classmethod
    def validate_scene_roi(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 4:
            raise ValueError("scene_roi must be [x, y, width, height]")
        return value

    def bind_base_dir(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        return self._base_dir / path


def load_navigation_route(path: str | Path) -> NavigationRoute:
    route_path = Path(path)
    with route_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Route file must contain a YAML mapping: {route_path}")
    route = NavigationRoute.model_validate(raw)
    route.bind_base_dir(route_path.parent)
    return route
