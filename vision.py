from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class RecognitionSize:
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_sequence(cls, values: Sequence[int] | None) -> "Rect | None":
        if values is None:
            return None
        if len(values) != 4:
            raise ValueError("ROI must contain exactly four integers: x, y, width, height")
        return cls(x=int(values[0]), y=int(values[1]), width=int(values[2]), height=int(values[3]))

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width // 2, y=self.y + self.height // 2)


@dataclass(frozen=True)
class MatchResult:
    found: bool
    score: float
    center: Point
    rect: Rect
    screen_center: Point
    screen_rect: Rect


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_to_recognition(image: np.ndarray, recognition_size: RecognitionSize) -> np.ndarray:
    image = _ensure_bgr(image)
    if image.shape[1] == recognition_size.width and image.shape[0] == recognition_size.height:
        return image.copy()
    return cv2.resize(
        image,
        (recognition_size.width, recognition_size.height),
        interpolation=cv2.INTER_AREA,
    )


def map_point_to_screen(
    point: Point,
    recognition_size: RecognitionSize,
    actual_size: RecognitionSize,
) -> Point:
    return Point(
        x=int(round(point.x * actual_size.width / recognition_size.width)),
        y=int(round(point.y * actual_size.height / recognition_size.height)),
    )


def map_rect_to_screen(
    rect: Rect,
    recognition_size: RecognitionSize,
    actual_size: RecognitionSize,
) -> Rect:
    top_left = map_point_to_screen(Point(rect.x, rect.y), recognition_size, actual_size)
    bottom_right = map_point_to_screen(
        Point(rect.x + rect.width, rect.y + rect.height),
        recognition_size,
        actual_size,
    )
    return Rect(
        x=top_left.x,
        y=top_left.y,
        width=max(0, bottom_right.x - top_left.x),
        height=max(0, bottom_right.y - top_left.y),
    )


def crop_roi(image: np.ndarray, roi: Rect | None) -> tuple[np.ndarray, Rect]:
    image = _ensure_bgr(image)
    if roi is None:
        full = Rect(x=0, y=0, width=image.shape[1], height=image.shape[0])
        return image.copy(), full

    x1 = min(max(roi.x, 0), image.shape[1])
    y1 = min(max(roi.y, 0), image.shape[0])
    x2 = min(max(roi.x + roi.width, 0), image.shape[1])
    y2 = min(max(roi.y + roi.height, 0), image.shape[0])
    clipped = Rect(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))
    return image[y1:y2, x1:x2].copy(), clipped


def save_debug_match(image: np.ndarray, result: MatchResult, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    debug = _ensure_bgr(image).copy()
    if result.found:
        x1, y1 = result.rect.x, result.rect.y
        x2, y2 = result.rect.x + result.rect.width, result.rect.y + result.rect.height
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(debug, (result.center.x, result.center.y), 4, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"{result.score:.3f}",
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), debug)


def match_template(
    screenshot_bgr: np.ndarray,
    template_bgr: np.ndarray,
    *,
    threshold: float,
    roi: Rect | None = None,
    recognition_size: RecognitionSize = RecognitionSize(),
    actual_size: RecognitionSize | None = None,
    debug_path: str | Path | None = None,
) -> MatchResult:
    screenshot_bgr = _ensure_bgr(screenshot_bgr)
    template_bgr = _ensure_bgr(template_bgr)
    actual = actual_size or RecognitionSize(width=screenshot_bgr.shape[1], height=screenshot_bgr.shape[0])
    recognition_image = resize_to_recognition(screenshot_bgr, recognition_size)
    search_image, clipped_roi = crop_roi(recognition_image, roi)

    if (
        search_image.size == 0
        or template_bgr.shape[0] > search_image.shape[0]
        or template_bgr.shape[1] > search_image.shape[1]
    ):
        rect = Rect(x=clipped_roi.x, y=clipped_roi.y, width=template_bgr.shape[1], height=template_bgr.shape[0])
        result = MatchResult(
            found=False,
            score=0.0,
            center=rect.center,
            rect=rect,
            screen_center=map_point_to_screen(rect.center, recognition_size, actual),
            screen_rect=map_rect_to_screen(rect, recognition_size, actual),
        )
        if debug_path is not None:
            save_debug_match(recognition_image, result, debug_path)
        return result

    scores = cv2.matchTemplate(search_image, template_bgr, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_location = cv2.minMaxLoc(scores)
    rect = Rect(
        x=clipped_roi.x + int(max_location[0]),
        y=clipped_roi.y + int(max_location[1]),
        width=int(template_bgr.shape[1]),
        height=int(template_bgr.shape[0]),
    )
    result = MatchResult(
        found=float(max_score) >= threshold,
        score=float(max_score),
        center=rect.center,
        rect=rect,
        screen_center=map_point_to_screen(rect.center, recognition_size, actual),
        screen_rect=map_rect_to_screen(rect, recognition_size, actual),
    )
    if debug_path is not None:
        save_debug_match(recognition_image, result, debug_path)
    return result
