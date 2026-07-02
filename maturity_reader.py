from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - OpenCV fallback still works without Pillow.
    Image = None
    ImageDraw = None
    ImageFont = None

from vision import RecognitionSize, Rect, crop_roi, match_template, resize_to_recognition


DEFAULT_CARD_ROI = Rect(x=55, y=250, width=170, height=80)
DEFAULT_TEXT_ROI = Rect(x=0, y=8, width=170, height=46)
DEFAULT_ANCHOR_ROI = Rect(x=0, y=120, width=280, height=520)
TEMPLATE_SIZE = (16, 24)
DIGIT_SCORE_THRESHOLD = 0.58
TOMORROW_TEMPLATE_MATCH_THRESHOLD = 0.78

# 数字几何约束：成熟时间的四位数字“瘦高”，而卡片边框、水分条、问号按钮是大色块，
# “成熟/明天”等汉字近正方形。用宽高比 + 相对中位高度的容差把这些噪声整体剔除。
DIGIT_ASPECT_MAX = 0.92          # 数字宽/高上限（汉字≈1.0 会被排除）
DIGIT_HEIGHT_MIN_RATIO = 0.62    # 相对行内中位高度的下限
DIGIT_HEIGHT_MAX_RATIO = 1.45    # 相对行内中位高度的上限
DIGIT_MAX_HEIGHT_RATIO = 0.85    # 单个组件相对裁剪高度的上限（剔除整条边框/按钮）

# “明天”前缀几何判据（以稳健的几何特征为主，模板匹配仅作补充确认）：
# 跨分辨率/渲染时模板匹配的 CCOEFF 分数极不稳定，因此不再以其为唯一依据。
# 思路：在“首位数字左侧、与数字同基线”的范围内寻找“汉字字身”——
#   高度≈数字、且至少一个分量接近正方形（汉字宽 ≈ 高），
#   并要求其横向跨度足够、紧邻时间数字；以此区分边框竖线、实心噪声条、数字状窄条。
TOMORROW_HEIGHT_MIN_RATIO = 0.50  # 前缀分量高度下限（相对数字高，剔除碎噪点）
TOMORROW_HEIGHT_MAX_RATIO = 1.90  # 前缀分量高度上限（相对数字高，剔除贯穿边框）
TOMORROW_SQUARE_RATIO = 0.60      # 汉字字身：宽 ≥ 0.60×自身高（剎除细窄绝条/数字“1”状噪声）
TOMORROW_SPAN_RATIO = 1.20        # 前缀横向跨度 ≥ 1.20×字高（“明天”实测≈2.4×）
TOMORROW_GAP_RATIO = 1.10         # 前缀右缘距首位数字 ≤ 1.10×字高（必须紧邻时间，排除卡片美术）
TOMORROW_MIN_DENSITY = 0.20       # 笔画密度下限（剔除空白/极稀疏）
TOMORROW_MAX_DENSITY = 0.62       # 笔画密度上限（剔除实心色块/实心噪声竖条）


@dataclass(frozen=True)
class MaturityReadResult:
    found: bool
    time_text: str | None
    score: float
    day_offset: int | None = None
    anchor_found: bool = False


class DigitComponent(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray


class ClassifiedDigit(NamedTuple):
    component: DigitComponent
    digit: int
    score: float


def read_maturity_time(
    screenshot: np.ndarray,
    recognition_size: RecognitionSize,
    *,
    card_roi: Rect | None = None,
    anchor_template: np.ndarray | None = None,
    anchor_threshold: float = 0.72,
    anchor_roi: Rect | None = None,
    tomorrow_template: np.ndarray | None = None,
    tomorrow_threshold: float = TOMORROW_TEMPLATE_MATCH_THRESHOLD,
) -> MaturityReadResult:
    recognition = resize_to_recognition(screenshot, recognition_size)
    tomorrow_template_mask = _prepare_tomorrow_template(tomorrow_template)
    if anchor_template is not None:
        anchor_result = match_template(
            recognition,
            anchor_template,
            threshold=anchor_threshold,
            roi=anchor_roi or DEFAULT_ANCHOR_ROI,
            recognition_size=recognition_size,
            actual_size=recognition_size,
        )
        if anchor_result.found:
            text_roi = _text_roi_from_anchor(anchor_result.rect, recognition)
            text_crop, _ = crop_roi(recognition, text_roi)
            result = read_maturity_time_from_text_crop(
                text_crop,
                tomorrow_template=tomorrow_template_mask,
                tomorrow_threshold=tomorrow_threshold,
            )
            if result.found:
                return MaturityReadResult(
                    found=True,
                    time_text=result.time_text,
                    score=result.score,
                    day_offset=result.day_offset,
                    anchor_found=True,
                )

    crop, _ = crop_roi(recognition, card_roi or DEFAULT_CARD_ROI)
    return read_maturity_time_from_crop(
        crop,
        tomorrow_template=tomorrow_template_mask,
        tomorrow_threshold=tomorrow_threshold,
    )


def read_maturity_time_from_crop(
    crop: np.ndarray,
    *,
    tomorrow_template: np.ndarray | None = None,
    tomorrow_threshold: float = TOMORROW_TEMPLATE_MATCH_THRESHOLD,
) -> MaturityReadResult:
    text_crop, _ = crop_roi(crop, DEFAULT_TEXT_ROI)
    return read_maturity_time_from_text_crop(
        text_crop,
        tomorrow_template=tomorrow_template,
        tomorrow_threshold=tomorrow_threshold,
    )


def read_maturity_time_from_text_crop(
    text_crop: np.ndarray,
    *,
    tomorrow_template: np.ndarray | None = None,
    tomorrow_threshold: float = TOMORROW_TEMPLATE_MATCH_THRESHOLD,
) -> MaturityReadResult:
    gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 35, 180)
    # 只保留位于“数字行”的瘦高组件，边框/水分条/问号按钮/“成熟”等噪声在此被整体剔除。
    row = _digit_components(mask)
    if len(row) < 4:
        return MaturityReadResult(found=False, time_text=None, score=0.0)

    classified: list[ClassifiedDigit] = []
    for component in row:
        digit, score = _classify_digit(component.image)
        classified.append(ClassifiedDigit(component=component, digit=digit, score=score))
    median_height = float(np.median([item.component.height for item in classified]))

    # 在数字行内滑动 4 连窗口；用置信度 + 高度/间距一致性给每组打分，
    # 选出最像真实 HH:MM 的一组，抵抗汉字偏旁拼出的“伪时间”。
    best: tuple[float, str, float, list[ClassifiedDigit]] | None = None
    for start in range(0, len(classified) - 3):
        group = classified[start : start + 4]
        if any(item.score < DIGIT_SCORE_THRESHOLD for item in group):
            continue
        hour = group[0].digit * 10 + group[1].digit
        minute = group[2].digit * 10 + group[3].digit
        if hour > 23 or minute > 59:
            continue
        average_score = sum(item.score for item in group) / 4
        rank = average_score + _row_consistency_bonus(group, median_height)
        if best is None or rank > best[0]:
            best = (rank, f"{hour:02d}:{minute:02d}", average_score, group)

    if best is None:
        return MaturityReadResult(found=False, time_text=None, score=0.0)

    _, clock_text, average_score, group = best

    # 第一步：先独立判定信息卡是否带“明天”前缀（基于首位数字左侧的汉字几何）。
    # 必须在拼装最终时间文本之前完成，避免“明天14:10”被误判成今天。
    first_component = group[0].component
    digit_height = float(np.median([item.component.height for item in group]))
    digit_center_y = float(
        np.median([item.component.y + item.component.height / 2 for item in group])
    )
    has_tomorrow = _detect_tomorrow_prefix(
        mask,
        first_digit_x=first_component.x,
        digit_height=digit_height,
        digit_center_y=digit_center_y,
        tomorrow_template=tomorrow_template,
        tomorrow_threshold=tomorrow_threshold,
    )
    day_offset = 1 if has_tomorrow else 0

    # 第二步：在确定前缀后，拼出具体 HH:MM。
    prefix = "明天" if day_offset == 1 else ""
    return MaturityReadResult(
        found=True,
        time_text=f"{prefix}{clock_text}",
        score=average_score,
        day_offset=day_offset,
    )


def _row_consistency_bonus(group: list[ClassifiedDigit], median_height: float) -> float:
    """真实四位数字高度接近、横向间距均匀（中间因冒号略宽）。
    用这两个一致性指标给加分，把偏旁/噪声拼出的伪时间排到后面。"""
    heights = [item.component.height for item in group]
    height_dev = max(abs(h - median_height) for h in heights) / max(1.0, median_height)
    centers = [item.component.x + item.component.width / 2 for item in group]
    gaps = [centers[i + 1] - centers[i] for i in range(3)]
    gap_mean = sum(gaps) / 3
    gap_dev = (max(gaps) - min(gaps)) / max(1.0, gap_mean)
    bonus = 0.0
    if height_dev < 0.25:
        bonus += 0.15
    if gap_dev < 0.7:
        bonus += 0.10
    return bonus


def _text_roi_from_anchor(anchor: Rect, image: np.ndarray) -> Rect:
    center_x = anchor.x + anchor.width // 2
    x = max(0, center_x - 122)
    y = max(0, anchor.y - 104)
    width = min(255, image.shape[1] - x)
    height = min(64, image.shape[0] - y)
    return Rect(x=x, y=y, width=width, height=height)


def _digit_components(mask: np.ndarray) -> list[DigitComponent]:
    """从二值掩码中提取“成熟时间数字行”的数字组件。

    步骤：连通域 → 几何过滤（剔除整条边框/水分条/问号按钮等大色块）→
    取“瘦高”组件估计中位数字高度 → 行聚类取数字所在基线行 → 按 x 排序。
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    crop_height = mask.shape[0]
    components: list[DigitComponent] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < 8 or height < 6 or width < 2:
            continue
        # 整条边框/按钮往往贯穿整幅裁剪高度，水分条/色块往往很宽，直接剔除。
        if height > DIGIT_MAX_HEIGHT_RATIO * crop_height:
            continue
        if width > 1.1 * height:
            continue
        components.append(
            DigitComponent(
                x=x,
                y=y,
                width=width,
                height=height,
                image=mask[y : y + height, x : x + width],
            )
        )

    # 用“瘦高”组件（汉字≈正方形被排除）估计数字中位高度，再据此收紧高度带。
    digit_like = [c for c in components if c.width <= DIGIT_ASPECT_MAX * c.height]
    if len(digit_like) < 4:
        return []
    median_height = float(np.median([c.height for c in digit_like]))
    low = DIGIT_HEIGHT_MIN_RATIO * median_height
    high = DIGIT_HEIGHT_MAX_RATIO * median_height
    candidates = [c for c in digit_like if low <= c.height <= high]
    if len(candidates) < 4:
        return []

    row = _largest_baseline_row(candidates, tolerance=0.55 * median_height)
    row.sort(key=lambda c: c.x)
    return row


def _largest_baseline_row(
    candidates: list[DigitComponent], *, tolerance: float
) -> list[DigitComponent]:
    """四位成熟时间数字共享同一基线：按竖直中心聚类，返回成员最多的一行。"""
    best: list[DigitComponent] = []
    for anchor in candidates:
        anchor_cy = anchor.y + anchor.height / 2
        group = [c for c in candidates if abs((c.y + c.height / 2) - anchor_cy) <= tolerance]
        if len(group) > len(best):
            best = group
    return best


def _detect_tomorrow_prefix(
    mask: np.ndarray,
    *,
    first_digit_x: int,
    digit_height: float,
    digit_center_y: float,
    tomorrow_template: np.ndarray | None = None,
    tomorrow_threshold: float = TOMORROW_TEMPLATE_MATCH_THRESHOLD,
) -> bool:
    """判断成熟时间是否带“明天”前缀（几何为主、模板为辅）。

    在“首位数字左侧、与数字同基线”的范围内寻找汉字字身：
    - 高度落在 [0.5, 1.9]×数字高（剔除碎噪点与贯穿整高的边框）；
    - 至少一个分量近正方形（宽 ≥ 0.70×字高），这是汉字与“数字状窄条/边框竖线”的关键区别；
    - 整体横向跨度 ≥ 1.20×字高，且右缘紧邻时间数字（缝隙 ≤ 1.10×字高）；
    - 笔画密度落在 [0.20, 0.62]（剔除实心色块/实心噪声竖条与空白）。
    若几何线索不足但提供了模板且明确命中，也判为有前缀（兼容偏旁被切碎的情况）。
    """
    if digit_height <= 0:
        return False
    # 左侧需要至少能放下一个汉字的空间，否则不可能有“明天”。
    if first_digit_x < 0.9 * digit_height:
        return False

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height_min = TOMORROW_HEIGHT_MIN_RATIO * digit_height
    height_max = TOMORROW_HEIGHT_MAX_RATIO * digit_height
    chars: list[tuple[int, int, int, int]] = []  # (x, y, w, h)
    has_square = False
    for index in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[index])
        # 以“分量中心在首位数字左侧”归属前缀：既纳入紧贴数字的“天”，又自然排除首位数字本身。
        if x + width / 2 >= first_digit_x:
            continue
        if height < height_min or height > height_max:  # 剔除碎噪点与贯穿边框
            continue
        if abs((y + height / 2) - digit_center_y) > digit_height:  # 与数字同基线
            continue
        chars.append((x, y, width, height))
        if width >= TOMORROW_SQUARE_RATIO * height:  # 近正方形 = 汉字字身（用组件自身高，防渲染偏小时失效）
            has_square = True
    if not chars:
        return False

    left = min(c[0] for c in chars)
    right = max(c[0] + c[2] for c in chars)
    top = min(c[1] for c in chars)
    bottom = max(c[1] + c[3] for c in chars)
    # 前缀必须紧邻时间数字，排除卡片左侧美术/图标误触发。
    if first_digit_x - right > TOMORROW_GAP_RATIO * digit_height:
        return False
    if (right - left) < TOMORROW_SPAN_RATIO * digit_height:
        return False

    cluster = mask[max(0, top):bottom, max(0, left):right]
    density = float(np.count_nonzero(cluster)) / float(max(1, cluster.size))
    if density < TOMORROW_MIN_DENSITY or density > TOMORROW_MAX_DENSITY:
        return False

    if has_square:
        return True
    # 几何无近正方形分量时，退而求其次：模板明确命中才认定（偏旁被切碎的边缘情况）。
    if tomorrow_template is not None:
        prefix_region = mask[max(0, top):bottom, 0:first_digit_x]
        return _match_tomorrow_template(prefix_region, tomorrow_template, tomorrow_threshold)
    return False


def _prepare_tomorrow_template(template: np.ndarray | None) -> np.ndarray | None:
    if template is None:
        return None
    if template.ndim == 3:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray = template
    mask = cv2.inRange(gray, 35, 190)
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width < 18 or height < 10:
        return None
    return mask[y : y + height, x : x + width]


def _match_tomorrow_template(prefix: np.ndarray, template: np.ndarray, threshold: float) -> bool:
    if prefix.shape[0] < template.shape[0] or prefix.shape[1] < template.shape[1]:
        return False
    if np.count_nonzero(prefix) < max(12, np.count_nonzero(template) // 5):
        return False
    result = cv2.matchTemplate(prefix, template, cv2.TM_CCOEFF_NORMED)
    _, best_score, _, _ = cv2.minMaxLoc(result)
    return best_score >= threshold


def _classify_digit(component: np.ndarray) -> tuple[int, float]:
    sample = _normalize_binary(component)
    best_digit = 0
    best_score = -1.0
    for digit, templates in _digit_templates().items():
        for template in templates:
            score = _iou(sample, template)
            if score > best_score:
                best_digit = digit
                best_score = score
    return best_digit, best_score


def _normalize_binary(image: np.ndarray, target_size: tuple[int, int] = TEMPLATE_SIZE) -> np.ndarray:
    binary = (image > 0).astype(np.uint8) * 255
    points = cv2.findNonZero(binary)
    if points is None:
        return np.zeros((target_size[1], target_size[0]), dtype=bool)
    x, y, width, height = cv2.boundingRect(points)
    tight = binary[y : y + height, x : x + width]
    resized = cv2.resize(tight, target_size, interpolation=cv2.INTER_AREA)
    return resized > 80


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


@lru_cache(maxsize=1)
def _digit_templates() -> dict[int, tuple[np.ndarray, ...]]:
    templates: dict[int, list[np.ndarray]] = {digit: [] for digit in range(10)}
    if Image is not None and ImageDraw is not None and ImageFont is not None:
        for font_path in _font_paths():
            if not font_path.exists():
                continue
            for font_size in range(16, 29):
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                except Exception:
                    continue
                for digit in range(10):
                    image = Image.new("L", (44, 44), 0)
                    draw = ImageDraw.Draw(image)
                    draw.text((2, 2), str(digit), font=font, fill=255)
                    templates[digit].append(_normalize_binary(np.array(image)))

    if not any(templates.values()):
        for scale in (0.45, 0.5, 0.55, 0.6, 0.65):
            for thickness in (1, 2):
                for digit in range(10):
                    image = np.zeros((44, 44), dtype=np.uint8)
                    cv2.putText(
                        image,
                        str(digit),
                        (2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    templates[digit].append(_normalize_binary(image))

    return {digit: tuple(values) for digit, values in templates.items()}


def _font_paths() -> list[Path]:
    return [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
