from datetime import datetime

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from maturity_reader import read_maturity_time
from run_scheduler import compute_manual_next_run, compute_next_run
from vision import RecognitionSize, Rect


def _synthetic_maturity_screen(text: str) -> np.ndarray:
    screen = np.full((720, 1280, 3), 245, dtype=np.uint8)
    card = screen[120:340, 25:225]
    card[:] = (248, 244, 229)
    image = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 27)
    draw.text((74, 261), f"{text}成熟", fill=(90, 84, 78), font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _draw_synthetic_shovel(screen: np.ndarray) -> np.ndarray:
    cv2.rectangle(screen, (78, 360), (154, 410), (64, 190, 250), thickness=-1)
    cv2.rectangle(screen, (78, 360), (154, 410), (40, 150, 220), thickness=2)
    cv2.circle(screen, (112, 383), 10, (70, 90, 160), thickness=3)
    cv2.line(screen, (119, 389), (136, 372), (70, 90, 160), thickness=5)
    return screen[360:410, 78:154].copy()


def _synthetic_anchor_maturity_screen(text: str) -> tuple[np.ndarray, np.ndarray]:
    screen = np.full((720, 1280, 3), 245, dtype=np.uint8)
    card = screen[120:420, 25:225]
    card[:] = (248, 244, 229)
    shovel_template = _draw_synthetic_shovel(screen)

    image = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 27)
    draw.text((56, 281), text, fill=(90, 84, 78), font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), shovel_template


def _synthetic_tomorrow_template() -> np.ndarray:
    screen, _ = _synthetic_anchor_maturity_screen("明天02:53成熟")
    return screen[281:311, 56:108].copy()


def _synthetic_noisy_card(text: str) -> tuple[np.ndarray, np.ndarray]:
    """模拟真实卡片：时间文字 + 边框 + 水分条 + 问号按钮 + 铲子锚点等噪声块。
    复现旧逻辑被边框/按钮/“成熟”噪声带偏的场景。"""
    screen = np.full((720, 1280, 3), 245, dtype=np.uint8)
    card = screen[120:420, 25:225]
    card[:] = (248, 244, 229)
    shovel_template = _draw_synthetic_shovel(screen)
    # 卡片整条左边框（高瘦但贯穿整高）——旧逻辑会把它误当数字。
    cv2.rectangle(screen, (28, 200, ), (33, 320), (120, 110, 100), thickness=-1)
    # 时间下方的蓝色水分条（很宽）。
    cv2.rectangle(screen, (40, 320), (190, 338), (200, 140, 60), thickness=-1)
    # 右下角问号按钮（大圆块）。
    cv2.circle(screen, (205, 330), 16, (90, 84, 78), thickness=-1)

    image = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 27)
    # “HH:MM成熟”：成熟两字紧跟数字之后，是旧逻辑最常见的误判来源。
    draw.text((56, 281), f"{text}成熟", fill=(90, 84, 78), font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), shovel_template


def test_read_maturity_time_ignores_border_bar_button_and_chengshu_noise():
    screen, shovel_template = _synthetic_noisy_card("13:04")
    tomorrow_template = _synthetic_tomorrow_template()

    result = read_maturity_time(
        screen,
        RecognitionSize(width=1280, height=720),
        anchor_template=shovel_template,
        anchor_roi=Rect(x=0, y=120, width=260, height=360),
        tomorrow_template=tomorrow_template,
    )

    assert result.found is True
    assert result.time_text == "13:04"
    assert result.day_offset == 0


def test_read_maturity_time_reads_tomorrow_prefix_with_card_noise():
    screen, shovel_template = _synthetic_noisy_card("明天02:53")
    tomorrow_template = _synthetic_tomorrow_template()

    result = read_maturity_time(
        screen,
        RecognitionSize(width=1280, height=720),
        anchor_template=shovel_template,
        anchor_roi=Rect(x=0, y=120, width=260, height=360),
        tomorrow_template=tomorrow_template,
    )

    assert result.found is True
    assert result.time_text == "明天02:53"
    assert result.day_offset == 1


def test_read_maturity_time_reads_tomorrow_1410_regression():
    """回归：曾把“明天14:10”误判成今天“14:10”，导致 32h 作物剩余时间从 ~27h 误算成 ~3h。"""
    screen, shovel_template = _synthetic_noisy_card("明天14:10")
    tomorrow_template = _synthetic_tomorrow_template()

    result = read_maturity_time(
        screen,
        RecognitionSize(width=1280, height=720),
        anchor_template=shovel_template,
        anchor_roi=Rect(x=0, y=120, width=260, height=360),
        tomorrow_template=tomorrow_template,
    )

    assert result.found is True
    assert result.time_text == "明天14:10"
    assert result.day_offset == 1


def test_read_maturity_time_parses_fixed_card_time():
    screen = _synthetic_maturity_screen("13:04")

    result = read_maturity_time(screen, RecognitionSize(width=1280, height=720))

    assert result.found is True
    assert result.time_text == "13:04"


def test_read_maturity_time_uses_shovel_anchor_for_tomorrow_prefixed_time():
    screen, shovel_template = _synthetic_anchor_maturity_screen("明天02:53成熟")
    tomorrow_template = _synthetic_tomorrow_template()

    result = read_maturity_time(
        screen,
        RecognitionSize(width=1280, height=720),
        anchor_template=shovel_template,
        anchor_roi=Rect(x=0, y=120, width=260, height=360),
        tomorrow_template=tomorrow_template,
    )

    assert result.found is True
    assert result.time_text == "明天02:53"


def test_read_maturity_time_does_not_guess_tomorrow_from_left_side_noise():
    screen, shovel_template = _synthetic_anchor_maturity_screen("07:51成熟")
    tomorrow_template = _synthetic_tomorrow_template()
    cv2.rectangle(screen, (18, 284), (29, 306), (90, 84, 78), thickness=-1)
    cv2.rectangle(screen, (36, 284), (47, 306), (90, 84, 78), thickness=-1)

    result = read_maturity_time(
        screen,
        RecognitionSize(width=1280, height=720),
        anchor_template=shovel_template,
        anchor_roi=Rect(x=0, y=120, width=260, height=360),
        tomorrow_template=tomorrow_template,
    )

    assert result.found is True
    assert result.time_text == "07:51"


def test_compute_next_run_uses_five_hour_default_without_maturity_time():
    now = datetime(2026, 6, 13, 6, 0, 0)

    result = compute_next_run(now=now, maturity_time=None)

    assert result.next_run == datetime(2026, 6, 13, 11, 0, 0)
    assert result.reason == "default_5h"


def test_compute_next_run_uses_maturity_minus_ten_minutes_when_sooner():
    now = datetime(2026, 6, 13, 8, 20, 0)

    result = compute_next_run(now=now, maturity_time="13:04")

    assert result.next_run == datetime(2026, 6, 13, 12, 54, 0)
    assert result.reason == "maturity_time"


def test_compute_next_run_does_not_schedule_in_the_past():
    now = datetime(2026, 6, 13, 13, 0, 0)

    result = compute_next_run(now=now, maturity_time="13:04")

    assert result.next_run == datetime(2026, 6, 13, 13, 1, 0)
    assert result.reason == "maturity_time_immediate"


def test_compute_next_run_respects_tomorrow_prefix():
    now = datetime(2026, 6, 13, 23, 0, 0)

    result = compute_next_run(now=now, maturity_time="明天00:30")

    assert result.next_run == datetime(2026, 6, 14, 0, 20, 0)
    assert result.reason == "maturity_time"


def test_compute_manual_next_run_uses_same_day_when_future():
    now = datetime(2026, 6, 13, 8, 20, 0)

    result = compute_manual_next_run(now=now, start_time="12:30")

    assert result.next_run == datetime(2026, 6, 13, 12, 30, 0)
    assert result.reason == "manual_time"


def test_compute_manual_next_run_rolls_to_tomorrow_when_time_has_passed():
    now = datetime(2026, 6, 13, 20, 0, 0)

    result = compute_manual_next_run(now=now, start_time="06:30")

    assert result.next_run == datetime(2026, 6, 14, 6, 30, 0)
    assert result.reason == "manual_time"


def test_compute_manual_next_run_rejects_invalid_clock_text():
    with pytest.raises(ValueError):
        compute_manual_next_run(now=datetime(2026, 6, 13, 8, 20, 0), start_time="25:61")
