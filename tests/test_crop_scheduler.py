from datetime import datetime, timedelta

from run_scheduler import (
    build_crop_optimization_plan,
    compute_crop_next_run,
    format_crop_optimization_report,
    parse_crop_type,
    remaining_until_clock_time,
)


def test_parse_crop_type_accepts_supported_values():
    assert parse_crop_type("1h") == 1
    assert parse_crop_type("8h") == 8
    assert parse_crop_type("16h") == 16
    assert parse_crop_type("32h") == 32
    assert parse_crop_type(32) == 32


def test_crop_scheduler_reaches_32h_minimum_harvest_time_in_three_waters():
    start = datetime(2026, 6, 15, 6, 0, 0)

    first = compute_crop_next_run(now=start, crop_type="32h")
    second = compute_crop_next_run(
        now=first.next_run,
        crop_type="32h",
        last_run_at=start,
        last_remaining_after_run=first.remaining_after_run,
    )
    third = compute_crop_next_run(
        now=second.next_run,
        crop_type="32h",
        last_run_at=first.next_run,
        last_remaining_after_run=second.remaining_after_run,
    )

    assert first.interval == timedelta(hours=10, minutes=40)
    assert second.interval == timedelta(hours=2, minutes=8)
    assert third.interval == timedelta(hours=10, minutes=40)
    assert third.next_run - start == timedelta(hours=23, minutes=28)
    assert third.phase == "harvest_window"


def test_crop_scheduler_uses_remaining_time_threshold_for_final_water():
    now = datetime(2026, 6, 15, 6, 0, 0)

    result = compute_crop_next_run(
        now=now,
        crop_type="32h",
        observed_remaining=timedelta(hours=10),
    )

    assert result.interval == timedelta(hours=8)
    assert result.next_run == datetime(2026, 6, 15, 14, 0, 0)
    assert result.phase == "harvest_window"


def test_remaining_until_clock_time_rolls_to_tomorrow_when_needed():
    now = datetime(2026, 6, 15, 23, 50, 0)

    assert remaining_until_clock_time(now=now, clock_time="00:10") == timedelta(minutes=20)


def test_remaining_until_clock_time_respects_tomorrow_prefix_even_when_today_is_future():
    now = datetime(2026, 6, 15, 1, 0, 0)

    assert remaining_until_clock_time(now=now, clock_time="明天02:53") == timedelta(days=1, hours=1, minutes=53)


def test_crop_scheduler_resets_to_new_cycle_after_harvest():
    start = datetime(2026, 6, 15, 6, 0, 0)
    next_cycle = datetime(2026, 6, 16, 5, 28, 1)

    result = compute_crop_next_run(
        now=next_cycle,
        crop_type="32h",
        last_run_at=start,
        last_remaining_after_run=timedelta(hours=13, minutes=20),
    )

    assert result.reason == "new_cycle_after_harvest"
    assert result.remaining_after_run == timedelta(hours=29, minutes=20)
    assert result.interval == timedelta(hours=10, minutes=40)


def test_crop_scheduler_harvests_naturally_when_too_close_to_maturity():
    # 复现 bug：32h 作物只剩 15 分钟时，0.8R=12min 低于最低浇水间隔 T/30=64min，
    # 旧逻辑把下次启动抬到 64min（晚于成熟）。新逻辑应判定自然成熟，并提前 5min 启动，
    # 让脚本到按钮前等待，成熟（now+15min）即收，规避偷菜窗口。
    now = datetime(2026, 6, 17, 8, 33, 0)

    result = compute_crop_next_run(
        now=now,
        crop_type="32h",
        observed_remaining=timedelta(minutes=15),
    )

    assert result.phase == "natural_harvest"
    assert result.harvest_at == now + timedelta(minutes=15)
    assert result.interval == timedelta(minutes=10)  # 15min - 5min 提前量
    assert result.next_run == now + timedelta(minutes=10)
    assert result.predicted_remaining_after_next_run == timedelta(0)


def test_crop_scheduler_natural_harvest_lead_does_not_go_negative():
    # 剩余比提前量还短时，立即启动（间隔 0），但仍记录真实成熟时刻。
    now = datetime(2026, 6, 17, 8, 33, 0)

    result = compute_crop_next_run(
        now=now,
        crop_type="32h",
        observed_remaining=timedelta(minutes=3),
        harvest_lead=timedelta(minutes=5),
    )

    assert result.phase == "natural_harvest"
    assert result.interval == timedelta(0)
    assert result.next_run == now
    assert result.harvest_at == now + timedelta(minutes=3)


def test_crop_scheduler_next_run_never_after_observed_maturity():
    # 不论何时接管，下次启动都不能晚于识别到的成熟时刻。
    now = datetime(2026, 6, 17, 8, 0, 0)
    for crop in ("1h", "8h", "16h", "32h"):
        total_minutes = parse_crop_type(crop) * 60
        for ratio in (0.01, 0.02, 0.04, 0.1, 0.3, 0.41, 0.45, 0.7, 0.95):
            observed = timedelta(minutes=total_minutes * ratio)
            result = compute_crop_next_run(now=now, crop_type=crop, observed_remaining=observed)
            assert result.next_run <= now + observed, (crop, ratio, result.phase)


def test_crop_scheduler_final_water_uses_minimum_interval_boundary():
    # 剩余正好 T/24（32h→80min）：0.8R=64min=T/30，仍可做最后一浇。
    now = datetime(2026, 6, 17, 8, 0, 0)

    result = compute_crop_next_run(
        now=now,
        crop_type="32h",
        observed_remaining=timedelta(minutes=80),
    )

    assert result.phase == "harvest_window"
    assert result.interval == timedelta(minutes=64)
    assert result.next_run <= now + timedelta(minutes=80)


def test_crop_optimization_plan_explains_16h_fastest_harvest_path():
    start = datetime(2026, 6, 16, 6, 0, 0)

    plan = build_crop_optimization_plan(now=start, crop_type="16h")
    report = format_crop_optimization_report(plan)

    assert [step.run_at for step in plan.steps] == [
        datetime(2026, 6, 16, 11, 20, 0),
        datetime(2026, 6, 16, 12, 24, 0),
        datetime(2026, 6, 16, 17, 44, 0),
    ]
    assert plan.steps[0].predicted_remaining_after_run == timedelta(hours=8)
    assert plan.steps[1].predicted_remaining_after_run == timedelta(hours=6, minutes=40)
    assert plan.steps[2].harvest is True
    assert "16h作物" in report
    assert "11:20定时启动" in report
    assert "17:44收获作物最快" in report
