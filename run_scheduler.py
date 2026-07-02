from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re


@dataclass(frozen=True)
class NextRunPlan:
    next_run: datetime
    reason: str


@dataclass(frozen=True)
class CropNextRunPlan:
    next_run: datetime
    reason: str
    crop_hours: int
    interval: timedelta
    remaining_after_run: timedelta
    predicted_remaining_after_next_run: timedelta
    phase: str
    # 自然成熟收获时的作物成熟墙钟时刻；其它阶段为 None。
    # next_run 会比它提前 harvest_lead，让脚本在按钮前等待，规避偷菜窗口。
    harvest_at: datetime | None = None


@dataclass(frozen=True)
class CropOptimizationStep:
    index: int
    run_at: datetime
    predicted_remaining_after_run: timedelta
    phase: str
    harvest: bool
    # 自然成熟步骤的成熟时刻（run_at 为提前到位的启动时刻）。
    harvest_at: datetime | None = None


@dataclass(frozen=True)
class CropOptimizationPlan:
    crop_hours: int
    detected_maturity_time: str | None
    observed_remaining: timedelta | None
    steps: tuple[CropOptimizationStep, ...]


SUPPORTED_CROP_HOURS = {1, 8, 16, 32}

# 作物自然成熟时，必须提前到农场“一键务农”按钮前等待的时间。
# 启动时刻 = 成熟时刻 - 该提前量，覆盖脚本启动与导航耗时，避免成熟后被偷菜。
DEFAULT_HARVEST_LEAD = timedelta(minutes=5)


def _round_delta(value: timedelta) -> timedelta:
    return timedelta(seconds=max(0, round(value.total_seconds())))


def format_duration(value: timedelta) -> str:
    total_minutes = max(0, round(value.total_seconds() / 60))
    days, remainder = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes or not parts:
        parts.append(f"{minutes}分钟")
    return "".join(parts)


def parse_crop_type(value: str | int) -> int:
    if isinstance(value, int):
        crop_hours = value
    else:
        crop_hours = int(value.strip().lower().removesuffix("h"))
    if crop_hours not in SUPPORTED_CROP_HOURS:
        raise ValueError("crop type must be one of: 1h, 8h, 16h, 32h")
    return crop_hours


def compute_crop_next_run(
    *,
    now: datetime,
    crop_type: str | int,
    last_run_at: datetime | None = None,
    last_remaining_after_run: timedelta | None = None,
    observed_remaining: timedelta | None = None,
    harvest_lead: timedelta = DEFAULT_HARVEST_LEAD,
) -> CropNextRunPlan:
    crop_hours = parse_crop_type(crop_type)
    total = timedelta(hours=crop_hours)
    initial_remaining = _round_delta(total * (11 / 12))
    threshold = _round_delta(total * (5 / 12))
    maintenance_window = _round_delta(total / 3)
    minimum_interval = _round_delta(total / 30)

    if observed_remaining is not None:
        remaining_after_run = _round_delta(observed_remaining)
        reason = "observed_maturity"
    elif last_run_at is not None and last_remaining_after_run is not None:
        elapsed = max(timedelta(0), now - last_run_at)
        remaining_after_run = _round_delta(last_remaining_after_run - elapsed * 1.25)
        if remaining_after_run <= timedelta(0):
            remaining_after_run = initial_remaining
            reason = "new_cycle_after_harvest"
        else:
            reason = "continue_cycle"
    else:
        remaining_after_run = initial_remaining
        reason = "new_cycle"

    harvest_at: datetime | None = None
    if remaining_after_run <= threshold:
        # 进入收获窗口：再浇最后一次水即可正好成熟。浇水按 1.25 倍速推进
        # （等待 0.8R 后浇水，自然消耗 0.8R + 浇水折算 0.2R = R，刚好归零）。
        final_interval = _round_delta(remaining_after_run * 0.8)
        if final_interval >= minimum_interval:
            interval = final_interval
            phase = "harvest_window"
        else:
            # 距成熟过近（剩余 < T/24）：连最低浇水间隔 T/30 都赶不上自然成熟，
            # 再浇水没有意义。此时作物会在 now+R 自然成熟，存在“成熟后未收获被偷”的风险，
            # 因此把启动时刻提前 harvest_lead，让脚本先到按钮前等待，到点立刻收获。
            harvest_at = now + remaining_after_run
            interval = max(timedelta(0), remaining_after_run - harvest_lead)
            phase = "natural_harvest"
    else:
        threshold_interval = _round_delta((remaining_after_run - threshold) * 0.8)
        if threshold_interval <= maintenance_window:
            # 一步即可把剩余压到阈值，下一次就是最后一浇；注意不得低于最低浇水间隔。
            interval = max(threshold_interval, minimum_interval)
            phase = "enter_threshold"
        else:
            # 维持期：每隔 T/3 浇一次水保持湿润，使成熟整体以 1.25 倍速推进。
            interval = maintenance_window
            phase = "keep_water"

    if phase == "natural_harvest":
        # 该次运行最终会收获作物（先到位等待、成熟后点击），剩余归零。
        predicted_remaining = timedelta(0)
    else:
        # 浇水期成熟以 1.25 倍速推进（自然 1 倍 + 浇水折算 0.25 倍）。
        predicted_remaining = _round_delta(remaining_after_run - interval * 1.25)
    return CropNextRunPlan(
        next_run=now + interval,
        reason=reason,
        crop_hours=crop_hours,
        interval=interval,
        remaining_after_run=remaining_after_run,
        predicted_remaining_after_next_run=predicted_remaining,
        phase=phase,
        harvest_at=harvest_at,
    )


def build_crop_optimization_plan(
    *,
    now: datetime,
    crop_type: str | int,
    maturity_time: str | None = None,
    last_run_at: datetime | None = None,
    last_remaining_after_run: timedelta | None = None,
    max_steps: int = 6,
    harvest_lead: timedelta = DEFAULT_HARVEST_LEAD,
) -> CropOptimizationPlan:
    crop_hours = parse_crop_type(crop_type)
    observed_remaining = remaining_until_clock_time(now=now, clock_time=maturity_time) if maturity_time else None
    current_plan = compute_crop_next_run(
        now=now,
        crop_type=crop_hours,
        last_run_at=last_run_at,
        last_remaining_after_run=last_remaining_after_run,
        observed_remaining=observed_remaining,
        harvest_lead=harvest_lead,
    )
    steps: list[CropOptimizationStep] = []
    for index in range(1, max_steps + 1):
        predicted = current_plan.predicted_remaining_after_next_run
        harvest = predicted <= timedelta(0)
        steps.append(
            CropOptimizationStep(
                index=index,
                run_at=current_plan.next_run,
                predicted_remaining_after_run=predicted,
                phase=current_plan.phase,
                harvest=harvest,
                harvest_at=current_plan.harvest_at,
            )
        )
        if harvest:
            break
        current_plan = compute_crop_next_run(
            now=current_plan.next_run,
            crop_type=crop_hours,
            observed_remaining=predicted,
            harvest_lead=harvest_lead,
        )
    return CropOptimizationPlan(
        crop_hours=crop_hours,
        detected_maturity_time=maturity_time,
        observed_remaining=observed_remaining,
        steps=tuple(steps),
    )


def format_crop_optimization_report(plan: CropOptimizationPlan) -> str:
    if plan.detected_maturity_time and plan.observed_remaining is not None:
        detected = f"检测到成熟时间{plan.detected_maturity_time}，剩余{format_duration(plan.observed_remaining)}"
    else:
        detected = "未读取到成熟时间，按作物类型理论最优解推算"
    lines = [f"{plan.crop_hours}h作物，{detected}", "一键务农最优解："]
    for step in plan.steps:
        run_text = step.run_at.strftime("%H:%M")
        if step.harvest and step.phase == "natural_harvest" and step.harvest_at is not None:
            # 自然成熟：提前到位等待，成熟即收，避免成熟后被偷菜。
            harvest_text = step.harvest_at.strftime("%H:%M")
            lines.append(f"{step.index}. {run_text}提前到按钮前等待，{harvest_text}自然成熟收获")
        elif step.harvest:
            lines.append(f"{step.index}. {run_text}收获作物最快")
        else:
            remaining_text = format_duration(step.predicted_remaining_after_run)
            lines.append(f"{step.index}. {run_text}定时启动，预计剩余成熟时间{remaining_text}")
    return "\n".join(lines)


def remaining_until_clock_time(*, now: datetime, clock_time: str) -> timedelta | None:
    try:
        target = _resolve_maturity_datetime(now=now, value=clock_time)
    except (TypeError, ValueError):
        return None
    return target - now


def _parse_clock_time(value: str) -> tuple[int, int]:
    normalized = value.strip().replace("：", ":")
    hour_text, minute_text = normalized.split(":", maxsplit=1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("clock time must be HH:MM")
    return hour, minute


def _parse_maturity_clock(value: str) -> tuple[int, int, int | None]:
    normalized = value.strip().replace("：", ":").replace("成熟", "")
    day_offset: int | None = None
    if "明天" in normalized:
        day_offset = 1
        normalized = normalized.replace("明天", "")
    elif "今天" in normalized:
        day_offset = 0
        normalized = normalized.replace("今天", "")

    match = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", normalized)
    if match is None:
        raise ValueError("maturity time must contain HH:MM")
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("maturity time must contain a valid HH:MM clock")
    return hour, minute, day_offset


def _resolve_maturity_datetime(*, now: datetime, value: str) -> datetime:
    hour, minute, day_offset = _parse_maturity_clock(value)
    if day_offset is None:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    target_date = now.date() + timedelta(days=day_offset)
    return datetime.combine(target_date, now.time()).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )


def compute_next_run(
    *,
    now: datetime,
    maturity_time: str | None,
    default_interval: timedelta = timedelta(hours=5),
    advance: timedelta = timedelta(minutes=10),
    immediate_delay: timedelta = timedelta(minutes=1),
) -> NextRunPlan:
    default_next = now + default_interval
    if not maturity_time:
        return NextRunPlan(next_run=default_next, reason="default_5h")

    try:
        maturity_at = _resolve_maturity_datetime(now=now, value=maturity_time)
    except (TypeError, ValueError):
        return NextRunPlan(next_run=default_next, reason="default_5h")

    early_start = maturity_at - advance
    if early_start <= now:
        return NextRunPlan(next_run=now + immediate_delay, reason="maturity_time_immediate")
    if early_start < default_next:
        return NextRunPlan(next_run=early_start, reason="maturity_time")
    return NextRunPlan(next_run=default_next, reason="default_5h")


def compute_manual_next_run(
    *,
    now: datetime,
    start_time: str,
) -> NextRunPlan:
    hour, minute = _parse_clock_time(start_time)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return NextRunPlan(next_run=next_run, reason="manual_time")
