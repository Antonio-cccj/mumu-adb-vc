from datetime import datetime, timezone

from run_scheduler import compute_manual_next_run
from time_utils import beijing_now


def test_beijing_now_uses_utc_plus_eight_wall_clock():
    utc_now = datetime(2026, 6, 13, 16, 54, 0, tzinfo=timezone.utc)

    assert beijing_now(utc_now) == datetime(2026, 6, 14, 0, 54, 0)


def test_manual_schedule_is_computed_against_beijing_time():
    utc_now = datetime(2026, 6, 13, 16, 54, 0, tzinfo=timezone.utc)
    now = beijing_now(utc_now)

    result = compute_manual_next_run(now=now, start_time="01:00")

    assert result.next_run == datetime(2026, 6, 14, 1, 0, 0)
