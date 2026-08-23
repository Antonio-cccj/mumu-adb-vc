from __future__ import annotations

from datetime import datetime, timedelta, timezone


BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def beijing_now(utc_now: datetime | None = None) -> datetime:
    """Return Beijing wall-clock time as a naive datetime.

    The scheduler stores and compares naive datetimes today, so this helper
    deliberately strips tzinfo after converting from UTC+8.
    """
    source = utc_now if utc_now is not None else datetime.now(timezone.utc)
    if source.tzinfo is None:
        source = source.replace(tzinfo=timezone.utc)
    return source.astimezone(BEIJING_TZ).replace(tzinfo=None)
