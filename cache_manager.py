from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# card_snapshots 保存用户可点击查看的作物信息卡截图，需永久保留，不参与自动清理。
PRESERVED_DIR_NAMES = {"template_sources", "failure_snapshots", "card_snapshots"}


@dataclass(frozen=True)
class CacheCleanupResult:
    removed_files: int
    removed_bytes: int


def clear_screenshot_cache(debug_dir: str | Path) -> CacheCleanupResult:
    root = Path(debug_dir).resolve()
    if not root.exists():
        return CacheCleanupResult(removed_files=0, removed_bytes=0)

    removed_files = 0
    removed_bytes = 0
    for path in list(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part in PRESERVED_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        try:
            file_size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        removed_files += 1
        removed_bytes += file_size

    _prune_empty_dirs(root)
    return CacheCleanupResult(removed_files=removed_files, removed_bytes=removed_bytes)


def _prune_empty_dirs(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        if path.name in PRESERVED_DIR_NAMES:
            continue
        try:
            path.rmdir()
        except OSError:
            pass
