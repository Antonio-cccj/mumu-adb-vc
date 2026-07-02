from __future__ import annotations

from client import main
from runtime_paths import ensure_app_working_dir


if __name__ == "__main__":
    ensure_app_working_dir()
    main()
