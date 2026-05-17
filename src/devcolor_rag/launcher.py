"""Single entry: ensure venv/deps, then launch devColorBot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _in_project_venv() -> bool:
    from devcolor_rag.paths import VENV_PYTHON

    if not VENV_PYTHON.exists():
        return False
    return Path(sys.executable).resolve() == VENV_PYTHON.resolve()


def _ensure_venv() -> None:
    """Create .venv, install package, re-exec with venv Python."""
    from devcolor_rag.paths import PROJECT_ROOT, VENV_PYTHON

    if _in_project_venv():
        return

    venv_dir = PROJECT_ROOT / ".venv"
    if not venv_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
        )

    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)],
        check=True,
    )

    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), "-m", "devcolor_rag.launcher", *sys.argv[1:]],
    )


def main() -> None:
    _ensure_venv()
    from devcolor_rag.cli import app_main

    app_main()


if __name__ == "__main__":
    main()
