"""Single entry: ensure venv/deps, then launch devColorBot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _package_importable(python: Path) -> bool:
    try:
        subprocess.run(
            [str(python), "-c", "import devcolor_rag"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _in_project_venv() -> bool:
    from devcolor_rag.paths import VENV_PYTHON

    if not VENV_PYTHON.exists():
        return False
    return Path(sys.executable).resolve() == VENV_PYTHON.resolve()


def _ensure_venv() -> None:
    """Create .venv, install package, re-exec with venv Python."""
    from devcolor_rag.paths import PROJECT_ROOT, VENV_PYTHON

    if _in_project_venv() and _package_importable(VENV_PYTHON):
        return

    console = None
    if VENV_PYTHON.exists():
        try:
            from rich.console import Console
            from devcolor_rag.theme import DEVCOLOR_THEME

            console = Console(theme=DEVCOLOR_THEME, force_terminal=True)
        except ImportError:
            pass

    def _bootstrap() -> None:
        venv_dir = PROJECT_ROOT / ".venv"
        if not venv_dir.exists():
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
            )
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
            check=True,
        )

    if console is not None:
        from devcolor_rag.spinner import run_with_spinner

        run_with_spinner(console, "Installing Python dependencies…", _bootstrap)
    else:
        print("▸ Installing Python dependencies…", file=sys.stderr, flush=True)
        _bootstrap()

    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), "-u", "-m", "devcolor_rag.launcher", *sys.argv[1:]],
    )


def main() -> None:
    try:
        _ensure_venv()
        from devcolor_rag.cli import app_main

        app_main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n✗ devColorBot failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
