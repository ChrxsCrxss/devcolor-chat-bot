"""Project paths."""

import sys
from pathlib import Path

# Repository root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "devcolorfaq.txt"


def venv_python() -> Path:
    """Cross-platform path to the project virtualenv interpreter."""
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


VENV_PYTHON = venv_python()
