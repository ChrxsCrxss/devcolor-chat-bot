"""Project paths."""

from pathlib import Path

# Repository root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "devcolorfaq.txt"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
