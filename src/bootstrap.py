"""Ensure src/ is on sys.path for scripts and analysis modules."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

__all__ = ["SRC", "REPO_ROOT"]
