"""Makes the medcare/ folder importable when running `pytest` from it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
