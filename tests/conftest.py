from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"

lib_path = str(LIB)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
