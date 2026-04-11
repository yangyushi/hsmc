from __future__ import annotations

import shutil
import sys
from importlib import resources
from pathlib import Path


def copy_simulate_slit(destination: str | Path = ".") -> Path:
    target_root = Path(destination).resolve()
    target = target_root / "simulate_slit"

    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}")

    source = resources.files("hsmc").joinpath("script", "simulate_slit")
    shutil.copytree(source, target)
    return target


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(
            "Usage: python -m hsmc._assets simulate_slit [destination]"
        )

    artifact = argv.pop(0)
    destination = argv.pop(0) if argv else "."

    if artifact != "simulate_slit":
        raise SystemExit(f"Unknown artifact: {artifact}")

    copy_simulate_slit(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
