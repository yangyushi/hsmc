#!/usr/bin/env python3
import importlib
import shutil
import sys

from common.workflow_support import (
    CONFIG_PATH,
    CONFIG_TEMPLATE_PATH,
    ensure_output_directories,
    ensure_workflow_metadata,
    write_workflow_log,
)


REQUIRED_MODULES = ("numpy", "matplotlib", "scipy", "hsmc", "tcc")


def fail(message: str, current_uuid=None, **fields) -> int:
    write_workflow_log("failure", "validate_workflow", current_uuid=current_uuid, error=message, **fields)
    print(message, file=sys.stderr)
    for key, value in fields.items():
        print(f"{key}: {value}", file=sys.stderr)
    return 1


def validate_environment(write_artifacts: bool = True, emit_output: bool = True) -> tuple[int, dict | None]:
    ensure_output_directories()
    if not CONFIG_PATH.is_file():
        return fail(
            f"Missing configuration file: {CONFIG_PATH.name}. "
            f"Copy {CONFIG_TEMPLATE_PATH.name} to {CONFIG_PATH.name} and fill in the values."
        ), None

    metadata = ensure_workflow_metadata() if write_artifacts else None
    if metadata is None:
        from common.workflow_support import compute_workflow_metadata
        metadata = compute_workflow_metadata()
    current_uuid = metadata["workflow_uuid"]
    if write_artifacts:
        write_workflow_log("stage_start", "validate_workflow", current_uuid=current_uuid)

    missing_modules = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing_modules.append((module_name, str(exc)))

    if missing_modules:
        details = "; ".join(f"{name}: {error}" for name, error in missing_modules)
        return fail(
            "Missing or broken Python dependencies for the slit workflow.",
            current_uuid=current_uuid,
            dependencies=details,
        ), metadata

    import tcc  # noqa: E402

    if shutil.which("tcc") is None:
        return fail(
            "Missing `tcc` executable in PATH.",
            current_uuid=current_uuid,
        ), metadata

    try:
        tcc.Parser("tcc")
        tcc.OTF()
    except Exception as exc:
        return fail(
            "The Python `tcc` package is installed but not usable in this workflow.",
            current_uuid=current_uuid,
            error_detail=str(exc),
        ), metadata

    if write_artifacts:
        write_workflow_log("stage_end", "validate_workflow", current_uuid=current_uuid)
    if emit_output:
        print(f"workflow_uuid={current_uuid}")
    return 0, metadata


def main() -> int:
    status, _ = validate_environment(write_artifacts=True, emit_output=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
