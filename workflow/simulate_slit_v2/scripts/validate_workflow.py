#!/usr/bin/env python3
import configparser
import importlib
import shutil
import sys

from common.slit_setup import VALID_PLANES
from common.workflow_support import (
    CONFIG_PATH,
    CONFIG_TEMPLATE_PATH,
    ensure_output_directories,
    is_auto,
    load_config,
    parse_numeric_frequency,
    workflow_uuid,
    write_workflow_log,
)


REQUIRED_MODULES = ("numpy", "matplotlib", "scipy", "hsmc", "tcc")
VALID_BOUNDARY_KINDS = ("hardwall", *VALID_PLANES)


def fail(message: str, current_uuid: str | None = None, **fields: object) -> int:
    write_workflow_log("failure", "validate_workflow", current_uuid=current_uuid, error=message, **fields)
    print(message, file=sys.stderr)
    for key, value in fields.items():
        print(f"{key}: {value}", file=sys.stderr)
    return 1


def _require_value(
    conf: configparser.ConfigParser,
    section: str,
    option: str,
) -> str:
    """Return a non-empty configuration value."""

    try:
        value = conf[section][option].strip()
    except KeyError as exc:
        raise ValueError(
            f"Missing configuration entry: [{section}] {option}"
        ) from exc
    if not value:
        raise ValueError(f"Empty configuration entry: [{section}] {option}")
    return value


def _require_int(
    conf: configparser.ConfigParser,
    section: str,
    option: str,
    minimum: int,
) -> int:
    """Return an integer configuration value with a lower bound."""

    value = _require_value(conf, section, option)
    try:
        parsed = int(float(value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value for [{section}] {option}: {value!r}"
        ) from exc
    if parsed < minimum:
        raise ValueError(
            f"Configuration entry [{section}] {option} must be >= {minimum}."
        )
    return parsed


def _require_float(
    conf: configparser.ConfigParser,
    section: str,
    option: str,
    minimum: float,
) -> float:
    """Return a float configuration value with a lower bound."""

    value = _require_value(conf, section, option)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float value for [{section}] {option}: {value!r}"
        ) from exc
    if parsed < minimum:
        raise ValueError(
            f"Configuration entry [{section}] {option} must be >= {minimum}."
        )
    return parsed


def _validate_dump_frequency(
    conf: configparser.ConfigParser,
    option: str,
) -> None:
    """Validate a configured dump frequency or the `auto` keyword."""

    value = _require_value(conf, "Run", option)
    if is_auto(value):
        return
    try:
        parsed = parse_numeric_frequency(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid dump frequency for [Run] {option}: {value!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"Configuration entry [Run] {option} must be a positive integer or auto."
        )


def validate_configuration() -> None:
    """Validate that the workflow configuration is complete and parseable."""

    conf = load_config()

    _require_int(conf, "System", "n", 1)
    _require_float(conf, "System", "vf_init", 0.0)
    _require_float(conf, "System", "vf_final", 0.0)
    _require_float(conf, "System", "r_skin", 0.0)

    _require_int(conf, "Run", "equilibrium", 0)
    _require_int(conf, "Run", "total", 0)
    _require_int(conf, "Run", "total_bulk", 0)
    _validate_dump_frequency(conf, "dump_frequency")
    _validate_dump_frequency(conf, "dump_frequency_bulk")
    _require_value(conf, "Run", "filename")

    _require_float(conf, "Boundary", "z", 0.0)
    kind = _require_value(conf, "Boundary", "kind")
    if kind not in VALID_BOUNDARY_KINDS:
        raise ValueError(
            "[Boundary] kind must be one of "
            + ", ".join(VALID_BOUNDARY_KINDS)
            + "."
        )
    _require_float(conf, "Boundary", "vf_crystal", 0.0)

    _require_int(conf, "Analyse", "nbins", 1)

    _require_int(conf, "ISF", "n", 1)
    _require_int(conf, "ISF", "length", 1)
    _require_int(conf, "ISF", "jump", 1)
    try:
        conf.getboolean("ISF", "show_isf")
        conf.getboolean("ISF", "plot_isf")
    except (ValueError, configparser.Error) as exc:
        raise ValueError(
            "Invalid boolean value in [ISF] show_isf or plot_isf."
        ) from exc


def validate_environment(write_artifacts: bool = True, emit_output: bool = True) -> tuple[int, str | None]:
    ensure_output_directories()
    if not CONFIG_PATH.is_file():
        return fail(
            f"Missing configuration file: {CONFIG_PATH.name}. "
            f"Copy {CONFIG_TEMPLATE_PATH.name} to {CONFIG_PATH.name} and fill in the values."
        ), None

    current_uuid = workflow_uuid()
    try:
        validate_configuration()
    except ValueError as exc:
        return fail(
            "Invalid workflow configuration.",
            current_uuid=current_uuid,
            error_detail=str(exc),
        ), current_uuid

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
        ), current_uuid

    import tcc  # noqa: E402

    if shutil.which("tcc") is None:
        return fail(
            "Missing `tcc` executable in PATH.",
            current_uuid=current_uuid,
        ), current_uuid

    try:
        tcc.Parser("tcc")
        tcc.OTF()
    except Exception as exc:
        return fail(
            "The Python `tcc` package is installed but not usable in this workflow.",
            current_uuid=current_uuid,
            error_detail=str(exc),
        ), current_uuid

    if write_artifacts:
        write_workflow_log("stage_end", "validate_workflow", current_uuid=current_uuid)
    if emit_output:
        print(f"workflow_uuid={current_uuid}")
    return 0, current_uuid


def main() -> int:
    status, _ = validate_environment(write_artifacts=True, emit_output=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
