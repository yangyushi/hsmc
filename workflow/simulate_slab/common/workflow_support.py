#!/usr/bin/env python3
"""Shared filesystem, metadata, and execution helpers for the slit workflow."""

import os
import re
import sys
import json
import math
import logging
import shutil
import hashlib
import subprocess
import configparser
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configure.ini"
CONFIG_TEMPLATE_PATH = ROOT_DIR / "configure.ini.example"
RESULT_DIR = ROOT_DIR / "result"
FIGURE_DIR = ROOT_DIR / "figure"
HASH_SIZE = 8
TCC_DIR = ROOT_DIR / "tcc"
UUID_PATTERN = re.compile(r"(?P<uuid>[0-9a-f]{64})")
WORKFLOW_CODE_DIRS = ("common", "scripts")
WORKFLOW_LOGGER_NAME = "workflow"
DEFAULT_WORKFLOW_UUID = "-"

JsonObject = dict[str, object]
_LOGGER_STATE: tuple[Path | None, bool] | None = None


class SupportsToDict(Protocol):
    """Protocol for summary objects that expose a mapping view."""

    def to_dict(self) -> Mapping[object, object]:
        """Return the summary as a mapping."""


class WorkflowLogFormatter(logging.Formatter):
    """Render compact workflow log lines."""

    def formatTime(self, record, datefmt=None) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    def format(self, record: logging.LogRecord) -> str:
        workflow_uuid = getattr(record, "workflow_uuid", DEFAULT_WORKFLOW_UUID)
        message = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        return (
            f"[{self.formatTime(record)}] "
            f"[{record.levelname}] "
            f"[{workflow_uuid}] "
            f"[{record.process}] "
            f"{message}"
        )


class WorkflowLoggerAdapter(logging.LoggerAdapter):
    """Bind workflow UUID and stage context to log messages."""

    def process(self, msg, kwargs):
        extra = dict(kwargs.get("extra", {}))
        extra.setdefault(
            "workflow_uuid",
            self.extra.get("workflow_uuid", DEFAULT_WORKFLOW_UUID),
        )
        kwargs["extra"] = extra
        stage = self.extra.get("stage")
        message = str(msg)
        if stage and not message.startswith(f"{stage}: "):
            message = f"{stage}: {message}"
        return message, kwargs


def active_workflow_uuid(current_uuid: str | None = None) -> str | None:
    """Resolve the active workflow UUID from arguments, env, or config."""

    if current_uuid is not None:
        return current_uuid
    env_uuid = os.environ.get("WORKFLOW_UUID")
    if env_uuid:
        return env_uuid
    if CONFIG_PATH.is_file():
        return workflow_uuid()
    return None


def _reset_workflow_logging() -> None:
    """Clear workflow logger handlers so setup can be repeated safely."""

    global _LOGGER_STATE

    logger = logging.getLogger(WORKFLOW_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _LOGGER_STATE = None


def configure_workflow_logging(
    current_uuid: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure the shared workflow logger for the active UUID."""

    global _LOGGER_STATE

    resolved_uuid = active_workflow_uuid(current_uuid)
    log_path = (
        root_log_path(resolved_uuid) if resolved_uuid is not None else None
    )
    desired_state = (log_path, console)
    logger = logging.getLogger(WORKFLOW_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers and _LOGGER_STATE == desired_state:
        return logger

    _reset_workflow_logging()
    logger = logging.getLogger(WORKFLOW_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = WorkflowLogFormatter()

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _LOGGER_STATE = desired_state
    return logger


def get_workflow_logger(
    stage: str | None = None,
    current_uuid: str | None = None,
    console: bool = True,
) -> WorkflowLoggerAdapter:
    """Return a stage-aware workflow logger adapter."""

    resolved_uuid = active_workflow_uuid(current_uuid)
    configure_workflow_logging(resolved_uuid, console=console)
    logger_name = WORKFLOW_LOGGER_NAME
    if stage:
        logger_name = f"{WORKFLOW_LOGGER_NAME}.{stage}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    return WorkflowLoggerAdapter(
        logger,
        {
            "workflow_uuid": resolved_uuid or DEFAULT_WORKFLOW_UUID,
            "stage": stage,
        },
    )


def _format_log_fields(fields: Mapping[str, object]) -> str:
    """Render compatibility key/value fields into readable text."""

    return ", ".join(f"{key}={fields[key]}" for key in sorted(fields))


def load_config(config_path: Path = CONFIG_PATH) -> configparser.ConfigParser:
    """Load the workflow configuration file from disk."""

    conf = configparser.ConfigParser()
    conf.read(config_path)
    return conf


def ensure_output_directories() -> None:
    """Create the output directories expected by workflow stages."""

    for path in (RESULT_DIR, FIGURE_DIR, TCC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def is_auto(value: str) -> bool:
    return str(value).strip().lower() == "auto"


def parse_numeric_frequency(value: str) -> int:
    """Parse an integer-like dump frequency from configuration text."""

    return int(float(str(value).strip()))


def snap_dump_frequency(tau_sweeps: float) -> int:
    """Round an ISF-derived sweep count up to the next power of ten."""

    if tau_sweeps <= 1:
        return 1
    exponent = math.ceil(math.log10(tau_sweeps))
    return max(1, int(10 ** exponent))


def _workflow_source_files() -> list[Path]:
    """
    Return the Python source files that define the workflow implementation.
    """

    source_files: list[Path] = []
    for relative_dir in WORKFLOW_CODE_DIRS:
        source_files.extend(sorted((ROOT_DIR / relative_dir).rglob("*.py")))
    return source_files


def _workflow_digest(config_path: Path) -> str:
    """
    Digest workflow source files
    """
    hasher = hashlib.blake2b(digest_size=HASH_SIZE)
    for path in _workflow_source_files():
        relative_path = path.relative_to(ROOT_DIR).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")

    hasher.update(Path(config_path).name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(Path(config_path).read_bytes())
    return hasher.hexdigest()


def workflow_uuid(config_path: Path = CONFIG_PATH) -> str:
    """
    Return the deterministic workflow digest from code under common/scripts
        and the config file.
    """

    if not Path(config_path).is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {config_path}. "
            f"Copy {CONFIG_TEMPLATE_PATH.name} to {Path(config_path).name}"
            "and fill in the values."
        )

    return _workflow_digest(Path(config_path))


def tagged_filename(
    stem: str, suffix: str, current_uuid: str | None = None
) -> str:
    """Build a UUID-tagged artifact filename for the active workflow state."""

    current_uuid = current_uuid or workflow_uuid()
    return f"{stem}.{current_uuid}{suffix}"


def root_log_path(current_uuid: str | None = None) -> Path:
    return ROOT_DIR / tagged_filename("workflow", ".log", current_uuid)


def result_path(
    stem: str, suffix: str, current_uuid: str | None = None
) -> Path:
    return RESULT_DIR / tagged_filename(stem, suffix, current_uuid)


def figure_path(
    stem: str, suffix: str, current_uuid: str | None = None
) -> Path:
    return FIGURE_DIR / tagged_filename(stem, suffix, current_uuid)


def slit_sample_path(
    conf: configparser.ConfigParser | None = None,
    current_uuid: str | None = None,
) -> Path:
    conf = conf or load_config()
    filename = Path(conf["Run"]["filename"]).name
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".xyz"
    return result_path(stem, suffix, current_uuid)


def bulk_sample_path(current_uuid: str | None = None) -> Path:
    return result_path("sample_bulk", ".xyz", current_uuid)


def isf_metadata_path(current_uuid: str | None = None) -> Path:
    return result_path("isf", ".json", current_uuid)


def isf_arrays_path(current_uuid: str | None = None) -> Path:
    return result_path("isf", ".npz", current_uuid)


def box_path(current_uuid: str | None = None) -> Path:
    return result_path("box", ".json", current_uuid)


def bulk_box_path(current_uuid: str | None = None) -> Path:
    return result_path("box_bulk", ".json", current_uuid)


def density_profile_path(current_uuid: str | None = None) -> Path:
    return result_path("density_profile", ".npz", current_uuid)


def tcc_bulk_path(current_uuid: str | None = None) -> Path:
    return result_path("tcc_bulk", ".npz", current_uuid)


def tcc_spatial_dist_path(current_uuid: str | None = None) -> Path:
    return result_path("tcc_spatial_dist", ".npz", current_uuid)


def inplane_profile_path(current_uuid: str | None = None) -> Path:
    return result_path("inplane_profile", ".npz", current_uuid)


def crystal_slab_path(current_uuid: str | None = None) -> Path:
    return result_path("crystal_slab", ".xyz", current_uuid)


def preview_html_path(current_uuid: str | None = None) -> Path:
    return figure_path("preview", ".html", current_uuid)


def read_json(path: Path) -> JsonObject:
    """Read a JSON object from disk."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def write_workflow_log(
    message: str,
    stage: str,
    current_uuid: str | None = None,
    **fields: object,
) -> None:
    """Compatibility wrapper for callers not yet using loggers directly."""

    resolved_uuid = active_workflow_uuid(current_uuid)
    if resolved_uuid is None:
        return

    logger = get_workflow_logger(stage, resolved_uuid)
    event_text = message.replace("_", " ")
    details = _format_log_fields(fields)
    if details:
        logger.info("%s (%s)", event_text, details)
        return
    logger.info("%s", event_text)


def config_uses_auto(config_path: Path = CONFIG_PATH) -> bool:
    """
    Report whether either dump frequency uses automatic ISF-derived tuning.
    """

    conf = load_config(config_path)
    return any(
        is_auto(conf["Run"][key])
        for key in ("dump_frequency", "dump_frequency_bulk")
    )


def slab_info_path(current_uuid: str | None = None) -> Path:
    return result_path("slab_info", ".json", current_uuid)


def load_isf_metadata(current_uuid: str | None = None) -> JsonObject:
    """Load ISF metadata and verify that it matches the active workflow UUID."""

    metadata = read_json(isf_metadata_path(current_uuid))
    expected_uuid = current_uuid or workflow_uuid()
    if metadata.get("workflow_uuid") != expected_uuid:
        raise RuntimeError(
            "Cached ISF metadata does not match the current workflow UUID."
        )
    return metadata


def _metadata_int_field(metadata: Mapping[str, object], key: str) -> int:
    """Return an integer-valued metadata field."""

    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Metadata field {key!r} must be numeric.")
    return int(value)


def isf_ready(current_uuid: str | None = None) -> bool:
    """Check whether UUID-matching ISF artifacts are available for reuse."""

    current_uuid = current_uuid or workflow_uuid()
    metadata_file = isf_metadata_path(current_uuid)
    arrays_file = isf_arrays_path(current_uuid)
    if not metadata_file.is_file() or not arrays_file.is_file():
        return False
    metadata = read_json(metadata_file)
    return metadata.get("workflow_uuid") == current_uuid


def resolve_dump_frequency(
    config_value: str,
    current_uuid: str | None = None,
) -> int:
    """
    Resolve either a literal dump frequency or an ISF-derived automatic value.
    """

    if is_auto(config_value):
        if not isf_ready(current_uuid):
            raise FileNotFoundError(
                "Missing ISF result files for auto dump frequency. "
                "Run python3 workflow.py so the ISF stage can populate"
                "the current UUID-tagged artifacts."
            )
        metadata = load_isf_metadata(current_uuid)
        return _metadata_int_field(metadata, "recommended_dump_frequency")
    return parse_numeric_frequency(config_value)


def run_stage(
    module_name: str,
    stage: str,
    current_uuid: str | None = None,
) -> None:
    """
    Run one workflow stage module with the current UUID in its environment.
    """

    current_uuid = active_workflow_uuid(current_uuid)
    logger = get_workflow_logger(stage, current_uuid)
    logger.info("Starting stage")
    env = os.environ.copy()
    env["WORKFLOW_UUID"] = current_uuid
    pythonpath_entries = [str(ROOT_DIR)]
    # The project may keep shared vendored libraries under a lib/ directory next
    # to the run tree. Including it on PYTHONPATH lets workflow stages import
    # lib.crystal_slab even when a run directory is copied out of the project.
    project_lib = ROOT_DIR / "lib"
    if project_lib.is_dir():
        pythonpath_entries.append(str(project_lib))
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env = {key: str(value) for key, value in env.items() if value is not None}

    try:
        subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=ROOT_DIR,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Stage failed with exit code %s while running %s",
            exc.returncode,
            module_name,
        )
        raise

    logger.info("Completed stage")


def cluster_population_mapping(
    summary: SupportsToDict | Mapping[object, object] | np.ndarray,
) -> dict[str, float]:
    """Normalize supported TCC summary objects into a JSON-friendly mapping."""

    if hasattr(summary, "to_dict"):
        summary = summary.to_dict()
    if hasattr(summary, "items"):
        return {str(key): float(value) for key, value in summary.items()}
    if isinstance(summary, np.ndarray) and summary.dtype.names:
        return {name: float(summary[name]) for name in summary.dtype.names}
    raise TypeError("Unsupported TCC summary type for serialization.")


def extract_uuid_from_name(path: Path) -> str | None:
    match = UUID_PATTERN.search(path.name)
    return match.group("uuid") if match else None


def generated_paths() -> list[Path]:
    """Collect generated artifact files that participate in workflow cleanup."""

    paths = []
    for directory in (RESULT_DIR, FIGURE_DIR):
        if directory.is_dir():
            paths.extend(path for path in directory.iterdir() if path.is_file())
    paths.extend(ROOT_DIR.glob("workflow.*.log"))
    return sorted(set(paths))


def group_generated_paths_by_uuid() -> tuple[dict[str, list[Path]], list[Path]]:
    """
    Group generated artifacts by embedded UUID and return untagged leftovers.
    """

    grouped = {}
    untagged = []
    for path in generated_paths():
        artifact_uuid = extract_uuid_from_name(path)
        if artifact_uuid is None:
            untagged.append(path)
            continue
        grouped.setdefault(artifact_uuid, []).append(path)
    return grouped, untagged


def remove_generated_outputs() -> None:
    """Delete generated workflow outputs and root-level workflow logs."""

    for directory in (RESULT_DIR, FIGURE_DIR, TCC_DIR):
        if directory.exists():
            shutil.rmtree(directory)
    for path in ROOT_DIR.glob("workflow.*.log"):
        path.unlink()
