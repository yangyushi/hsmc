#!/usr/bin/env python3
import configparser
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configure.ini"
CONFIG_TEMPLATE_PATH = ROOT_DIR / "configure.ini.example"
RESULT_DIR = ROOT_DIR / "result"
FIGURE_DIR = ROOT_DIR / "figure"
TCC_DIR = ROOT_DIR / "tcc"
UUID_PATTERN = re.compile(
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)

WORKFLOW_CODE_FILES = (
    "workflow.py",
    "common/__init__.py",
    "common/slit_setup.py",
    "common/workflow_support.py",
    "scripts/__init__.py",
    "scripts/analysis.py",
    "scripts/bulk.py",
    "scripts/get_relax_time.py",
    "scripts/plot.py",
    "scripts/simulate.py",
    "scripts/validate_workflow.py",
)


def load_config(config_path=CONFIG_PATH):
    conf = configparser.ConfigParser()
    conf.read(config_path)
    return conf


def ensure_output_directories():
    for path in (RESULT_DIR, FIGURE_DIR, TCC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def is_auto(value: str) -> bool:
    return str(value).strip().lower() == "auto"


def parse_numeric_frequency(value: str) -> int:
    return int(float(str(value).strip()))


def snap_dump_frequency(tau_sweeps: float) -> int:
    if tau_sweeps <= 1:
        return 1
    exponent = math.ceil(math.log10(tau_sweeps))
    return max(1, int(10 ** exponent))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uuid_from_payload(payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"simulate_slit:{digest}"))


def compute_workflow_metadata(config_path=CONFIG_PATH) -> dict:
    if not Path(config_path).is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {config_path}. "
            f"Copy {CONFIG_TEMPLATE_PATH.name} to {Path(config_path).name} and fill in the values."
        )

    file_hashes = {}
    payload_parts = []
    for relative_path in WORKFLOW_CODE_FILES:
        path = ROOT_DIR / relative_path
        file_hashes[relative_path] = _hash_file(path)
        payload_parts.append(f"{relative_path}:{file_hashes[relative_path]}")

    config_hash = _hash_file(Path(config_path))
    payload_parts.append(f"{Path(config_path).name}:{config_hash}")
    workflow_uuid = _uuid_from_payload("\n".join(payload_parts))

    return {
        "workflow_uuid": workflow_uuid,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_dir": str(ROOT_DIR),
        "config_path": str(Path(config_path)),
        "config_sha256": config_hash,
        "hashed_files": file_hashes,
    }


def workflow_uuid(config_path=CONFIG_PATH) -> str:
    return compute_workflow_metadata(config_path)["workflow_uuid"]


def tagged_filename(stem: str, suffix: str, current_uuid=None) -> str:
    current_uuid = current_uuid or workflow_uuid()
    return f"{stem}.{current_uuid}{suffix}"


def root_log_path(current_uuid=None) -> Path:
    return ROOT_DIR / tagged_filename("workflow", ".log", current_uuid)


def root_metadata_path(current_uuid=None) -> Path:
    return ROOT_DIR / tagged_filename("workflow", ".json", current_uuid)


def result_path(stem: str, suffix: str, current_uuid=None) -> Path:
    return RESULT_DIR / tagged_filename(stem, suffix, current_uuid)


def figure_path(stem: str, suffix: str, current_uuid=None) -> Path:
    return FIGURE_DIR / tagged_filename(stem, suffix, current_uuid)


def slit_sample_path(conf=None, current_uuid=None) -> Path:
    conf = conf or load_config()
    filename = Path(conf["Run"]["filename"]).name
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".xyz"
    return result_path(stem, suffix, current_uuid)


def bulk_sample_path(current_uuid=None) -> Path:
    return result_path("sample_bulk", ".xyz", current_uuid)


def isf_metadata_path(current_uuid=None) -> Path:
    return result_path("isf", ".json", current_uuid)


def isf_arrays_path(current_uuid=None) -> Path:
    return result_path("isf", ".npz", current_uuid)


def box_path(current_uuid=None) -> Path:
    return result_path("box", ".json", current_uuid)


def bulk_box_path(current_uuid=None) -> Path:
    return result_path("box_bulk", ".json", current_uuid)


def density_profile_path(current_uuid=None) -> Path:
    return result_path("density_profile", ".npz", current_uuid)


def tcc_bulk_path(current_uuid=None) -> Path:
    return result_path("tcc_bulk", ".npz", current_uuid)


def tcc_spatial_dist_path(current_uuid=None) -> Path:
    return result_path("tcc_spatial_dist", ".npz", current_uuid)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_workflow_metadata(current_uuid=None) -> dict:
    metadata = compute_workflow_metadata()
    if current_uuid is not None and metadata["workflow_uuid"] != current_uuid:
        raise RuntimeError("Workflow UUID mismatch while preparing metadata.")
    write_json(root_metadata_path(metadata["workflow_uuid"]), metadata)
    return metadata


def write_workflow_log(message: str, stage: str, current_uuid=None, **fields):
    if current_uuid is None:
        if not CONFIG_PATH.is_file():
            return
        current_uuid = workflow_uuid()

    log_file = root_log_path(current_uuid)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "stage": stage,
        "message": message,
        "workflow_uuid": current_uuid,
    }
    payload.update(fields)
    parts = [f"{key}={payload[key]}" for key in sorted(payload)]
    with open(log_file, "a") as f:
        f.write(" ".join(parts) + "\n")


def config_uses_auto(config_path=CONFIG_PATH) -> bool:
    conf = load_config(config_path)
    return any(
        is_auto(conf["Run"][key])
        for key in ("dump_frequency", "dump_frequency_bulk")
    )


def load_isf_metadata(current_uuid=None) -> dict:
    metadata = read_json(isf_metadata_path(current_uuid))
    expected_uuid = current_uuid or workflow_uuid()
    if metadata.get("workflow_uuid") != expected_uuid:
        raise RuntimeError("Cached ISF metadata does not match the current workflow UUID.")
    return metadata


def isf_ready(current_uuid=None) -> bool:
    current_uuid = current_uuid or workflow_uuid()
    metadata_file = isf_metadata_path(current_uuid)
    arrays_file = isf_arrays_path(current_uuid)
    if not metadata_file.is_file() or not arrays_file.is_file():
        return False
    metadata = read_json(metadata_file)
    return metadata.get("workflow_uuid") == current_uuid


def resolve_dump_frequency(config_value, current_uuid=None) -> int:
    if is_auto(config_value):
        if not isf_ready(current_uuid):
            raise FileNotFoundError(
                "Missing ISF result files for auto dump frequency. "
                "Run python3 workflow.py so the ISF stage can populate the current UUID-tagged artifacts."
            )
        return int(load_isf_metadata(current_uuid)["recommended_dump_frequency"])
    return parse_numeric_frequency(config_value)


def run_stage(module_name: str, stage: str, current_uuid=None):
    if current_uuid is None and CONFIG_PATH.is_file():
        current_uuid = workflow_uuid()
    write_workflow_log("stage_start", stage, current_uuid=current_uuid)
    env = os.environ.copy()
    env["WORKFLOW_UUID"] = current_uuid
    pythonpath_entries = [str(ROOT_DIR)]
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
        write_workflow_log(
            "failure",
            stage,
            current_uuid=current_uuid,
            exit_code=exc.returncode,
            module=module_name,
        )
        raise

    write_workflow_log("stage_end", stage, current_uuid=current_uuid)


def cluster_population_mapping(summary) -> dict:
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
    paths = []
    for directory in (RESULT_DIR, FIGURE_DIR):
        if directory.is_dir():
            paths.extend(path for path in directory.iterdir() if path.is_file())
    paths.extend(ROOT_DIR.glob("workflow.*.json"))
    paths.extend(ROOT_DIR.glob("workflow.*.log"))
    return sorted(set(paths))


def group_generated_paths_by_uuid() -> tuple[dict[str, list[Path]], list[Path]]:
    grouped = {}
    untagged = []
    for path in generated_paths():
        artifact_uuid = extract_uuid_from_name(path)
        if artifact_uuid is None:
            untagged.append(path)
            continue
        grouped.setdefault(artifact_uuid, []).append(path)
    return grouped, untagged


def remove_generated_outputs():
    for directory in (RESULT_DIR, FIGURE_DIR, TCC_DIR):
        if directory.exists():
            shutil.rmtree(directory)
    for path in ROOT_DIR.glob("workflow.*.json"):
        path.unlink()
    for path in ROOT_DIR.glob("workflow.*.log"):
        path.unlink()
