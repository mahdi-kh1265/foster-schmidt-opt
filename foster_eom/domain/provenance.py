"""Provenance and reproducibility (spec §28).

Run manifests record everything needed to reproduce or audit a run:
software versions, RNG seeds, model hashes, timestamps, and termination
status.  Content hashing ensures immutable model references.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys
import uuid
from typing import Any

from pydantic import BaseModel, Field

import foster_eom


class RunManifest(BaseModel):
    """Run manifest for reproducibility (spec §28.1).

    Attributes
    ----------
    run_id : str
        Unique run identifier (UUID).
    project_schema_version : str
        Schema version of the project file.
    project_spec_hash : str
        Content hash of the project specification.
    software_version : str
        foster_eom package version.
    software_git_commit : str | None
        Git commit hash if available.
    python_version : str
        Python interpreter version.
    platform_info : str
        OS and architecture.
    package_versions : dict[str, str]
        Versions of key dependencies.
    random_seed : int
        RNG seed used.
    eom_model_hash : str | None
        Content hash of the EOM model.
    component_library_hashes : dict[str, str]
        Content hashes of component library snapshots.
    solver_settings : dict[str, Any]
        Solver configuration snapshot.
    worker_count : int | None
        Number of parallel workers used.
    start_time : str
        ISO-format start timestamp.
    end_time : str | None
        ISO-format end timestamp (None if in progress).
    termination_reason : str | None
        Why the run ended.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_schema_version: str = ""
    project_spec_hash: str = ""
    software_version: str = Field(default_factory=lambda: foster_eom.__version__)
    software_git_commit: str | None = None
    python_version: str = Field(default_factory=lambda: sys.version)
    platform_info: str = Field(default_factory=lambda: platform.platform())
    package_versions: dict[str, str] = Field(default_factory=dict)
    random_seed: int = 0
    eom_model_hash: str | None = None
    component_library_hashes: dict[str, str] = Field(default_factory=dict)
    solver_settings: dict[str, Any] = Field(default_factory=dict)
    worker_count: int | None = None
    start_time: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    end_time: str | None = None
    termination_reason: str | None = None


def collect_dependency_versions() -> dict[str, str]:
    """Collect versions of key dependencies.

    Returns
    -------
    dict[str, str]
        Package name → version string for installed dependencies.
    """
    versions: dict[str, str] = {}
    for pkg_name in ["numpy", "scipy", "pydantic", "yaml", "skrf"]:
        try:
            if pkg_name == "yaml":
                import yaml

                versions["PyYAML"] = getattr(yaml, "__version__", "unknown")
            elif pkg_name == "skrf":
                import skrf  # type: ignore[import-untyped]

                versions["scikit-rf"] = getattr(skrf, "__version__", "unknown")
            else:
                mod = __import__(pkg_name)
                versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def hash_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file.

    Parameters
    ----------
    path : str
        Filesystem path to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def hash_string(data: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string.

    Parameters
    ----------
    data : str
        String to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_dict(d: dict[str, Any]) -> str:
    """Compute a stable SHA-256 hash of a JSON-serializable dict.

    Keys are sorted to ensure path/insertion-order independence.

    Parameters
    ----------
    d : dict
        Dictionary to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hash_string(canonical)
