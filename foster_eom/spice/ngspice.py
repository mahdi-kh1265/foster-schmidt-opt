"""P11 ngspice subprocess adapter.

Uses ngspice batch mode with ``wrdata`` ASCII output.  No binary rawfile
parser.

Source convention: the netlist emits ``Vsrc AC 1 0``.  SPICE returns
complex small-signal phasors; Python multiplies by ``vth_phasor`` after parse.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foster_eom.spice.netlist import SpiceNetlist

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NgspiceNotFoundError(RuntimeError):
    """Raised when ngspice is not found on PATH."""


class NgspiceRunError(RuntimeError):
    """Raised when ngspice exits non-zero or produces no data."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NgspiceResult:
    """Raw AC analysis result from ngspice.

    All complex values are unit-source small-signal phasors (from ``Vsrc AC 1 0``).
    Multiply by ``source_spec.vth_phasor`` in Python before comparison.

    Parameters
    ----------
    frequencies_hz : np.ndarray
        Frequencies matching the ``.AC`` command.
    node_voltages : dict[str, np.ndarray]
        SPICE node name -> complex voltage array (unit source).
    sense_currents : dict[str, np.ndarray]
        Sense source name -> complex current array (unit source).
        Positive current = into DUT for ``Vsense``; into branch for branch sensors.
    solver_version : str
    """

    frequencies_hz: np.ndarray
    node_voltages: dict[str, np.ndarray]
    sense_currents: dict[str, np.ndarray]
    solver_version: str


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_ngspice() -> str | None:
    """Return ngspice version string, or None if not on PATH."""
    exe = shutil.which("ngspice")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            ["ngspice", "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        output = result.stdout + result.stderr
        # Extract version: look for "ngspice-XX" or "ngspice XX"
        m = re.search(r"ngspice[-\s]+(\d[\d.]*)", output, re.IGNORECASE)
        if m:
            return m.group(0)
        return output.strip().split("\n")[0] if output.strip() else "ngspice (version unknown)"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Netlist .control block builder
# ---------------------------------------------------------------------------


def _build_control_netlist(
    netlist: SpiceNetlist,
    sense_names: list[str],
    node_names: list[str],
) -> str:
    """Inject a .control block into the netlist for wrdata ASCII output.

    Parameters
    ----------
    netlist : SpiceNetlist
    sense_names : list[str]
        Sense source SPICE names to probe (e.g. ``["Vsense", "Vsns_b1_L1"]``).
    node_names : list[str]
        SPICE node names to save (non-ground nodes).

    Returns
    -------
    str
        Full netlist text with injected .control block.
    """
    # Remove existing .end and any trailing .control/.endc blocks
    base = netlist.netlist_text
    # Strip trailing .end line
    lines = base.rstrip().split("\n")
    # Remove last '.end' if present
    if lines and lines[-1].strip().lower() == ".end":
        lines = lines[:-1]
    # Remove inline .AC line if it will be superseded by .control
    # (for irregular grids the ac_command is already a .control block;
    # for regular grids we move .AC inside .control for unified flow)
    filtered: list[str] = []
    in_ctrl = False
    for line in lines:
        s = line.strip().lower()
        if s.startswith(".control"):
            in_ctrl = True
            continue
        if s.startswith(".endc"):
            in_ctrl = False
            continue
        if in_ctrl:
            continue
        filtered.append(line)

    base_lines = filtered

    # Build .control block
    ctrl: list[str] = [".control"]

    # AC analysis: re-emit from netlist's ac_command (may be .AC or irregular list)
    ac_cmd = netlist.ac_command
    if ac_cmd.startswith(".control"):
        # Extract inner lines
        inner = []
        for ln in ac_cmd.split("\n"):
            s = ln.strip()
            if s.lower() in (".control", ".endc"):
                continue
            inner.append("  " + s)
        ctrl.extend(inner)
    else:
        # Strip leading dot for .control context
        ctrl.append("  " + ac_cmd.lstrip("."))

    # Probe nodes
    for nname in node_names:
        ctrl.append(f"  wrdata /dev/stdout v({nname})")
    # Probe sense currents
    for sname in sense_names:
        ctrl.append(f"  wrdata /dev/stdout i({sname})")

    ctrl.append(".endc")
    ctrl.append("")
    ctrl.append(".end")

    return "\n".join(base_lines) + "\n" + "\n".join(ctrl) + "\n"


# ---------------------------------------------------------------------------
# ASCII wrdata parser
# ---------------------------------------------------------------------------


def _parse_wrdata_stdout(output: str, sense_names: list[str], node_names: list[str]) -> dict[str, Any]:
    """Parse interleaved wrdata blocks from stdout.

    ngspice wrdata to /dev/stdout emits blocks of:
        <name>
        freq  real  imag
        ...

    Returns
    -------
    dict with keys: "freq", "nodes" (dict), "currents" (dict).
    """
    result: dict[str, Any] = {"freq": None, "nodes": {}, "currents": {}}

    # Split output into labelled blocks
    # Each block starts with a line that is just the variable name
    blocks: list[tuple[str, list[tuple[float, float, float]]]] = []
    current_label: str | None = None
    current_rows: list[tuple[float, float, float]] = []

    for raw_line in output.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("*") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1 and not _is_numeric(parts[0]):
            # New block label
            if current_label is not None:
                blocks.append((current_label, current_rows))
            current_label = parts[0].lower()
            current_rows = []
        elif len(parts) >= 3 and _is_numeric(parts[0]):
            try:
                f = float(parts[0])
                re_ = float(parts[1])
                im = float(parts[2])
                current_rows.append((f, re_, im))
            except ValueError:
                pass

    if current_label is not None:
        blocks.append((current_label, current_rows))

    # Assign blocks to nodes/currents
    freq_arr: np.ndarray | None = None
    for label, rows in blocks:
        if not rows:
            continue
        freqs = np.array([r[0] for r in rows])
        vals = np.array([complex(r[1], r[2]) for r in rows])
        if freq_arr is None:
            freq_arr = freqs

        # Match against requested names
        matched = False
        for nname in node_names:
            if label == f"v({nname.lower()})":
                result["nodes"][nname] = vals
                matched = True
                break
        if not matched:
            for sname in sense_names:
                if label == f"i({sname.lower()})":
                    result["currents"][sname] = vals
                    matched = True
                    break

    if freq_arr is not None:
        result["freq"] = freq_arr

    return result


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace("e", "E").replace("E+", "E"))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_ngspice(
    netlist: SpiceNetlist,
    work_dir: Path | None = None,
    timeout_s: float = 30.0,
) -> NgspiceResult:
    """Write netlist to temp dir, run ngspice -b, parse ASCII output.

    Parameters
    ----------
    netlist : SpiceNetlist
    work_dir : Path | None
        Temporary directory to use.  Created and cleaned up automatically
        if None.
    timeout_s : float
        Subprocess timeout.

    Returns
    -------
    NgspiceResult

    Raises
    ------
    NgspiceNotFoundError
        If ngspice is not on PATH.
    NgspiceRunError
        If ngspice exits non-zero or no frequencies are parsed.
    """
    version = detect_ngspice()
    if version is None:
        raise NgspiceNotFoundError(
            "ngspice not found on PATH. Install ngspice or skip SPICE validation."
        )

    # Determine what to probe
    sense_names = list(netlist.sense_source_map.values())
    node_names = [v for v in netlist.node_map.values() if v != "0"]

    # Build control netlist
    ctrl_text = _build_control_netlist(netlist, sense_names, node_names)

    ctx_mgr = (
        tempfile.TemporaryDirectory() if work_dir is None
        else _NullContextManager(str(work_dir))
    )

    with ctx_mgr as tmpdir:
        tmppath = Path(tmpdir)
        nl_path = tmppath / "circuit.sp"
        nl_path.write_text(ctrl_text, encoding="utf-8")

        try:
            proc = subprocess.run(
                ["ngspice", "-b", str(nl_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise NgspiceRunError(f"ngspice timed out after {timeout_s}s") from exc
        except FileNotFoundError as exc:
            raise NgspiceNotFoundError("ngspice not found") from exc

        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            raise NgspiceRunError(
                f"ngspice exited {proc.returncode}.\nSTDERR:\n{stderr[:2000]}"
            )

    parsed = _parse_wrdata_stdout(stdout, sense_names, node_names)

    freq_arr = parsed.get("freq")
    if freq_arr is None or len(freq_arr) == 0:
        raise NgspiceRunError(
            f"ngspice produced no frequency data.\nSTDOUT:\n{stdout[:2000]}"
            f"\nSTDERR:\n{stderr[:2000]}"
        )

    return NgspiceResult(
        frequencies_hz=freq_arr,
        node_voltages=parsed["nodes"],
        sense_currents=parsed["currents"],
        solver_version=version,
    )


class _NullContextManager:
    """Context manager that returns a fixed string (for existing directories)."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, *args: object) -> None:
        pass

