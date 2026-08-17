"""P11 ngspice subprocess adapter.

Uses ngspice batch mode with per-variable `wrdata` ASCII output files.
No binary rawfile parser; no /dev/stdout (not available on Windows).

Source convention: the netlist emits `Vsrc AC 1 0`.  SPICE returns
complex small-signal phasors; Python multiplies by `vth_phasor` after parse.

Windows-compatibility notes
---------------------------
* `/dev/stdout` is not available; each variable is written to a separate
  temp file and read back.
* ngspice `wrdata` uses the CWD at the time the control block runs, so
  the process is launched with `cwd=tmpdir` and simple relative file names
  are used (e.g. `v_n_in.dat`, `i_Vsense.dat`).
* ngspice does not exit after simulation unless `.control` contains `quit`.
* `detect_ngspice` searches PATH first, then checks a set of common Windows
  install paths as a fallback.
* The `-b` flag runs batch mode (no GUI).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foster_eom.spice.netlist import SpiceNetlist

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NgspiceNotFoundError(RuntimeError):
    """Raised when ngspice is not found on PATH or common install paths."""


class NgspiceRunError(RuntimeError):
    """Raised when ngspice exits non-zero or produces no data."""


# ---------------------------------------------------------------------------
# Common Windows install paths for ngspice
# ---------------------------------------------------------------------------

_WINDOWS_NGSPICE_CANDIDATES: list[str] = [
    r"C:\ngspice47\Spice64\bin\ngspice.exe",
    r"C:\ngspice\Spice64\bin\ngspice.exe",
    r"C:\ngspice\bin\ngspice.exe",
    r"C:\Program Files\ngspice\bin\ngspice.exe",
    r"C:\Program Files (x86)\ngspice\bin\ngspice.exe",
]


def _find_ngspice_exe() -> str | None:
    """Return the path to ngspice executable, or None."""
    # 1. Search PATH
    exe = shutil.which("ngspice")
    if exe:
        return exe
    # 2. Check Windows fallback candidates
    for candidate in _WINDOWS_NGSPICE_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NgspiceResult:
    """Raw AC analysis result from ngspice.

    All complex values are unit-source small-signal phasors (from `Vsrc AC 1 0`).
    Multiply by `source_spec.vth_phasor` in Python before comparison.

    Parameters
    ----------
    frequencies_hz : np.ndarray
        Frequencies matching the `.AC` command.
    node_voltages : dict[str, np.ndarray]
        SPICE node name -> complex voltage array (unit source).
    sense_currents : dict[str, np.ndarray]
        Sense source name -> complex current array (unit source).
        Positive current = into DUT for `Vsense`; into branch for branch sensors.
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
    """Return ngspice version string, or None if not found."""
    exe = _find_ngspice_exe()
    if exe is None:
        return None
    try:
        # Run a minimal netlist in batch mode (-b) with quit in .control.
        # ngspice hangs waiting for input unless .control has 'quit'.
        # ngspice-47 on Windows outputs nothing to stdout/stderr in batch mode.
        test_netlist = ".title version_check\n.control\nquit\n.endc\n.end\n"
        with tempfile.TemporaryDirectory() as td:
            sp_path = Path(td) / "ver.sp"
            sp_path.write_text(test_netlist, encoding="utf-8")
            result = subprocess.run(
                [exe, "-b", str(sp_path)],
                capture_output=True,
                text=True,
                timeout=10.0,
                cwd=td,
            )
        output = result.stdout + result.stderr
        # Extract version from output if present
        m = re.search(r"ngspice[-\s]+(\d[\d.]*)", output, re.IGNORECASE)
        if m:
            return m.group(0)
        # Fall back: extract version from executable path (e.g. C:\ngspice47\...)
        m2 = re.search(r"ngspice[-_]?(\d+)", exe, re.IGNORECASE)
        if m2:
            return f"ngspice-{m2.group(1)}"
        return "ngspice (version unknown)"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Netlist .control block builder (Windows-compatible file output)
# ---------------------------------------------------------------------------


def _build_control_netlist(
    netlist: SpiceNetlist,
    sense_names: list[str],
    node_names: list[str],
    var_file_map: dict[str, str],
    freqs_for_irregular: list[float] | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Inject a .control block into the netlist.

    For regular grids (LIN/DEC), one ``.AC`` command is used and each variable
    gets a single wrdata file (``var_file_map`` values).

    For irregular grids, we interleave ``ac lin 1 f f`` + ``wrdata`` calls
    so each frequency gets its own numbered file (``v_in_f0.dat``,
    ``v_in_f1.dat``, …).  Python merges them after the run.

    Returns
    -------
    (netlist_text, freq_file_map)
        ``freq_file_map[var_key]`` is the list of per-frequency filenames
        (single-element list for regular grids, N-element for irregular).
    """
    base = netlist.netlist_text
    lines = base.rstrip().split("\n")
    if lines and lines[-1].strip().lower() == ".end":
        lines = lines[:-1]

    # Strip any existing .control/.endc block
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

    # Build .control block
    ctrl: list[str] = [".control"]
    ac_cmd = netlist.ac_command
    is_irregular = ac_cmd.strip().lower().startswith(".control")

    # freq_file_map: var_key -> list[filename]
    freq_file_map: dict[str, list[str]] = {}

    if is_irregular and freqs_for_irregular:
        # Interleave per-frequency: ac lin 1 f f → wrdata per var → repeat
        for fi, f_hz in enumerate(freqs_for_irregular):
            ctrl.append(f"  ac lin 1 {f_hz:.10g} {f_hz:.10g}")
            for nname in node_names:
                vkey = f"v({nname.lower()})"
                safe = re.sub(r"[^a-zA-Z0-9_]", "_", nname)
                fname = f"v_{safe}_f{fi}.dat"
                ctrl.append(f"  wrdata {fname} v({nname})")
                freq_file_map.setdefault(vkey, []).append(fname)
            for sname in sense_names:
                ikey = f"i({sname.lower()})"
                safe = re.sub(r"[^a-zA-Z0-9_]", "_", sname)
                fname = f"i_{safe}_f{fi}.dat"
                ctrl.append(f"  wrdata {fname} i({sname})")
                freq_file_map.setdefault(ikey, []).append(fname)
    else:
        # Regular LIN/DEC: single .AC command then wrdata
        ctrl.append("  " + ac_cmd.lstrip("."))
        for nname in node_names:
            vkey = f"v({nname.lower()})"
            fname = var_file_map.get(vkey, f"v_{nname.lower()}.dat")
            ctrl.append(f"  wrdata {fname} v({nname})")
            freq_file_map[vkey] = [fname]
        for sname in sense_names:
            ikey = f"i({sname.lower()})"
            fname = var_file_map.get(ikey, f"i_{sname.lower()}.dat")
            ctrl.append(f"  wrdata {fname} i({sname})")
            freq_file_map[ikey] = [fname]

    ctrl.append("  quit")
    ctrl.append(".endc")
    ctrl.append("")
    ctrl.append(".end")

    text = "\n".join(filtered) + "\n" + "\n".join(ctrl) + "\n"
    return text, freq_file_map


# ---------------------------------------------------------------------------
# ASCII wrdata parser (simple 3-column format: freq re im)
# ---------------------------------------------------------------------------


def _parse_wrdata_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a single wrdata file: `freq  real  imag` per row.

    Returns (frequencies_hz, complex_values).
    """
    freqs: list[float] = []
    vals: list[complex] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("*") or line.startswith("#"):
                continue
            parts = line.split()
            # Each wrdata row for a single variable has exactly 3 columns.
            if len(parts) >= 3:
                try:
                    f_hz = float(parts[0])
                    re_ = float(parts[1])
                    im = float(parts[2])
                    freqs.append(f_hz)
                    vals.append(complex(re_, im))
                except ValueError:
                    pass

    return np.array(freqs), np.array(vals, dtype=complex)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_ngspice(
    netlist: SpiceNetlist,
    work_dir: Path | None = None,
    timeout_s: float = 30.0,
) -> NgspiceResult:
    """Write netlist to temp dir, run ngspice -b, parse per-variable ASCII files.

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
        If ngspice is not found on PATH or common Windows install paths.
    NgspiceRunError
        If ngspice exits non-zero or no frequencies are parsed.
    """
    exe = _find_ngspice_exe()
    if exe is None:
        raise NgspiceNotFoundError(
            "ngspice not found on PATH or common install paths. "
            "Install ngspice or skip SPICE validation."
        )

    version = detect_ngspice() or "ngspice (detected)"

    # Determine what to probe
    sense_names = list(netlist.sense_source_map.values())
    node_names = [v for v in netlist.node_map.values() if v != "0"]
    freqs_list = list(netlist.frequencies_hz)

    # Base var -> filename mapping for regular-grid case (simple names, no separators)
    var_file_map: dict[str, str] = {}
    for nname in node_names:
        vkey = f"v({nname.lower()})"
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", nname)
        var_file_map[vkey] = f"v_{safe}.dat"
    for sname in sense_names:
        ikey = f"i({sname.lower()})"
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", sname)
        var_file_map[ikey] = f"i_{safe}.dat"

    ctx_mgr = (
        tempfile.TemporaryDirectory() if work_dir is None else _NullContextManager(str(work_dir))
    )

    with ctx_mgr as tmpdir:
        tmppath = Path(tmpdir)
        ctrl_text, freq_file_map = _build_control_netlist(
            netlist,
            sense_names,
            node_names,
            var_file_map,
            freqs_for_irregular=freqs_list,
        )
        nl_path = tmppath / "circuit.sp"
        nl_path.write_text(ctrl_text, encoding="utf-8")

        try:
            proc = subprocess.run(
                [exe, "-b", str(nl_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmpdir,  # wrdata uses relative paths from CWD
            )
        except subprocess.TimeoutExpired as exc:
            raise NgspiceRunError(f"ngspice timed out after {timeout_s}s") from exc
        except FileNotFoundError as exc:
            raise NgspiceNotFoundError(f"ngspice executable not found: {exe}") from exc

        stdout = proc.stdout
        stderr = proc.stderr

        # ngspice on Windows may return exit code 1 even on success.
        # Only fail on returncode >= 2.
        if proc.returncode >= 2:
            raise NgspiceRunError(f"ngspice exited {proc.returncode}.\nSTDERR:\n{stderr[:2000]}")

        # Parse per-variable files (merge per-frequency files for irregular grids)
        node_voltages: dict[str, np.ndarray] = {}
        sense_currents: dict[str, np.ndarray] = {}
        freq_arr: np.ndarray | None = None

        for nname in node_names:
            vkey = f"v({nname.lower()})"
            fnames = freq_file_map.get(vkey, [var_file_map.get(vkey, "")])
            all_freqs: list[float] = []
            all_vals: list[complex] = []
            for fname in fnames:
                fpath = tmppath / fname
                if fpath.exists():
                    f_hz, vals = _parse_wrdata_file(fpath)
                    all_freqs.extend(f_hz.tolist())
                    all_vals.extend(vals.tolist())
            if all_freqs:
                if freq_arr is None:
                    freq_arr = np.array(all_freqs)
                node_voltages[nname] = np.array(all_vals, dtype=complex)

        for sname in sense_names:
            ikey = f"i({sname.lower()})"
            fnames = freq_file_map.get(ikey, [var_file_map.get(ikey, "")])
            all_freqs2: list[float] = []
            all_vals2: list[complex] = []
            for fname in fnames:
                fpath = tmppath / fname
                if fpath.exists():
                    f_hz, vals = _parse_wrdata_file(fpath)
                    all_freqs2.extend(f_hz.tolist())
                    all_vals2.extend(vals.tolist())
            if all_freqs2:
                if freq_arr is None:
                    freq_arr = np.array(all_freqs2)
                sense_currents[sname] = np.array(all_vals2, dtype=complex)

        if freq_arr is None or len(freq_arr) == 0:
            raise NgspiceRunError(
                f"ngspice produced no frequency data.\n"
                f"STDOUT:\n{stdout[:2000]}\nSTDERR:\n{stderr[:2000]}"
            )

    return NgspiceResult(
        frequencies_hz=freq_arr,
        node_voltages=node_voltages,
        sense_currents=sense_currents,
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
