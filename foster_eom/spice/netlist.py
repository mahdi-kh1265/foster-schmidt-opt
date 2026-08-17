"""P11 SPICE netlist builder.

Converts a CircuitGraph + SourceSpec to a deterministic ngspice netlist.

Convention
----------
* Source: ``Vsrc <pos> <neg> AC 1 0`` (unit phasor, zero phase).
  Python scales SPICE complex output by ``source_spec.vth_phasor`` after run.
  No sqrt(2) factor.
* A 0-V sense source ``Vsense <n_jct> <n_dut> DC 0`` is inserted in series
  at the DUT reference plane.  ``I(Vsense) > 0`` = into DUT.
  ``Z_in = V(n_dut) / I(Vsense)``.
* Branch sense sources inserted per ``MeasurementPlan.branch_element_ids``.
* ``TabularImpedanceComponent`` and unknown model types are deferred:
  added to ``unsupported_elements``.  Caller must NOT run a partial topology.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from foster_eom.circuit.graph import CircuitGraph, ElementKind
from foster_eom.domain.source import SourceSpec
from foster_eom.models.components import (
    IdealCapacitor,
    IdealInductor,
    IdealResistor,
    LumpedLossyCapacitor,
    LumpedLossyInductor,
    TabularImpedanceComponent,
)
from foster_eom.spice.result import MeasurementPlan

# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

_REPLACE_MAP = str.maketrans(
    {
        "/": "_",
        "[": "_",
        "]": "_",
        " ": "_",
        ".": "_",
        "(": "_",
        ")": "_",
        ",": "_",
        "-": "_",
    }
)


def _sanitize(name: str, prefix: str, used: set[str]) -> str:
    """Return a collision-safe, deterministic SPICE token.

    * Replace problematic chars.
    * Prepend prefix if starts with digit or empty.
    * Append 4-char hex suffix on collision.
    """
    s = name.translate(_REPLACE_MAP)
    if not s:
        s = prefix + "x"
    if s[0].isdigit():
        s = prefix + s
    candidate = s
    counter = 0
    while candidate in used:
        counter += 1
        candidate = s + "_" + format((abs(hash(name)) + counter) & 0xFFFF, "04x")
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# SpiceNetlist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpiceNetlist:
    """Immutable SPICE netlist container."""

    title: str
    netlist_text: str
    sha256: str
    node_map: dict[str, str]
    element_map: dict[str, str]
    sense_source_map: dict[str, str]
    unsupported_elements: list[str]
    unsupported_model_reasons: dict[str, str]
    source_vth_phasor: complex
    source_phase_deg: float
    frequencies_hz: tuple[float, ...]
    ac_command: str


# ---------------------------------------------------------------------------
# Frequency grid detection
# ---------------------------------------------------------------------------


def _detect_grid(frequencies_hz: Sequence[float]) -> str | None:
    """Detect LIN/DEC grid; return .AC command or None for irregular."""
    freqs = list(frequencies_hz)
    n = len(freqs)
    if n < 2:
        return None
    f0, f1 = freqs[0], freqs[-1]
    if f0 <= 0 or f1 <= f0:
        return None

    # LIN check
    step = (f1 - f0) / (n - 1)
    if step > 0:
        recon = [f0 + i * step for i in range(n)]
        if all(abs(recon[i] - freqs[i]) / max(abs(freqs[i]), 1e-30) < 1e-9 for i in range(n)):
            return f".AC LIN {n} {f0:.6g} {f1:.6g}"

    # DEC check
    if f0 > 0:
        log0, log1 = math.log10(f0), math.log10(f1)
        if log1 > log0:
            log_step = (log1 - log0) / (n - 1)
            recon = [10 ** (log0 + i * log_step) for i in range(n)]
            if all(abs(recon[i] - freqs[i]) / max(abs(freqs[i]), 1e-30) < 1e-9 for i in range(n)):
                n_per_dec = (n - 1) / (log1 - log0)
                if abs(n_per_dec - round(n_per_dec)) < 1e-6:
                    pts = round(n_per_dec)
                    return f".AC DEC {pts} {f0:.6g} {f1:.6g}"

    return None


# ---------------------------------------------------------------------------
# Subcircuit templates
# ---------------------------------------------------------------------------


def _lossy_inductor_subckt(model: LumpedLossyInductor, subckt_name: str) -> str:
    """SPICE subcircuit for LumpedLossyInductor.

    Topology: Z = (R_dcr + jwL) || (1/jwC_par)
    Ports: p, n.  Internal node: nd_i1 (between R and L).
    C_par connects p-n (parallel with the whole series arm).
    """
    lines = [
        f".SUBCKT {subckt_name} p n",
        f"* LumpedLossyInductor L={model.l_h:.6g}H "
        f"R_dcr={model.r_dcr_ohm:.6g}ohm C_par={model.c_par_f:.6g}F",
        f"Rdcr p nd_i1 {model.r_dcr_ohm:.10g}",
        f"Lmain nd_i1 n {model.l_h:.10g}",
    ]
    if model.c_par_f > 0.0:
        lines.append(f"Cpar p n {model.c_par_f:.10g}")
    lines.append(".ENDS")
    return "\n".join(lines)


def _lossy_capacitor_subckt(model: LumpedLossyCapacitor, subckt_name: str) -> str:
    """SPICE subcircuit for LumpedLossyCapacitor.

    Topology: Z = R_esr + jwL_esl + 1/(jwC)
    Ports: p, n.  Internal nodes: nd_i1, nd_i2.
    """
    lines = [
        f".SUBCKT {subckt_name} p n",
        f"* LumpedLossyCapacitor C={model.c_f:.6g}F "
        f"R_esr={model.r_esr_ohm:.6g}ohm L_esl={model.l_esl_h:.6g}H",
        f"Resr p nd_i1 {model.r_esr_ohm:.10g}",
        f"Lesl nd_i1 nd_i2 {model.l_esl_h:.10g}",
        f"Cmain nd_i2 n {model.c_f:.10g}",
        ".ENDS",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_netlist(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    frequencies_hz: Sequence[float],
    title: str = "foster_eom",
    measurement_plan: MeasurementPlan | None = None,
    catalog_combo: Any | None = None,
) -> SpiceNetlist:
    """Build a deterministic SPICE netlist from a CircuitGraph.

    If any element has an unsupported model, ``unsupported_elements`` is
    non-empty.  The caller must return ``status="unsupported"`` and must NOT
    invoke ngspice with a partial topology.
    """
    plan = measurement_plan or MeasurementPlan()
    freqs = list(frequencies_hz)

    # ---- Node map ----------------------------------------------------------
    used_nodes: set[str] = {"0"}
    node_map: dict[str, str] = {graph.ground_node_id: "0"}
    for nid in sorted(graph.nodes):
        if nid == graph.ground_node_id:
            continue
        node_map[nid] = _sanitize(nid, "n", used_nodes)

    # ---- Unsupported scan --------------------------------------------------
    unsupported_elements: list[str] = []
    unsupported_model_reasons: dict[str, str] = {}
    for eid in sorted(graph.elements):
        elem = graph.elements[eid]
        if elem.kind == ElementKind.ONE_PORT_MODEL:
            m = elem.model
            if isinstance(
                m,
                (
                    IdealResistor,
                    IdealInductor,
                    IdealCapacitor,
                    LumpedLossyInductor,
                    LumpedLossyCapacitor,
                ),
            ):
                pass
            elif isinstance(m, TabularImpedanceComponent):
                unsupported_elements.append(eid)
                unsupported_model_reasons[eid] = (
                    "tabular_component: no validated one-port-impedance SPICE "
                    "embedding implemented in V1; deferred."
                )
            else:
                mtype = type(m).__name__ if m is not None else "None"
                unsupported_elements.append(eid)
                unsupported_model_reasons[eid] = f"unsupported_model_type:{mtype}"

    # ---- Subcircuit definitions --------------------------------------------
    subckt_lines: list[str] = []
    subckt_name_map: dict[str, str] = {}
    used_subckts: set[str] = set()
    for eid in sorted(graph.elements):
        elem = graph.elements[eid]
        if elem.kind != ElementKind.ONE_PORT_MODEL:
            continue
        m = elem.model
        if isinstance(m, LumpedLossyInductor):
            sname = _sanitize(f"LLOSSY_{eid}", "S", used_subckts)
            subckt_name_map[eid] = sname
            subckt_lines.append(_lossy_inductor_subckt(m, sname))
        elif isinstance(m, LumpedLossyCapacitor):
            sname = _sanitize(f"CLOSSY_{eid}", "S", used_subckts)
            subckt_name_map[eid] = sname
            subckt_lines.append(_lossy_capacitor_subckt(m, sname))

    # ---- Provenance --------------------------------------------------------
    provenance: dict[str, dict[str, Any]] = {}
    if catalog_combo is not None:
        entries = getattr(catalog_combo, "slot_entries", {})
        for eid, entry in entries.items():
            provenance[eid] = {
                "part_number": getattr(entry, "part_number", "?"),
                "model_tier": getattr(entry, "model_tier", None),
            }

    # ---- Sense source plan -------------------------------------------------
    used_extra: set[str] = set(node_map.values())
    sense_source_map: dict[str, str] = {}

    n_dut = node_map[graph.input_port.node_pos]
    n_src_pos = _sanitize("n_src_pos", "n", used_extra)
    n_sense_jct = _sanitize("n_sense_jct", "n", used_extra)
    sense_source_map["__input__"] = "Vsense"

    # Branch sense sources
    branch_sense: dict[str, tuple[str, str]] = {}
    for eid in list(plan.branch_element_ids):
        if eid not in graph.elements:
            continue
        vsname = _sanitize(f"Vsns_{eid}", "V", used_extra)
        n_jct = _sanitize(f"n_jct_{eid}", "n", used_extra)
        branch_sense[eid] = (vsname, n_jct)
        sense_source_map[eid] = vsname

    # ---- Element names -----------------------------------------------------
    used_elems: set[str] = set()
    pfx_map = {
        ElementKind.RESISTOR: "R",
        ElementKind.INDUCTOR: "L",
        ElementKind.CAPACITOR: "C",
        ElementKind.ONE_PORT_MODEL: "X",
    }
    element_map: dict[str, str] = {}
    for eid in sorted(graph.elements):
        elem = graph.elements[eid]
        pfx = pfx_map.get(elem.kind, "X")
        element_map[eid] = _sanitize(f"{pfx}_{eid}", pfx, used_elems)

    # ---- AC command --------------------------------------------------------
    ac_command = _detect_grid(freqs) or _explicit_ac_cmd(freqs)

    # ---- Netlist text ------------------------------------------------------
    n_src_neg = node_map[graph.input_port.node_neg]
    eom_id = plan.eom_element_id or graph.eom_element_id

    L: list[str] = []
    L.append(f"* {title}")
    L.append("* foster_eom P11 SPICE export")
    L.append("* source_convention: spice=AC_1_0_unit_phasor,scale=vth_phasor_in_python")
    L.append("* current_direction_convention: Vsense_oriented:I(Vsense)>0_into_DUT")
    L.append(f"* vth_phasor={source_spec.vth_phasor!r}  phase_deg={source_spec.phase_deg!r}")
    L.append("")

    for sc in subckt_lines:
        L.append(sc)
        L.append("")

    L.append("* --- Source ---")
    L.append(
        f"Vsrc {n_src_pos} {n_src_neg} AC 1 0  $ unit phasor; multiply by vth_phasor in Python"
    )
    rs_val = source_spec.z_source_real_ohm
    L.append(f"Rs {n_src_pos} {n_sense_jct} {rs_val:.10g}  $ source impedance")
    L.append(
        f"Vsense {n_sense_jct} {n_dut} DC 0"
        f"  $ sense source: I(Vsense)>0 into DUT; Z_in=V({n_dut})/I(Vsense)"
    )
    L.append("")

    L.append("* --- Passive elements ---")
    for eid in sorted(graph.elements):
        elem = graph.elements[eid]
        ename = element_map[eid]
        prov = provenance.get(eid, {})
        prov_str = f" part={prov['part_number']} tier={prov['model_tier']}" if prov else ""
        comment = f"$ {eid} [{elem.kind.value}]{prov_str}"

        if eid in unsupported_elements:
            L.append(f"* UNSUPPORTED: {eid}: {unsupported_model_reasons[eid]}")
            continue

        pos_node = node_map[elem.node_pos]
        neg_node = node_map[elem.node_neg]

        if eid in branch_sense:
            vsname, n_jct = branch_sense[eid]
            L.append(f"{vsname} {pos_node} {n_jct} DC 0  $ branch sense for {eid}")
            pos_node = n_jct

        if (
            elem.kind == ElementKind.RESISTOR
            or elem.kind == ElementKind.INDUCTOR
            or elem.kind == ElementKind.CAPACITOR
        ):
            assert elem.value is not None
            L.append(f"{ename} {pos_node} {neg_node} {elem.value:.10g}  {comment}")
        elif elem.kind == ElementKind.ONE_PORT_MODEL:
            m = elem.model
            if isinstance(m, IdealResistor):
                L.append(f"R_{ename} {pos_node} {neg_node} {m.r_ohm:.10g}  {comment}")
            elif isinstance(m, IdealInductor):
                L.append(f"L_{ename} {pos_node} {neg_node} {m.l_h:.10g}  {comment}")
            elif isinstance(m, IdealCapacitor):
                L.append(f"C_{ename} {pos_node} {neg_node} {m.c_f:.10g}  {comment}")
            elif isinstance(m, (LumpedLossyInductor, LumpedLossyCapacitor)):
                sname = subckt_name_map[eid]
                L.append(f"{ename} {pos_node} {neg_node} {sname}  {comment}")

    L.append("")

    if eom_id and eom_id in graph.elements:
        eom_elem = graph.elements[eom_id]
        ep = node_map[eom_elem.node_pos]
        en = node_map[eom_elem.node_neg]
        L.append(f"* EOM: {eom_id}  V_eom=V({ep})-V({en})")
        L.append("")

    L.append("* --- Analysis ---")
    L.append(ac_command)
    L.append("")
    L.append(".end")

    text = "\n".join(L) + "\n"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return SpiceNetlist(
        title=title,
        netlist_text=text,
        sha256=sha,
        node_map=node_map,
        element_map=element_map,
        sense_source_map=sense_source_map,
        unsupported_elements=unsupported_elements,
        unsupported_model_reasons=unsupported_model_reasons,
        source_vth_phasor=source_spec.vth_phasor,
        source_phase_deg=source_spec.phase_deg,
        frequencies_hz=tuple(freqs),
        ac_command=ac_command,
    )


def _explicit_ac_cmd(freqs: list[float]) -> str:
    """Produce a .control block of per-frequency AC solves for irregular grids."""
    inner = "\n  ".join(f"ac lin 1 {f:.10g} {f:.10g}" for f in freqs)
    return f".control\n  {inner}\n.endc"
