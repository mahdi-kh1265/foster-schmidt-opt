"""Controller for library management."""

from __future__ import annotations

from typing import Any

from foster_eom.catalog.library import ComponentLibrary
from foster_eom.catalog.query import ComponentQuery
from foster_eom.catalog.vendor_pack import VendorPackManifest, VendorPackWorkflow
from foster_eom.gui.view_models.library_vm import ComponentDetailsVM, LibraryStats, PartRow


class LibraryCtrl:
    @staticmethod
    def get_stats(path: str, query: ComponentQuery | None = None) -> LibraryStats:
        """Open library, query stats, return VM."""
        lib = ComponentLibrary(path)
        try:
            parts = []

            if query is None:
                query = ComponentQuery()
            components = lib.query(query)

            n_measured = 0
            n_parametric = 0
            n_ideal = 0
            n_inductors = 0
            n_capacitors = 0

            for c in components:
                conditions = lib.get_model_conditions(c.id)

                tier_str = "None"
                validity = "None"
                origin = "None"

                if conditions:
                    best_cond = conditions[-1] if conditions else None
                    if best_cond:
                        tier_str = best_cond.model_tier.value
                        origin = best_cond.model_origin.value

                        if best_cond.validity_hz():
                            v_min, v_max = best_cond.validity_hz()
                            if v_min is not None and v_max is not None:
                                validity = f"{v_min / 1e6:.1f} - {v_max / 1e6:.1f} MHz"

                        if best_cond.model_tier.value == "measured":
                            n_measured += 1
                        elif best_cond.model_tier.value == "parametric":
                            n_parametric += 1
                        elif best_cond.model_tier.value == "ideal":
                            n_ideal += 1

                if c.kind.value == "inductor":
                    n_inductors += 1
                elif c.kind.value == "capacitor":
                    n_capacitors += 1

                parts.append(
                    PartRow(
                        id=c.id,
                        vendor=c.vendor,
                        part_number=c.part_number,
                        kind=c.kind.value,
                        value=f"{c.value_nom:.2e}",
                        tier=tier_str,
                        validity=validity,
                        origin=origin,
                    )
                )

            return LibraryStats(
                total_parts=len(parts),
                n_inductors=n_inductors,
                n_capacitors=n_capacitors,
                n_measured=n_measured,
                n_parametric=n_parametric,
                n_ideal=n_ideal,
                parts=parts,
                sha256=lib.library_sha256(),
            )
        finally:
            lib.close()

    @staticmethod
    def get_component_details(path: str, component_id: str) -> ComponentDetailsVM:
        lib = ComponentLibrary(path)
        try:
            c = lib.get(component_id)
            conditions = lib.get_model_conditions(component_id)

            dcr_esr = "N/A"
            srf = "N/A"
            validity_hz = "N/A"
            model_tier = "N/A"
            model_origin = "N/A"
            model_file_sha256 = "N/A"

            if conditions:
                best_cond = conditions[-1]
                if best_cond.esr_ohm is not None:
                    dcr_esr = f"{best_cond.esr_ohm:.3e} Ω"
                if best_cond.srf_hz is not None:
                    srf = f"{best_cond.srf_hz / 1e6:.2f} MHz"
                if best_cond.validity_hz():
                    v_min, v_max = best_cond.validity_hz()
                    if v_min is not None and v_max is not None:
                        validity_hz = f"{v_min / 1e6:.1f} - {v_max / 1e6:.1f} MHz"

                model_tier = best_cond.model_tier.value
                model_origin = best_cond.model_origin.value
                if best_cond.model_file_sha256:
                    model_file_sha256 = best_cond.model_file_sha256

            return ComponentDetailsVM(
                vendor=c.vendor,
                part_number=c.part_number,
                kind=c.kind.value,
                package=c.package if c.package else "N/A",
                value_nom=f"{c.value_nom:.2e}",
                tolerance=f"{c.value_tol_frac * 100:.1f}%" if c.value_tol_frac is not None else "N/A",
                dcr_esr=dcr_esr,
                srf=srf,
                voltage_rating=f"{c.voltage_max_v:.1f} V" if c.voltage_max_v is not None else "N/A",
                current_rating=f"{c.current_max_a:.1f} A" if c.current_max_a is not None else "N/A",
                validity_hz=validity_hz,
                model_tier=model_tier,
                model_origin=model_origin,
                model_file_sha256=model_file_sha256,
                is_synthetic=(c.vendor == "POSM-DEMO"),
            )
        finally:
            lib.close()

    @staticmethod
    def import_pack(pack_spec: Any, lib_path: str) -> VendorPackManifest:
        """Import a vendor pack into the library.

        Returns the VendorPackManifest.
        """
        lib = ComponentLibrary(lib_path)
        try:
            wf = VendorPackWorkflow(lib)
            return wf.run(pack_spec)
        finally:
            lib.close()
