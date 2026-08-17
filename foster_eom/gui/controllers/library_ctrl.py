"""Controller for library management."""

from __future__ import annotations

from foster_eom.catalog.library import ComponentLibrary
from foster_eom.catalog.vendor_pack import VendorPackWorkflow
from foster_eom.gui.view_models.library_vm import LibraryStats, PartRow


class LibraryCtrl:
    @staticmethod
    def get_stats(path: str) -> LibraryStats:
        """Open library, query stats, return VM."""
        lib = ComponentLibrary(path)
        try:
            parts = []

            # Simple query to get all parts for the table
            from foster_eom.catalog.query import ComponentQuery

            query = ComponentQuery()
            components = lib.query(query)

            n_measured = 0
            n_parametric = 0
            n_ideal = 0

            for c in components:
                conditions = lib.get_model_conditions(c.id)

                tier_str = "None"
                validity = "None"
                origin = "None"

                if conditions:
                    best_cond = conditions[0]
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

                parts.append(
                    PartRow(
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
                n_measured=n_measured,
                n_parametric=n_parametric,
                n_ideal=n_ideal,
                parts=parts,
                sha256=lib.library_sha256(),
            )
        finally:
            lib.close()

    @staticmethod
    def import_pack(pack_path: str, lib_path: str) -> str:
        """Import a vendor pack into the library.

        Returns the new library SHA256.
        """
        lib = ComponentLibrary(lib_path)
        try:
            wf = VendorPackWorkflow()
            wf.run(pack_path, lib)
            return lib.library_sha256()
        finally:
            lib.close()
