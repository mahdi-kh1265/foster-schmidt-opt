"""Component library with SQLite backend (Prompt 08).

``ComponentLibrary`` provides CRUD, query, model construction, and import
operations backed by a SQLite database and content-addressed file store.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from foster_eom.catalog.component import (
    ComponentKind,
    FallbackPolicy,
    LibraryComponent,
    ModelCondition,
    ModelOrigin,
    ModelTier,
    tier_rank,
)
from foster_eom.catalog.file_store import ContentAddressedStore
from foster_eom.catalog.model_bridge import build_model
from foster_eom.catalog.query import ComponentQuery
from foster_eom.catalog.schema import create_db
from foster_eom.models.base import OnePortModel


class ComponentLibrary:
    """SQLite-backed component library with content-addressed file store.

    Parameters
    ----------
    path : str | Path
        Path to the ``.fseom.db`` file. Created if it does not exist.
        The ``models/`` file store directory is created alongside it.
    """

    def __init__(self, path: str | Path) -> None:
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = create_db(self.db_path)
        self.file_store = ContentAddressedStore(self.db_path.parent / "models")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> ComponentLibrary:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def add(
        self,
        component: LibraryComponent,
        on_conflict: str = "error",
    ) -> str:
        """Add a component to the library.

        Returns
        -------
        str
            The component ID on insert, or ``'__skipped_dup__'`` for
            idempotent no-ops (same content SHA).

        Raises
        ------
        ValueError
            On content mismatch when ``on_conflict='error'``.
        """
        if on_conflict not in ("error", "merge", "replace"):
            raise ValueError(
                f"on_conflict must be 'error', 'merge', or 'replace', got '{on_conflict}'."
            )

        # Ensure content_sha256 is computed
        if not component.content_sha256:
            component.content_sha256 = component.compute_content_sha256()

        # Check for existing
        cur = self._conn.execute(
            "SELECT id, content_sha256 FROM components WHERE vendor = ? AND part_number = ?",
            (component.vendor, component.part_number),
        )
        existing = cur.fetchone()

        if existing is not None:
            if existing["content_sha256"] == component.content_sha256:
                return "__skipped_dup__"  # idempotent no-op

            if on_conflict == "error":
                raise ValueError(
                    f"Component '{component.vendor}/{component.part_number}' "
                    f"already exists with different content. Use "
                    f"on_conflict='merge' or 'replace' to update."
                )

            existing_id: str = existing["id"]
            if on_conflict == "replace":
                self._replace_component(existing_id, component)
            else:  # merge
                self._merge_component(existing_id, component)
            return existing_id

        # Insert new
        self._insert_component(component)

        # Always add an ideal model condition
        ideal_mc = ModelCondition(
            id=str(uuid4()),
            component_id=component.id,
            model_tier=ModelTier.IDEAL,
            model_origin=ModelOrigin.IDEAL,
            import_ts=component.import_ts,
        )
        self._insert_model_condition(ideal_mc)

        return component.id

    def get(self, component_id: str) -> LibraryComponent:
        """Get a component by ID."""
        cur = self._conn.execute("SELECT * FROM components WHERE id = ?", (component_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"No component with id '{component_id}'.")
        return self._row_to_component(row)

    def get_by_part(self, vendor: str, part_number: str) -> LibraryComponent:
        """Get a component by vendor + part number."""
        cur = self._conn.execute(
            "SELECT * FROM components WHERE vendor = ? AND part_number = ?",
            (vendor, part_number),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"No component for vendor='{vendor}', part='{part_number}'.")
        return self._row_to_component(row)

    def delete(self, component_id: str) -> None:
        """Delete a component and cascade-delete its model conditions."""
        cur = self._conn.execute("DELETE FROM components WHERE id = ?", (component_id,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"No component with id '{component_id}'.")

    def add_model_condition(self, condition: ModelCondition) -> str:
        """Add a model condition record for a component."""
        if not condition.id:
            condition.id = str(uuid4())
        self._insert_model_condition(condition)
        return condition.id

    def get_model_conditions(self, component_id: str) -> list[ModelCondition]:
        """Get all model conditions for a component."""
        cur = self._conn.execute(
            "SELECT * FROM model_conditions WHERE component_id = ? "
            "ORDER BY model_tier, variant_label",
            (component_id,),
        )
        return [self._row_to_model_condition(r) for r in cur.fetchall()]

    # -------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------

    def query(self, spec: ComponentQuery) -> list[LibraryComponent]:
        """Query for components matching all filter criteria."""
        clauses: list[str] = []
        params: list[Any] = []

        if spec.kind is not None:
            clauses.append("c.kind = ?")
            params.append(spec.kind.value)
        if spec.vendor is not None:
            clauses.append("c.vendor = ?")
            params.append(spec.vendor)
        if spec.package is not None:
            clauses.append("c.package = ?")
            params.append(spec.package)
        if spec.value_min is not None:
            clauses.append("c.value_nom >= ?")
            params.append(spec.value_min)
        if spec.value_max is not None:
            clauses.append("c.value_nom <= ?")
            params.append(spec.value_max)
        if spec.tol_max_frac is not None:
            clauses.append("c.value_tol_frac IS NOT NULL AND c.value_tol_frac <= ?")
            params.append(spec.tol_max_frac)
        if spec.voltage_min_v is not None:
            clauses.append("c.voltage_max_v IS NOT NULL AND c.voltage_max_v >= ?")
            params.append(spec.voltage_min_v)
        if spec.current_min_a is not None:
            clauses.append("c.current_max_a IS NOT NULL AND c.current_max_a >= ?")
            params.append(spec.current_min_a)
        if spec.current_sat_min_a is not None:
            clauses.append("c.current_sat_a IS NOT NULL AND c.current_sat_a >= ?")
            params.append(spec.current_sat_min_a)
        if spec.in_stock_only:
            clauses.append("c.stock_status = 'in_stock'")
        if spec.part_number_glob is not None:
            clauses.append("c.part_number LIKE ?")
            params.append(spec.part_number_glob)

        # Joins with model_conditions for RF/model filters
        need_mc_join = any(
            [
                spec.srf_min_hz,
                spec.q_min,
                spec.esr_max_ohm,
                spec.model_tier_min,
                spec.freq_range_hz,
            ]
        )

        if need_mc_join:
            mc_clauses: list[str] = []
            mc_params: list[Any] = []

            if spec.srf_min_hz is not None:
                mc_clauses.append("mc.srf_hz IS NOT NULL AND mc.srf_hz >= ?")
                mc_params.append(spec.srf_min_hz)
            if spec.q_min is not None:
                # Q is meaningful only with q_at_f_hz
                mc_clauses.append(
                    "mc.q_value IS NOT NULL AND mc.q_at_f_hz IS NOT NULL AND mc.q_value >= ?"
                )
                mc_params.append(spec.q_min)
            if spec.esr_max_ohm is not None:
                mc_clauses.append("mc.esr_ohm IS NOT NULL AND mc.esr_ohm <= ?")
                mc_params.append(spec.esr_max_ohm)
            if spec.model_tier_min is not None:
                eligible_tiers = [
                    t.value for t in ModelTier if tier_rank(t) >= tier_rank(spec.model_tier_min)
                ]
                placeholders = ",".join("?" * len(eligible_tiers))
                mc_clauses.append(f"mc.model_tier IN ({placeholders})")
                mc_params.extend(eligible_tiers)
            if spec.freq_range_hz is not None:
                lo, hi = spec.freq_range_hz
                mc_clauses.append(
                    "mc.validity_hz_lo IS NOT NULL AND mc.validity_hz_hi IS NOT NULL "
                    "AND mc.validity_hz_lo <= ? AND mc.validity_hz_hi >= ?"
                )
                mc_params.extend([lo, hi])

            mc_where = " AND ".join(mc_clauses) if mc_clauses else "1=1"
            sql = (
                "SELECT DISTINCT c.* FROM components c "
                "INNER JOIN model_conditions mc ON mc.component_id = c.id "
                f"WHERE {mc_where}"
            )
            if clauses:
                sql += " AND " + " AND ".join(clauses)
            all_params = mc_params + params
        else:
            where = " AND ".join(clauses) if clauses else "1=1"
            sql = f"SELECT * FROM components c WHERE {where}"
            all_params = params

        sql += " ORDER BY c.vendor, c.part_number"

        cur = self._conn.execute(sql, all_params)
        return [self._row_to_component(r) for r in cur.fetchall()]

    # -------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------

    def build_model(
        self,
        component_id: str,
        *,
        required_tier: ModelTier | None = None,
        freq_range: tuple[float, float] | None = None,
        fallback: FallbackPolicy = FallbackPolicy.STRICT,
    ) -> OnePortModel:
        """Build a OnePortModel for a library component."""
        comp = self.get(component_id)
        conditions = self.get_model_conditions(component_id)
        return build_model(
            comp,
            conditions,
            self.file_store,
            required_tier=required_tier,
            freq_range=freq_range,
            fallback=fallback,
        )

    # -------------------------------------------------------------------
    # Import helpers
    # -------------------------------------------------------------------

    def import_csv(
        self,
        path: str | Path,
        importer: Any | None = None,
        column_map: dict[str, str] | None = None,
        on_conflict: str = "error",
    ) -> Any:
        """Import components from a CSV file.

        Parameters
        ----------
        path : str | Path
        importer : CatalogImporter | None
            Custom importer. If None, uses ``GenericCSVImporter``.
        column_map : dict | None
            Custom column mapping for ``GenericCSVImporter``.
        on_conflict : str
        """
        from foster_eom.catalog.importers.csv_generic import GenericCSVImporter

        if importer is None:
            importer = GenericCSVImporter(column_map=column_map)
        return importer.import_to(self, Path(path), on_conflict=on_conflict)

    def import_touchstone(
        self,
        directory: str | Path,
        **kwargs: Any,
    ) -> Any:
        """Import Touchstone files from a directory."""
        from foster_eom.catalog.importers.touchstone import TouchstoneImporter

        imp = TouchstoneImporter(**kwargs)
        return imp.import_to(self, Path(directory))

    # -------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------

    def library_sha256(self) -> str:
        """Compute deterministic SHA-256 of the library's logical manifest.

        Hashes a sorted manifest of normalized component metadata + model
        conditions + file SHA references. Does NOT hash raw SQLite bytes.
        """
        h = hashlib.sha256()

        # Components sorted by (vendor, part_number)
        cur = self._conn.execute("SELECT * FROM components ORDER BY vendor, part_number")
        for row in cur.fetchall():
            comp_tuple = (
                row["vendor"],
                row["part_number"],
                row["kind"],
                row["value_nom"],
                row["value_tol_frac"],
                row["package"],
                row["voltage_max_v"],
                row["current_max_a"],
                row["current_sat_a"],
            )
            h.update(repr(comp_tuple).encode("utf-8"))

            # Model conditions for this component, sorted
            mc_cur = self._conn.execute(
                "SELECT * FROM model_conditions WHERE component_id = ? "
                "ORDER BY model_tier, model_origin, variant_label, import_ts",
                (row["id"],),
            )
            for mc_row in mc_cur.fetchall():
                mc_tuple = (
                    mc_row["model_tier"],
                    mc_row["model_origin"],
                    mc_row["model_file_sha256"],
                    mc_row["parametric_params"],
                    mc_row["validity_hz_lo"],
                    mc_row["validity_hz_hi"],
                    mc_row["variant_label"],
                )
                h.update(repr(mc_tuple).encode("utf-8"))

        return h.hexdigest()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _insert_component(self, c: LibraryComponent) -> None:
        self._conn.execute(
            """INSERT INTO components (
                id, kind, vendor, part_number, package, description,
                value_nom, value_tol_frac, voltage_max_v, current_max_a,
                current_sat_a, temp_min_c, temp_max_c, power_max_w,
                stock_status, stock_ts, datasheet_url,
                import_source, import_sha256, import_ts,
                content_sha256, user_notes
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )""",
            (
                c.id,
                c.kind.value,
                c.vendor,
                c.part_number,
                c.package,
                c.description,
                c.value_nom,
                c.value_tol_frac,
                c.voltage_max_v,
                c.current_max_a,
                c.current_sat_a,
                c.temp_min_c,
                c.temp_max_c,
                c.power_max_w,
                c.stock_status,
                c.stock_ts,
                c.datasheet_url,
                c.import_source,
                c.import_sha256,
                c.import_ts,
                c.content_sha256,
                c.user_notes,
            ),
        )
        self._conn.commit()

    def _replace_component(self, existing_id: str, c: LibraryComponent) -> None:
        """Replace all fields of an existing component."""
        c.id = existing_id
        self._conn.execute("DELETE FROM components WHERE id = ?", (existing_id,))
        self._insert_component(c)

    def _merge_component(self, existing_id: str, c: LibraryComponent) -> None:
        """Merge: update only non-None incoming fields that differ."""
        existing = self.get(existing_id)
        updates: dict[str, Any] = {}
        merge_fields = [
            "package",
            "description",
            "value_tol_frac",
            "voltage_max_v",
            "current_max_a",
            "current_sat_a",
            "temp_min_c",
            "temp_max_c",
            "power_max_w",
            "stock_status",
            "stock_ts",
            "datasheet_url",
            "user_notes",
        ]
        for field in merge_fields:
            new_val = getattr(c, field)
            if new_val is not None and new_val != "" and new_val != getattr(existing, field):
                updates[field] = new_val

        if updates:
            # Recompute content SHA
            for k, v in updates.items():
                setattr(existing, k, v)
            updates["content_sha256"] = existing.compute_content_sha256()

            set_clause = ", ".join(f"{k} = ?" for k in updates)
            vals = [*list(updates.values()), existing_id]
            self._conn.execute(
                f"UPDATE components SET {set_clause} WHERE id = ?",
                vals,
            )
            self._conn.commit()

    def _insert_model_condition(self, mc: ModelCondition) -> None:
        self._conn.execute(
            """INSERT INTO model_conditions (
                id, component_id, model_tier, model_origin,
                model_file_sha256, model_file_ext, n_ports,
                fixture_type, fixture_port_z, fixture_port_gnd,
                parametric_params,
                srf_hz, q_at_f_hz, q_value, esr_ohm,
                validity_hz_lo, validity_hz_hi,
                measurement_temp_c, measurement_bias_v,
                variant_label, import_ts
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?
            )""",
            (
                mc.id,
                mc.component_id,
                mc.model_tier.value,
                mc.model_origin.value,
                mc.model_file_sha256,
                mc.model_file_ext,
                mc.n_ports,
                mc.fixture_type,
                mc.fixture_port_z,
                mc.fixture_port_gnd,
                mc.parametric_params_json(),
                mc.srf_hz,
                mc.q_at_f_hz,
                mc.q_value,
                mc.esr_ohm,
                mc.validity_hz_lo,
                mc.validity_hz_hi,
                mc.measurement_temp_c,
                mc.measurement_bias_v,
                mc.variant_label,
                mc.import_ts,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_component(row: sqlite3.Row) -> LibraryComponent:
        return LibraryComponent(
            id=row["id"],
            kind=ComponentKind(row["kind"]),
            vendor=row["vendor"],
            part_number=row["part_number"],
            package=row["package"] or "",
            description=row["description"] or "",
            value_nom=row["value_nom"],
            value_tol_frac=row["value_tol_frac"],
            voltage_max_v=row["voltage_max_v"],
            current_max_a=row["current_max_a"],
            current_sat_a=row["current_sat_a"],
            temp_min_c=row["temp_min_c"],
            temp_max_c=row["temp_max_c"],
            power_max_w=row["power_max_w"],
            stock_status=row["stock_status"],
            stock_ts=row["stock_ts"],
            datasheet_url=row["datasheet_url"],
            import_source=row["import_source"],
            import_sha256=row["import_sha256"],
            import_ts=row["import_ts"],
            content_sha256=row["content_sha256"],
            user_notes=row["user_notes"] or "",
        )

    @staticmethod
    def _row_to_model_condition(row: sqlite3.Row) -> ModelCondition:
        return ModelCondition(
            id=row["id"],
            component_id=row["component_id"],
            model_tier=ModelTier(row["model_tier"]),
            model_origin=ModelOrigin(row["model_origin"]),
            model_file_sha256=row["model_file_sha256"],
            model_file_ext=row["model_file_ext"],
            n_ports=row["n_ports"],
            fixture_type=row["fixture_type"],
            fixture_port_z=row["fixture_port_z"],
            fixture_port_gnd=row["fixture_port_gnd"],
            parametric_params=ModelCondition.parametric_params_from_json(row["parametric_params"]),
            srf_hz=row["srf_hz"],
            q_at_f_hz=row["q_at_f_hz"],
            q_value=row["q_value"],
            esr_ohm=row["esr_ohm"],
            validity_hz_lo=row["validity_hz_lo"],
            validity_hz_hi=row["validity_hz_hi"],
            measurement_temp_c=row["measurement_temp_c"],
            measurement_bias_v=row["measurement_bias_v"],
            variant_label=row["variant_label"] or "",
            import_ts=row["import_ts"],
        )
