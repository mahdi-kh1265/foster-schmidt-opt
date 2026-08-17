"""SQLite schema management (Prompt 08).

Creates and migrates the component library database schema.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

CURRENT_SCHEMA_VERSION = 1

_SCHEMA_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS components (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    vendor          TEXT NOT NULL,
    part_number     TEXT NOT NULL,
    package         TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    value_nom       REAL NOT NULL,
    value_tol_frac  REAL,
    voltage_max_v   REAL,
    current_max_a   REAL,
    current_sat_a   REAL,
    temp_min_c      REAL,
    temp_max_c      REAL,
    power_max_w     REAL,
    stock_status    TEXT,
    stock_ts        TEXT,
    datasheet_url   TEXT,
    import_source   TEXT NOT NULL,
    import_sha256   TEXT,
    import_ts       TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    user_notes      TEXT NOT NULL DEFAULT '',
    UNIQUE(vendor, part_number)
);

CREATE TABLE IF NOT EXISTS model_conditions (
    id                 TEXT PRIMARY KEY,
    component_id       TEXT NOT NULL REFERENCES components(id) ON DELETE CASCADE,
    model_tier         TEXT NOT NULL,
    model_origin       TEXT NOT NULL,
    model_file_sha256  TEXT,
    model_file_ext     TEXT,
    n_ports            INTEGER,
    fixture_type       TEXT,
    fixture_port_z     INTEGER,
    fixture_port_gnd   INTEGER,
    parametric_params  TEXT,
    srf_hz             REAL,
    q_at_f_hz          REAL,
    q_value            REAL,
    esr_ohm            REAL,
    validity_hz_lo     REAL,
    validity_hz_hi     REAL,
    measurement_temp_c REAL,
    measurement_bias_v REAL,
    variant_label      TEXT DEFAULT '',
    import_ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kind       ON components(kind);
CREATE INDEX IF NOT EXISTS idx_value_nom  ON components(value_nom);
CREATE INDEX IF NOT EXISTS idx_package    ON components(package);
CREATE INDEX IF NOT EXISTS idx_mc_comp_id ON model_conditions(component_id);
CREATE INDEX IF NOT EXISTS idx_mc_tier    ON model_conditions(model_tier);
"""


def create_db(path: Path) -> sqlite3.Connection:
    """Create or open the library database, applying schema if needed.

    Returns an open connection with WAL mode and foreign keys enabled.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Check if schema_version table exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cur.fetchone() is None:
        # Fresh database — apply full schema
        conn.executescript(_SCHEMA_DDL)
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (CURRENT_SCHEMA_VERSION, now),
        )
        conn.commit()
    else:
        # Check version
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        version = row[0] if row else 0
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {version} is newer than supported "
                f"version {CURRENT_SCHEMA_VERSION}."
            )
        # Migration stubs for future versions would go here

    return conn


def check_wal_mode(conn: sqlite3.Connection) -> bool:
    """Check that the database is in WAL journal mode."""
    cur = conn.execute("PRAGMA journal_mode")
    row = cur.fetchone()
    return row is not None and row[0].lower() == "wal"
