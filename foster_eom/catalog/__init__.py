"""Component library backend (Prompt 08).

Public API for the SQLite-backed component catalog with content-addressed
file store, query/filter API, and multi-tier model construction.
"""

from foster_eom.catalog.component import (
    ComponentKind,
    FallbackPolicy,
    LibraryComponent,
    ModelCondition,
    ModelOrigin,
    ModelTier,
)
from foster_eom.catalog.file_store import ContentAddressedStore
from foster_eom.catalog.fixture import FixtureSpec, FixtureType
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.catalog.model_bridge import ModelNotAvailableError
from foster_eom.catalog.query import ComponentQuery

__all__ = [
    "ComponentKind",
    "ComponentLibrary",
    "ComponentQuery",
    "ContentAddressedStore",
    "FallbackPolicy",
    "FixtureSpec",
    "FixtureType",
    "LibraryComponent",
    "ModelCondition",
    "ModelNotAvailableError",
    "ModelOrigin",
    "ModelTier",
]
