"""Top-level project specification (spec §6.1).

Aggregates all sub-specifications into a single frozen Pydantic model
that can be serialized to/from ``*.fseom.yaml``.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field

from foster_eom.domain.component import ComponentPolicy
from foster_eom.domain.constraints import (
    MatchConstraints,
    QBandwidthConstraints,
    RobustnessSpec,
    StressConstraints,
)
from foster_eom.domain.eom import EOMModelSpec
from foster_eom.domain.frequency_plan import FrequencyPlan
from foster_eom.domain.objectives import AnalysisSpec, ExportSpec, OptimizationSpec
from foster_eom.domain.source import SourceSpec
from foster_eom.domain.topology import TopologySearchSpec


class ProjectMeta(BaseModel, frozen=True):
    """Project metadata."""

    name: str = ""
    description: str = ""
    notes: str = ""


class ProjectSpec(BaseModel, frozen=True):
    """Complete project specification (spec §6.1).

    This is the top-level configuration object persisted in ``.fseom.yaml``.

    Attributes
    ----------
    schema_version : str
        Schema version string for migration support.
    project : ProjectMeta
        Name, description, and notes.
    source : SourceSpec
        RF source specification.
    eom : EOMModelSpec
        EOM model definition.
    frequencies : FrequencyPlan
        Target frequencies and sweep configuration.
    matching : MatchConstraints
        Source-side impedance match constraints.
    topology : TopologySearchSpec
        Topology enumeration and pole placement.
    components : ComponentPolicy
        Component value limits and catalog policy.
    q_bandwidth : QBandwidthConstraints
        Q and bandwidth constraints.
    stress : StressConstraints
        Component and source stress limits.
    robustness : RobustnessSpec
        Tolerance analysis configuration.
    optimization : OptimizationSpec
        Optimizer settings.
    analysis : AnalysisSpec
        Post-optimization analysis settings.
    export : ExportSpec
        Export settings.
    created_at : str
        ISO-format creation timestamp.
    modified_at : str
        ISO-format last-modified timestamp.
    """

    schema_version: str = "0.1"
    project: ProjectMeta = Field(default_factory=ProjectMeta)
    source: SourceSpec
    eom: EOMModelSpec
    frequencies: FrequencyPlan
    matching: MatchConstraints = Field(default_factory=MatchConstraints)
    topology: TopologySearchSpec = Field(default_factory=TopologySearchSpec)
    components: ComponentPolicy = Field(default_factory=ComponentPolicy)
    q_bandwidth: QBandwidthConstraints = Field(default_factory=QBandwidthConstraints)
    stress: StressConstraints = Field(default_factory=StressConstraints)
    robustness: RobustnessSpec = Field(default_factory=RobustnessSpec)
    optimization: OptimizationSpec = Field(default_factory=OptimizationSpec)
    analysis: AnalysisSpec = Field(default_factory=AnalysisSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    modified_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )
