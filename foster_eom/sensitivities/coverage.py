from enum import Enum, auto
from dataclasses import dataclass

class DerivativeRoute(Enum):
    """How the derivative of a continuous coordinate propagates to the system."""
    MNA_DERIVED = auto()          # Propagates only through MNA physical components
    COORDINATE_DERIVED = auto()   # Propagates only through direct coordinate constraints
    COMBINED = auto()             # Propagates through both MNA and coordinate constraints
    UNSUPPORTED = auto()

class MNAStampKind(Enum):
    """The types of MNA elements a variable perturbs."""
    CAPACITOR = auto()
    INDUCTOR = auto()

class CoordinateConstraintKind(Enum):
    """The types of coordinate-only constraints a variable feeds into."""
    POLE_SEPARATION = auto()
    # Simple component bounds are not explicitly routed here unless needed

@dataclass(frozen=True)
class DerivativeCoverage:
    route: DerivativeRoute
    mna_stamps: tuple[MNAStampKind, ...] = ()
    coordinate_constraints: tuple[CoordinateConstraintKind, ...] = ()

def get_derivative_coverage(var_type: str) -> DerivativeCoverage:
    """Return the derivative coverage path for a given continuous variable type.
    
    Parameters
    ----------
    var_type : str
        The ``var_type`` string emitted by ``DecisionVariableMapper.VariableDescriptor``.
        
    Returns
    -------
    DerivativeCoverage
        The explicit routing of this variable's derivatives.
    """
    if var_type == "logk0":
        return DerivativeCoverage(DerivativeRoute.MNA_DERIVED, (MNAStampKind.CAPACITOR,))
    elif var_type == "logkinf":
        return DerivativeCoverage(DerivativeRoute.MNA_DERIVED, (MNAStampKind.INDUCTOR,))
    elif var_type == "logkm":
        return DerivativeCoverage(DerivativeRoute.MNA_DERIVED, (MNAStampKind.CAPACITOR, MNAStampKind.INDUCTOR))
    elif var_type == "fp":
        return DerivativeCoverage(
            DerivativeRoute.COMBINED,
            (MNAStampKind.INDUCTOR,),
            (CoordinateConstraintKind.POLE_SEPARATION,)
        )
    
    return DerivativeCoverage(DerivativeRoute.UNSUPPORTED)
