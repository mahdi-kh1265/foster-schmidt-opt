from enum import Enum, auto

class DerivativeStatus(Enum):
    """Classification of derivative support for a continuous coordinate."""
    ANALYTICAL_SUPPORTED = auto()
    EXPLICIT_COORDINATE_ONLY = auto()
    UNSUPPORTED = auto()

def get_derivative_status(var_type: str) -> DerivativeStatus:
    """Return the derivative coverage status for a given continuous variable type.
    
    Parameters
    ----------
    var_type : str
        The ``var_type`` string emitted by ``DecisionVariableMapper.VariableDescriptor``.
        
    Returns
    -------
    DerivativeStatus
        The support classification.
    """
    if var_type in ("logk0", "logkinf", "logkm", "fp"):
        return DerivativeStatus.EXPLICIT_COORDINATE_ONLY
    
    return DerivativeStatus.UNSUPPORTED
