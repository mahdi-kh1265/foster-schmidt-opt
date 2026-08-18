import pytest
from foster_eom.sensitivities.coverage import DerivativeStatus, get_derivative_status
from foster_eom.optimize.variable_map import VariableDescriptor

def test_all_known_variables_have_coverage():
    # Enumerate the known kinds from DecisionVariableMapper's documentation
    known_kinds = ["logk0", "logkinf", "logkm", "fp"]
    
    for kind in known_kinds:
        status = get_derivative_status(kind)
        # Cannot be silently zero/unsupported without explicitly acknowledging it
        assert status in (
            DerivativeStatus.ANALYTICAL_SUPPORTED,
            DerivativeStatus.EXPLICIT_COORDINATE_ONLY,
            DerivativeStatus.UNSUPPORTED
        )
        
        # In P12.5-D, we plan to implement them as EXPLICIT_COORDINATE_ONLY mapping to stamps
        assert status != DerivativeStatus.UNSUPPORTED, f"Variable {kind} lacks derivative coverage"

def test_unknown_variable_is_unsupported():
    status = get_derivative_status("unknown_future_var")
    assert status == DerivativeStatus.UNSUPPORTED
