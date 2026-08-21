import numpy as np
import scipy.optimize


def _obj(x):
    print(f"obj({x[0]:.4f})")
    return np.sum(x**2)

def _jac(x):
    print(f"jac({x[0]:.4f})")
    return 2*x

scipy.optimize.minimize(_obj, [0.5, 0.5], method='trust-constr', jac=_jac, bounds=[(0, 1), (0, 1)], options={'maxiter': 2})
