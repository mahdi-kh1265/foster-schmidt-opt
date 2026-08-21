import sys
import time

import numpy as np
import scipy.linalg

sys.path.insert(0, 'tests/unit')
from test_network_builder import _DummyModel, _make_components, _make_sign_pattern, _make_topology

from foster_eom.circuit.measurements import compute_measurements
from foster_eom.circuit.mna import SolverOptions, assemble_mna, solve_mna, solve_mna_factorized
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.foster.network_builder import build_foster_circuit

topo=_make_topology(b1_cells=3,b2_cells=3,b1_c0=True,b1_linf=True,b2_c0=True,b2_linf=True)
sign=_make_sign_pattern()
built=build_foster_circuit(topo,sign,_make_components(3,1e-10,1e-6),_make_components(3,1e-10,1e-6),_DummyModel())
g=built.graph
src=SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=1.0, z_source_real_ohm=50.0, z_ref_ohm=50.0)
Y,I,nm=assemble_mna(g,src,5e6)
print("n_nodes",len(nm),"n_elems",len(g.elements))
opts=SolverOptions()
N=2000
def t(f,n=N):
    f(); t0=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t0)/n*1e6
print("graph.validate      %8.2f us"%t(lambda: g.validate()))
print("assemble_mna        %8.2f us"%t(lambda: assemble_mna(g,src,5e6)))
print("cond                %8.2f us"%t(lambda: np.linalg.cond(Y)))
print("isfinite Y+I        %8.2f us"%t(lambda: (np.all(np.isfinite(Y)),np.all(np.isfinite(I)))))
print("linalg.solve        %8.2f us"%t(lambda: np.linalg.solve(Y,I)))
print("lu_factor           %8.2f us"%t(lambda: scipy.linalg.lu_factor(Y)))
lu=scipy.linalg.lu_factor(Y)
print("lu_solve            %8.2f us"%t(lambda: scipy.linalg.lu_solve(lu,I)))
print("residual chk        %8.2f us"%t(lambda: float(np.linalg.norm(Y@np.linalg.solve(Y,I)-I))))
print("solve_mna total     %8.2f us"%t(lambda: solve_mna(Y,I,opts)))
print("solve_mna_fact tot  %8.2f us"%t(lambda: solve_mna_factorized(Y,I,opts)))
V,_,diag=solve_mna(Y,I,opts)
print("compute_measurements%8.2f us"%t(lambda: compute_measurements(g,src,V,nm,5e6,diag)))
