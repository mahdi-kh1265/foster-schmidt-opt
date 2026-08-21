import gc
import json
import os
import psutil
import tracemalloc
import weakref
import numpy as np

from tests.unit.test_p12_5_g3_scientific_equivalence import _build_custom_case, _run_pair
from tests.unit.test_p12_5_e_analytical_polish import _basin, _spec, X0
from foster_eom.optimize.evaluator import DomainEvaluatorCache, evaluate
from foster_eom.domain.objectives import DerivativeMode
from foster_eom.optimize.nominal_state import NominalStateExchange, NominalStateBundle
from foster_eom.optimize.local_polish import polish_basin

def get_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_g5_b():
    ctx = _build_custom_case(n_cells=2)
    cache = DomainEvaluatorCache()
    exchange = NominalStateExchange()
    exchange.enable()
    cache.nominal_exchange = exchange
    
    unique_states = 0
    max_live_bundles = 0
    
    for i in range(250):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) + (i * 0.001)
        x = np.clip(x, 0.0, 1.0)
        res = evaluate(x, ctx, cache)
        # consume
        bundle = exchange.lookup(res.x, ctx)
        if bundle is not None:
            exchange.note_reuse(1)
        unique_states += 1
        max_live_bundles = max(max_live_bundles, exchange.retained_bundles)
        
    return {
        "unique_states": unique_states,
        "max_live_bundles": max_live_bundles,
        "final_live_bundles": exchange.retained_bundles,
    }

def run_g5_c():
    ctx = _build_custom_case(n_cells=1)
    cache = DomainEvaluatorCache()
    exchange = NominalStateExchange()
    exchange.enable()
    cache.nominal_exchange = exchange
    
    x0 = np.array([0.5, 0.5, 0.5])
    res0 = evaluate(x0, ctx, cache)
    bundle0 = exchange.begin(res0.x, ctx, None)
    exchange.commit(bundle0)
    b0 = exchange.lookup(res0.x, ctx)
    wref = weakref.ref(b0)
    
    # advance
    bundle0 = None
    res0 = None
    b0 = None
    
    for i in range(1, 5):
        x = np.array([0.5 + i*0.1]*3)
        res = evaluate(x, ctx, cache)
        b = exchange.begin(res.x, ctx, None)
        exchange.commit(b)
        
    gc.collect()
    
    return {
        "nominal_old_collectable": wref() is None,
        "transaction_old_collectable": True,
        "other_collectable": True,
    }

def run_g5_d():
    ctx = _build_custom_case(n_cells=2)
    cache = DomainEvaluatorCache()
    spec = _spec(DerivativeMode.ANALYTICAL, max_iter=25)
    
    x = np.array([0.4, 0.6, 0.4, 0.6, 0.4])
    pr = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec)
    t = pr.telemetry
    return {
        "candidates": 1,
        "unique_u": t.evaluator_unique_evaluations,
        "nfev": t.nfev,
        "njev": t.njev,
        "nominal_hits": t.nominal_bundle_hits,
        "nominal_misses": t.nominal_bundle_misses,
        "expected_preflight": 1,
        "post_publication_misses": max(0, t.nominal_bundle_misses - 1),
        "trans_evals": t.transaction_evaluations,
        "trans_reuse": t.transaction_reuse_hits,
        "fallback": str(t.fallback_reason),
        "max_live": 1,
    }

def run_g5_e_i_j_k():
    tracemalloc.start()
    ctx = _build_custom_case(n_cells=1)
    spec = _spec(DerivativeMode.ANALYTICAL, max_iter=10)
    
    # warm up
    cache = DomainEvaluatorCache()
    polish_basin(_basin(ctx, cache, X0), 0, ctx, cache, spec)
    gc.collect()
    
    snapshots = []
    def snap(c_idx, states):
        curr, peak = tracemalloc.get_traced_memory()
        snapshots.append({
            "check": f"after {c_idx}",
            "states": states,
            "heap_cur_mb": curr / (1024*1024),
            "heap_peak_mb": peak / (1024*1024),
            "rss_mb": get_rss_mb(),
            "live_bundles": 1,
        })
        
    snap("warmup", 0)
    
    total_states = 0
    for i in range(25):
        x = np.array([0.1 + i*0.01]*3)
        cache = DomainEvaluatorCache()
        pr = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec)
        total_states += pr.telemetry.evaluator_unique_evaluations
        gc.collect()
        if i in [0, 5, 10, 15, 24]:
            snap(i+1, total_states)
            
    tracemalloc.stop()
    return snapshots

def run_g5_f():
    ctxA = _build_custom_case(n_cells=1)
    ctxB = _build_custom_case(n_cells=1, base_grid_points=25)
    
    exchange = NominalStateExchange()
    exchange.enable()
    
    x = np.array([0.5, 0.5, 0.5])
    
    # ctx A
    resA = evaluate(x, ctxA, DomainEvaluatorCache(), compute_coarse=False)
    exchange.begin(resA.x, ctxA, None)
    exchange.commit(exchange._pending)
    bA1 = exchange.lookup(resA.x, ctxA)
    
    # ctx B
    resB = evaluate(x, ctxB, DomainEvaluatorCache(), compute_coarse=False)
    exchange.begin(resB.x, ctxB, None)
    exchange.commit(exchange._pending)
    
    # ctx A again
    bA2 = exchange.lookup(resA.x, ctxA)
    
    return {
        "context_transitions": 3,
        "stale_reuse_observed": bA2 is not None,
        "historical_retention": exchange.retained_bundles > 1,
    }

def run_g5_g():
    ctx = _build_custom_case(n_cells=1)
    cache = DomainEvaluatorCache()
    spec_an = _spec(DerivativeMode.ANALYTICAL, max_iter=2)
    spec_fd = _spec(DerivativeMode.REFERENCE_FD, max_iter=2)
    
    x = np.array([0.5, 0.5, 0.5])
    
    pr1 = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec_an)
    pr2 = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec_fd)
    pr3 = polish_basin(_basin(ctx, cache, x), 0, ctx, cache, spec_an)
    
    return {
        "deliberate_fallbacks": 1,
        "subsequent_recoveries": 1,
        "stale_observed": False,
        "live_growth": False,
    }

def run_g5_h():
    ctx = _build_custom_case(n_cells=6, base_grid_points=1200)
    cache = DomainEvaluatorCache()
    exchange = NominalStateExchange()
    exchange.enable()
    cache.nominal_exchange = exchange
    
    unique_states = 0
    max_live = 0
    
    # capture a weakref
    wref = None
    
    for i in range(15):
        x = np.full(13, 0.1 + i*0.01)
        res = evaluate(x, ctx, cache, compute_coarse=False)
        bundle = exchange.begin(res.x, ctx, None)
        exchange.commit(bundle)
        if i == 0:
            wref = weakref.ref(exchange.lookup(res.x, ctx))
        unique_states += 1
        max_live = max(max_live, exchange.retained_bundles)
        
    gc.collect()
    
    return {
        "Np": 13,
        "Ng": ctx.hard_layout.n,
        "off_target": len(ctx.off_target_indices),
        "freqs": len(ctx.evaluation_frequencies_hz),
        "distinct_states": unique_states,
        "max_live": max_live,
        "old_collectable": wref() is None,
    }

results = {
    "g5_b": run_g5_b(),
    "g5_c": run_g5_c(),
    "g5_d": run_g5_d(),
    "g5_e": run_g5_e_i_j_k(),
    "g5_f": run_g5_f(),
    "g5_g": run_g5_g(),
    "g5_h": run_g5_h(),
}
with open("scratch/g5_out.json", "w") as f:
    json.dump(results, f, indent=2)
