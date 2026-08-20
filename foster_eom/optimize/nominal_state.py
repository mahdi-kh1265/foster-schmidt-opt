"""Current-iterate nominal MNA state exchange (P12.5-F).

P12.5-E measured that an analytical polish iterate performs *two* full nominal
frequency sweeps at the same ``u``: one inside the production ``evaluate()`` and
one inside :class:`~foster_eom.sensitivities.transaction.DerivativeTransaction`.
This module removes the second one by letting ``evaluate()`` publish the
per-frequency nominal state it already computed, and letting the transaction
consume it.

What is shared
--------------
Only the *nominal linear-algebra state*: the assembled ``Y`` and ``b``, the node
map, the nominal solution ``V`` and the validity outcome ``solve_mna`` reached
for those very arrays.  Nothing derived is shared — the transaction still builds
its own derivative stamps, observables and solution records with unchanged
mathematics, and still performs its own LU factorization and back-substitutions.

Why that is exact
-----------------
For a hit, ``x``, the :class:`EvaluationContext` object, the frequency grid and
the circuit graph are all *identical* (not merely close), so the assembly is a
pure function that would reproduce the same ``Y`` and ``b`` bit-for-bit, and
``cond(Y)`` and the nonfinite pre-screen are pure functions of those arrays.
Only those provably redundant stages are skipped; the factorization, the
solution-finiteness gate and the residual gate are still executed on the reused
system by :func:`~foster_eom.circuit.mna.refactorize_shared_mna`.

Bounding
--------
At most **one committed bundle** (the current iterate) plus at most one
in-progress bundle is retained.  There is no historical cache: publishing a new
iterate drops the previous one, and a failed evaluation publishes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from foster_eom.circuit.mna import SolvedMNASystem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from foster_eom.circuit.graph import CircuitGraph
    from foster_eom.optimize.evaluator import EvaluationContext


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NominalStateIdentity:
    """Everything that must match exactly before a nominal state may be reused.

    ``context`` is compared by object identity: the context is frozen and owns
    the domain/topology, the source spec, the EOM and component models, the
    compiled constraint layouts and the frequency grid, so a shared context
    object is the strongest available statement that the nominal problem is the
    same one.  The remaining fields are re-checked explicitly so that a context
    that was rebuilt (even to equal values) is treated as a miss rather than
    silently trusted.
    """

    x_key: tuple[float, ...]
    context: Any
    domain_id: str
    frequencies_hz: tuple[float, ...]
    target_indices: tuple[int, ...]
    off_target_indices: tuple[int, ...]

    @classmethod
    def build(cls, x_key: tuple[float, ...], context: EvaluationContext) -> NominalStateIdentity:
        return cls(
            x_key=x_key,
            context=context,
            domain_id=context.domain.domain_id,
            frequencies_hz=context.evaluation_frequencies_hz,
            target_indices=context.target_indices,
            off_target_indices=context.off_target_indices,
        )

    def matches(self, other: NominalStateIdentity) -> bool:
        """Exact-identity match.  No approximate ``x`` comparison is performed."""
        return (
            self.context is other.context
            and self.x_key == other.x_key
            and self.domain_id == other.domain_id
            and self.frequencies_hz == other.frequencies_hz
            and self.target_indices == other.target_indices
            and self.off_target_indices == other.off_target_indices
        )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass
class NominalStateBundle:
    """Derivative-ready nominal state for exactly one ``x``."""

    identity: NominalStateIdentity
    graph: CircuitGraph
    states: dict[int, SolvedMNASystem] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def get(self, freq_index: int) -> SolvedMNASystem | None:
        return self.states.get(freq_index)


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------


class NominalStateExchange:
    """Publishes the current iterate's nominal state from ``evaluate()``.

    Disabled by default (``enabled is False``): with capture off, ``evaluate()``
    takes exactly the frozen P05 code path and nothing is retained.  The
    analytical derivative provider enables it for the duration of one polish;
    ``REFERENCE_FD`` never does, so the FD reference path is untouched.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._committed: NominalStateBundle | None = None
        self._pending: NominalStateBundle | None = None

        # Telemetry.  Meanings are deliberately narrow:
        self.bundles_published: int = 0  # evaluate() iterates that published state
        self.bundles_dropped: int = 0  # iterates whose evaluation failed -> nothing published
        self.states_captured: int = 0  # per-frequency nominal states published
        self.states_reused: int = 0  # per-frequency nominal states consumed
        self.lookup_hits: int = 0  # transaction builds that found the current-x bundle
        self.lookup_misses: int = 0  # transaction builds that had to sweep themselves
        self.peak_retained_states: int = 0  # high-water mark of retained per-freq states

    # -- lifecycle ----------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        """Stop capturing and release all retained heavy state."""
        self.enabled = False
        self._committed = None
        self._pending = None

    # -- producer side (evaluate) -------------------------------------------

    def begin(
        self,
        x_key: tuple[float, ...],
        context: EvaluationContext,
        graph: CircuitGraph,
    ) -> NominalStateBundle | None:
        """Open a bundle for ``x_key``.  Returns ``None`` when capture is off."""
        if not self.enabled:
            return None
        self._pending = NominalStateBundle(
            identity=NominalStateIdentity.build(x_key, context),
            graph=graph,
        )
        return self._pending

    def record(self, bundle: NominalStateBundle | None, freq_index: int, system: object) -> None:
        """Record one successfully solved frequency into an open bundle."""
        if bundle is None or bundle is not self._pending:
            return
        if not isinstance(system, SolvedMNASystem):
            return
        bundle.states[freq_index] = system
        self.states_captured += 1
        retained = self.retained_states
        if retained > self.peak_retained_states:
            self.peak_retained_states = retained

    def commit(self, bundle: NominalStateBundle | None) -> None:
        """Publish the open bundle as *the* current iterate, dropping the previous."""
        if bundle is None or bundle is not self._pending:
            return
        self._committed = bundle
        self._pending = None
        self.bundles_published += 1

    def abandon(self, bundle: NominalStateBundle | None) -> None:
        """Discard the open bundle (evaluation failed) and keep nothing from it."""
        if bundle is None:
            return
        if bundle is self._pending:
            self._pending = None
        if self._committed is bundle:  # pragma: no cover - defensive
            self._committed = None
        self.bundles_dropped += 1

    def settle(self, ok: bool) -> None:
        """Close whatever bundle is open: publish it if ``ok``, else drop it."""
        bundle = self._pending
        if bundle is None:
            return
        if ok:
            self.commit(bundle)
        else:
            self.abandon(bundle)

    # -- consumer side (DerivativeTransaction) ------------------------------

    def lookup(
        self, x_key: tuple[float, ...], context: EvaluationContext
    ) -> NominalStateBundle | None:
        """Return the current-iterate bundle iff every identity matches exactly."""
        bundle = self._committed
        if bundle is None:
            self.lookup_misses += 1
            return None
        if not bundle.identity.matches(NominalStateIdentity.build(x_key, context)):
            self.lookup_misses += 1
            return None
        self.lookup_hits += 1
        return bundle

    def note_reuse(self, n: int = 1) -> None:
        self.states_reused += n

    # -- introspection ------------------------------------------------------

    @property
    def retained_bundles(self) -> int:
        """Number of bundles currently holding heavy state (structurally <= 2)."""
        return int(self._committed is not None) + int(self._pending is not None)

    @property
    def retained_states(self) -> int:
        """Number of per-frequency nominal states currently retained."""
        total = 0
        if self._committed is not None:
            total += self._committed.n_states
        if self._pending is not None:
            total += self._pending.n_states
        return total

    def counters(self) -> dict[str, int]:
        """Flat, unambiguous counter snapshot."""
        return {
            "nominal_bundles_published": self.bundles_published,
            "nominal_bundles_dropped": self.bundles_dropped,
            "nominal_states_captured": self.states_captured,
            "nominal_states_reused": self.states_reused,
            "nominal_lookup_hits": self.lookup_hits,
            "nominal_lookup_misses": self.lookup_misses,
            "nominal_retained_bundles": self.retained_bundles,
            "nominal_retained_states": self.retained_states,
            "nominal_peak_retained_states": self.peak_retained_states,
        }
