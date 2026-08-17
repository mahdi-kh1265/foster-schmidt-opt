"""Combination search over per-slot NeighborhoodEntry lists (Prompt 09).

``generate_combos()`` — returns all combinations to evaluate.
Uses exhaustive enumeration when product <= threshold, beam search otherwise.
"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foster_eom.realization.spec import NeighborhoodEntry, RealizationSpec


# ---------------------------------------------------------------------------
# Combo = ordered slot assignment
# ---------------------------------------------------------------------------

# A Combo is a list of (element_id, NeighborhoodEntry) pairs in slot order.
Combo = list[tuple[str, "NeighborhoodEntry"]]


def generate_combos(
    neighborhoods: dict[str, list[NeighborhoodEntry]],
    spec: RealizationSpec,
) -> tuple[list[Combo], bool, bool]:
    """Generate combinations to evaluate.

    Parameters
    ----------
    neighborhoods : dict[str, list[NeighborhoodEntry]]
        Per-slot candidate lists (already sorted by log_ratio, truncated).
    spec : RealizationSpec
        Controls exhaustive_threshold, beam_width, random_seed.

    Returns
    -------
    combos : list[Combo]
        Ordered list of combos to evaluate (best-first heuristic ordering).
    search_exhaustive : bool
        True if all combinations were enumerated.
    search_truncated : bool
        True if beam was used (incomplete search).
    """
    # Slot ordering is deterministic — use dict insertion order (Python 3.7+)
    slot_ids = list(neighborhoods.keys())
    per_slot = [neighborhoods[sid] for sid in slot_ids]

    # Total combinations
    counts = [len(entries) for entries in per_slot]
    if any(c == 0 for c in counts):
        # Caller should have caught empty slots; return empty
        return [], False, False

    total = 1
    for c in counts:
        total *= c

    if total <= spec.exhaustive_threshold:
        combos = _exhaustive(slot_ids, per_slot)
        return combos, True, False
    else:
        combos = _beam(slot_ids, per_slot, spec)
        return combos, False, True


# ---------------------------------------------------------------------------
# Exhaustive enumeration
# ---------------------------------------------------------------------------


def _exhaustive(
    slot_ids: list[str],
    per_slot: list[list[NeighborhoodEntry]],
) -> list[Combo]:
    """Enumerate all combinations; sorted by sum of log_ratios (best first)."""
    combos: list[Combo] = []
    for entries in itertools.product(*per_slot):
        combo: Combo = list(zip(slot_ids, entries, strict=True))
        combos.append(combo)
    # Sort by total log_ratio (closest overall)
    combos.sort(key=lambda c: sum(e.log_ratio for _, e in c))
    return combos


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def _beam(
    slot_ids: list[str],
    per_slot: list[list[NeighborhoodEntry]],
    spec: RealizationSpec,
) -> list[Combo]:
    """Greedy-first + scored beam search.

    State: list of partial combos, each a list of (element_id, entry) pairs.
    Score: sum of log_ratios so far (lower = closer to continuous target).
    Prune to beam_width after each slot expansion.
    Diversity pass: ensure >=1 combo per distinct first-slot part.
    """
    rng = random.Random(spec.random_seed)
    B = spec.beam_width

    # Start: expand first slot
    if not per_slot:
        return []

    beam: list[tuple[float, Combo]] = []
    for entry in per_slot[0]:
        partial: Combo = [(slot_ids[0], entry)]
        beam.append((entry.log_ratio, partial))

    for slot_i in range(1, len(slot_ids)):
        next_beam: list[tuple[float, Combo]] = []
        for score, partial in beam:
            for entry in per_slot[slot_i]:
                new_score = score + entry.log_ratio
                new_partial = [*partial, (slot_ids[slot_i], entry)]
                next_beam.append((new_score, new_partial))
        # Sort by score; add random tiebreak
        next_beam.sort(key=lambda t: (t[0], rng.random()))
        beam = next_beam[:B]

    # Full combos from beam (already B best)
    full_combos: list[Combo] = [combo for _, combo in beam]

    # Diversity pass: ensure at least one combo per distinct first-slot part
    first_parts_seen: set[str] = {combo[0][1].component_id for combo in full_combos}
    for entry in per_slot[0]:
        if entry.component_id not in first_parts_seen:
            # Greedily complete with closest remaining
            div_combo: Combo = [(slot_ids[0], entry)]
            for _slot_i, (sid, slot_entries) in enumerate(
                zip(slot_ids[1:], per_slot[1:], strict=True), start=1
            ):
                div_combo.append((sid, slot_entries[0]))  # closest in each slot
            full_combos.append(div_combo)
            first_parts_seen.add(entry.component_id)

    # Sort by total log_ratio
    full_combos.sort(key=lambda c: sum(e.log_ratio for _, e in c))
    return full_combos
