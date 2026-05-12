"""
layout_gen.rl.env.route_action — ROUTE-phase action helpers.

Adds a single metal rectangle on a chosen layer at a chosen position
and size. The Phase 4-part-2c routing model is intentionally simple:

  * Each ROUTE action emits one rect (no multi-segment paths yet).
  * The policy chooses the net being extended (so the rect carries a
    net annotation in the LayoutState — useful for LVS-style checks).
  * Layers come from a small vocabulary of common interconnect layers
    (li1, met1, met2, met3 by default). PDK-specific layer mapping
    happens later in the GDS-write step via :class:`PDKRules`.

Why one rect per action instead of a path?
  * Keeps the action space MultiDiscrete with finite per-dim sizes
    (PPO + MaskablePPO compatible).
  * Lets PPO learn longer paths as **sequences** of single-rect
    actions. Multi-segment paths become an emergent behaviour rather
    than a baked-in primitive — and can be inspected per step.

Size + position vocabularies
----------------------------
Sizes (``w``, ``h``) are log-spaced over [0.1 µm, 1.0 µm] to span both
short stubs and long horizontal/vertical bus segments at the typical
bitcell scale. Positions reuse the PLACE-phase x/y bins so the policy
shares one spatial coordinate system across phases.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from layout_gen.synth.geo.state import LayoutState, Rect


# ── Layer + size vocabularies ────────────────────────────────────────────────

ROUTE_LAYERS: tuple[str, ...] = ("li1", "met1", "met2", "met3")
N_ROUTE_LAYERS = len(ROUTE_LAYERS)

DEFAULT_SIZE_BINS  = 8
ROUTE_SIZE_MIN_UM  = 0.10
ROUTE_SIZE_MAX_UM  = 1.00


def size_bins(n_bins: int = DEFAULT_SIZE_BINS,
              *, lo: float = ROUTE_SIZE_MIN_UM,
              hi: float = ROUTE_SIZE_MAX_UM) -> np.ndarray:
    """Log-spaced rectangle sizes (w, h) in µm. Same spacing for both
    axes so the policy can build either thin or square rects from one
    bin index per axis."""
    return np.exp(np.linspace(math.log(lo), math.log(hi), n_bins)).astype(np.float32)


def layer_index(layer: str) -> int:
    if layer not in ROUTE_LAYERS:
        raise ValueError(f"Unknown route layer {layer!r}; "
                         f"expected one of {ROUTE_LAYERS}")
    return ROUTE_LAYERS.index(layer)


def layer_from_index(idx: int) -> str:
    if not 0 <= idx < N_ROUTE_LAYERS:
        raise ValueError(f"Bad route layer index {idx}")
    return ROUTE_LAYERS[idx]


# ── Segment add ──────────────────────────────────────────────────────────────

def add_route_segment(
    state:    LayoutState,
    *,
    layer:    str,
    x_um:     float,
    y_um:     float,
    w_um:     float,
    h_um:     float,
    net_name: str = "",
) -> Rect:
    """Append one metal rectangle to *state* and return it.

    The rect spans ``(x_um, y_um) → (x_um + w_um, y_um + h_um)`` and
    is tagged with ``net_name`` for downstream LVS / connectivity
    introspection. ``shape_type`` is set to ``"wire"`` so the existing
    repair-phase featurizer can distinguish routing rects from device
    geometry.
    """
    return state.add(
        layer=layer,
        x0=x_um, y0=y_um, x1=x_um + w_um, y1=y_um + h_um,
        net=net_name,
        shape_type="wire",
    )


__all__ = [
    "ROUTE_LAYERS", "N_ROUTE_LAYERS",
    "DEFAULT_SIZE_BINS", "ROUTE_SIZE_MIN_UM", "ROUTE_SIZE_MAX_UM",
    "size_bins", "layer_index", "layer_from_index",
    "add_route_segment",
]
