"""
layout_gen.rl.env.action_space — composite action for the layout RL env.

Phase 1 implements the **repair** action set: edits to existing polygons
that mirror the 6 primitives in :mod:`layout_gen.repair.perturb`. The
action is encoded as a fixed-size :class:`gymnasium.spaces.MultiDiscrete`
so that any downstream RL algorithm (PPO, DQN, MaskablePPO, …) can
consume it without custom plumbing.

Action layout (each dim is one categorical choice)
--------------------------------------------------
========  =====  =========================================================
Index     Size   Meaning
========  =====  =========================================================
0 kind    6      shift_edge | shrink_rect | grow_rect | translate |
                 delete_rect | nudge_offgrid
1 target  T_MAX  Index into the polygon list (masked to live rids).
2 edge    4      left | right | bottom | top (used by shift_edge).
3 sign_x  2      -, +  (used by translate/nudge for dx).
4 sign_y  2      -, +  (used by translate/nudge for dy; for shift_edge
                 sign_y==1 means outward, ==0 means inward).
5 mag     M_BINS Magnitude bin index. Magnitudes are log-spaced over
                 [DELTA_MIN, DELTA_MAX] µm (positive). For
                 ``nudge_offgrid`` the magnitude is rescaled by
                 :data:`OFFGRID_SCALE` so it lands in the sub-grid range.
========  =====  =========================================================

Place / route action heads (Phase 4) extend this MultiDiscrete with
additional dims; existing repair-only checkpoints remain compatible by
masking the new dims off.

The full bundle (kind, target, edge, sign_x, sign_y, mag) is decoded by
:meth:`ActionSpace.decode` into a :class:`~layout_gen.repair.perturb.PerturbAction`
that the env can apply directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gymnasium import spaces

from layout_gen.repair.perturb import PerturbAction
from layout_gen.synth.geo.state import LayoutState


# ── Vocabulary (mirrors repair.features for cross-compat) ────────────────────

ACTION_KINDS: tuple[str, ...] = (
    "shift_edge", "shrink_rect", "grow_rect",
    "translate", "delete_rect", "nudge_offgrid",
)
N_KINDS  = len(ACTION_KINDS)

EDGE_NAMES: tuple[str, ...] = ("left", "right", "bottom", "top")
N_EDGES  = len(EDGE_NAMES)

# Default per-dim sizes. The env wires its own TARGET_CAP via constructor.
DEFAULT_TARGET_CAP = 256
DEFAULT_MAG_BINS   = 16

# Magnitude range — log-spaced, in µm. Matches the perturb library's
# delta_min_um / delta_max_um defaults so the BC-trained policy starts
# in the same regime as the trajectory dataset.
DELTA_MIN_UM = 0.005
DELTA_MAX_UM = 0.10

# Sub-grid nudge rescale: the nudge_offgrid magnitudes are ~1 order of
# magnitude smaller than the regular edits.
OFFGRID_SCALE = 0.1

# Kinds that move *all four* edges of a rect — they ignore ``edge``.
_OMNIDIRECTIONAL = frozenset({"shrink_rect", "grow_rect"})

# Kinds that don't read the magnitude bin.
_MAGNITUDE_FREE  = frozenset({"delete_rect"})


# ── Magnitude binning ────────────────────────────────────────────────────────

def magnitude_bins(n_bins: int = DEFAULT_MAG_BINS,
                   *, lo: float = DELTA_MIN_UM,
                   hi: float = DELTA_MAX_UM) -> np.ndarray:
    """Return a (n_bins,) array of log-spaced magnitudes in µm."""
    return np.exp(np.linspace(math.log(lo), math.log(hi), n_bins)).astype(np.float32)


# ── Composite action ─────────────────────────────────────────────────────────

@dataclass
class EnvAction:
    """Decoded form of a raw MultiDiscrete sample.

    Held for diagnostics; the env applies the equivalent
    :class:`PerturbAction` via :func:`apply_env_action`.
    """
    kind:   str
    rid:    int           # rectangle ID in the live LayoutState (-1 if invalid)
    edge:   str           # one of EDGE_NAMES
    sign_x: int           # -1 or +1
    sign_y: int           # -1 or +1
    mag:    float         # µm, positive


# ── Action space ─────────────────────────────────────────────────────────────

class ActionSpace:
    """Wraps a :class:`gymnasium.spaces.MultiDiscrete` with decoding helpers.

    Parameters
    ----------
    target_cap :
        Maximum number of polygons the policy can address. Layouts with
        more polygons are truncated (a warning is emitted by the env);
        layouts with fewer pad the head with masked positions.
    mag_bins :
        Number of magnitude bins. More bins → finer magnitude resolution
        at the cost of a larger action space.
    """

    def __init__(self,
                 *,
                 target_cap: int = DEFAULT_TARGET_CAP,
                 mag_bins:   int = DEFAULT_MAG_BINS) -> None:
        self.target_cap = target_cap
        self.mag_bins   = mag_bins
        self._mag_table = magnitude_bins(mag_bins)
        self.nvec       = (N_KINDS, target_cap, N_EDGES, 2, 2, mag_bins)
        self.gym_space  = spaces.MultiDiscrete(self.nvec)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def decode(self,
               raw: Sequence[int],
               idx_to_rid: dict[int, int]) -> EnvAction:
        """Decode a raw MultiDiscrete sample into an :class:`EnvAction`.

        *idx_to_rid* maps polygon-tensor index → live rectangle ID. Indices
        not present in the map produce ``rid == -1`` (the env should treat
        these as no-ops via masking; they only occur if the policy ignores
        the action mask).
        """
        kind_i, target_i, edge_i, sx_i, sy_i, mag_i = (int(x) for x in raw)
        kind = ACTION_KINDS[kind_i]
        rid  = idx_to_rid.get(target_i, -1)
        edge = EDGE_NAMES[edge_i]
        sign_x = -1 if sx_i == 0 else 1
        sign_y = -1 if sy_i == 0 else 1
        mag  = float(self._mag_table[mag_i])
        if kind == "nudge_offgrid":
            mag *= OFFGRID_SCALE
        return EnvAction(kind=kind, rid=rid, edge=edge,
                         sign_x=sign_x, sign_y=sign_y, mag=mag)

    def to_perturb(self, env_action: EnvAction) -> PerturbAction | None:
        """Translate to a :class:`PerturbAction` ready for ``perturb.apply``.

        Returns ``None`` when the action is structurally invalid (target
        rid not present). Callers should treat ``None`` as a no-op.
        """
        if env_action.rid < 0:
            return None
        kind = env_action.kind
        if kind == "shift_edge":
            # sign_y picks inward (-) vs outward (+). The perturb shift_edge
            # convention: positive delta moves the edge outward.
            delta = env_action.mag * env_action.sign_y
            return PerturbAction("shift_edge", target=env_action.rid,
                                 params={"side": env_action.edge, "delta": delta})
        if kind == "shrink_rect":
            return PerturbAction("shrink_rect", target=env_action.rid,
                                 params={"delta": env_action.mag})
        if kind == "grow_rect":
            return PerturbAction("grow_rect", target=env_action.rid,
                                 params={"delta": env_action.mag})
        if kind in ("translate", "nudge_offgrid"):
            dx = env_action.mag * env_action.sign_x
            dy = env_action.mag * env_action.sign_y
            return PerturbAction(kind, target=env_action.rid,
                                 params={"dx": dx, "dy": dy})
        if kind == "delete_rect":
            # Snapshot is filled by the env after applying (so an undo is
            # possible if needed). Phase 1 doesn't undo, so leave it None.
            return PerturbAction("delete_rect", target=env_action.rid)
        return None


# ── Masking ──────────────────────────────────────────────────────────────────

def action_mask_for(
    state:        LayoutState,
    rid_to_idx:   dict[int, int],
    *,
    target_cap:   int = DEFAULT_TARGET_CAP,
    mag_bins:     int = DEFAULT_MAG_BINS,
    forbid_kinds: frozenset[str] = frozenset(),
) -> np.ndarray:
    """Produce the per-dim action mask consumed by MaskablePPO.

    Returns
    -------
    mask : np.ndarray of bool, shape (sum(nvec),)
        Concatenated per-dim masks in MultiDiscrete order
        (kind, target, edge, sign_x, sign_y, mag). MaskablePPO splits
        this back per-component.
    """
    parts: list[np.ndarray] = []

    # Kind: forbid explicitly disallowed kinds; in Phase 1 we forbid
    # nothing by default. The trainer can pass forbid_kinds={"delete_rect"}
    # for tiny cells where deletion is unrecoverable.
    kind_m = np.ones(N_KINDS, dtype=bool)
    for fk in forbid_kinds:
        if fk in ACTION_KINDS:
            kind_m[ACTION_KINDS.index(fk)] = False
    parts.append(kind_m)

    # Target: only indices 0..len(rid_to_idx)-1 (and < target_cap) are live.
    n_live = min(len(rid_to_idx), target_cap)
    tgt_m = np.zeros(target_cap, dtype=bool)
    tgt_m[:n_live] = True
    if n_live == 0:
        # Pathological — empty layout. Allow index 0 so the action sample
        # is well-formed; the decoder will treat it as a no-op.
        tgt_m[0] = True
    parts.append(tgt_m)

    # Edge / sign / mag have no dynamic constraint in Phase 1.
    parts.append(np.ones(N_EDGES, dtype=bool))
    parts.append(np.ones(2,        dtype=bool))
    parts.append(np.ones(2,        dtype=bool))
    parts.append(np.ones(mag_bins, dtype=bool))

    return np.concatenate(parts)


__all__ = [
    "ACTION_KINDS", "N_KINDS",
    "EDGE_NAMES", "N_EDGES",
    "DEFAULT_TARGET_CAP", "DEFAULT_MAG_BINS",
    "DELTA_MIN_UM", "DELTA_MAX_UM", "OFFGRID_SCALE",
    "magnitude_bins",
    "EnvAction", "ActionSpace",
    "action_mask_for",
]
