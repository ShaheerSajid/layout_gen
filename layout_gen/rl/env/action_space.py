"""
layout_gen.rl.env.action_space — composite action for the layout RL env.

Phase 1 implements the **repair** action set: edits to existing polygons
that mirror the 6 primitives in :mod:`layout_gen.repair.perturb`. The
action is encoded as a fixed-size :class:`gymnasium.spaces.MultiDiscrete`
so that any downstream RL algorithm (PPO, DQN, MaskablePPO, …) can
consume it without custom plumbing.

Phase 4 part 2 adds **place**: when an :class:`ActionSpace` is built
with ``enable_place=True``, the kind vocabulary grows by one
(``place_device``) and four PLACE-specific dims are appended to the
MultiDiscrete:

  device  — pointer-style choice over the topology graph's devices
  x_bin   — discretised X position over the cell width
  y_bin   — discretised Y position over the cell height
  orient  — one of R0 / MX / MY / R180

REPAIR and PLACE share the same observation + trunk; only the head
slots that apply to the current phase carry real signal — the rest
are masked off by :func:`action_mask_for`.

Action layout (each dim is one categorical choice)
--------------------------------------------------
========  =====  =========================================================
Index     Size   Meaning
========  =====  =========================================================
0 kind    K      shift_edge | shrink_rect | grow_rect | translate |
                 delete_rect | nudge_offgrid  [+ place_device when
                 enable_place=True].  K is 6 or 7.
1 target  T_MAX  Index into the polygon list (masked to live rids).
2 edge    4      left | right | bottom | top (used by shift_edge).
3 sign_x  2      -, +  (used by translate/nudge for dx).
4 sign_y  2      -, +  (used by translate/nudge for dy; for shift_edge
                 sign_y==1 means outward, ==0 means inward).
5 mag     M_BINS Magnitude bin index. Magnitudes are log-spaced over
                 [DELTA_MIN, DELTA_MAX] µm (positive). For
                 ``nudge_offgrid`` the magnitude is rescaled by
                 :data:`OFFGRID_SCALE` so it lands in the sub-grid range.

PLACE-only dims (only present when enable_place=True)
6 device  D_CAP  Index into the topology graph's device list.
7 x_bin   X_B    Bin in [0, cell_width_um].
8 y_bin   Y_B    Bin in [0, cell_height_um].
9 orient  4      R0 | MX | MY | R180.
========  =====  =========================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gymnasium import spaces

from layout_gen.repair.perturb import PerturbAction
from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.place_action import (
    N_ORIENTATIONS, ORIENTATIONS, orientation_from_index,
)


# ── Vocabulary (REPAIR — Phase 1) ────────────────────────────────────────────

REPAIR_KINDS: tuple[str, ...] = (
    "shift_edge", "shrink_rect", "grow_rect",
    "translate", "delete_rect", "nudge_offgrid",
)
N_REPAIR_KINDS = len(REPAIR_KINDS)

# Kept under the old name for back-compat with imports.
ACTION_KINDS: tuple[str, ...] = REPAIR_KINDS
N_KINDS = N_REPAIR_KINDS

# Vocabulary (PLACE — Phase 4)
PLACE_KINDS: tuple[str, ...] = ("place_device",)
N_PLACE_KINDS = len(PLACE_KINDS)

EDGE_NAMES: tuple[str, ...] = ("left", "right", "bottom", "top")
N_EDGES  = len(EDGE_NAMES)

# Default per-dim sizes.
DEFAULT_TARGET_CAP    = 256
DEFAULT_MAG_BINS      = 16
DEFAULT_DEVICE_CAP    = 32
DEFAULT_POSITION_BINS = 16
DEFAULT_CELL_WIDTH_UM  = 4.0
DEFAULT_CELL_HEIGHT_UM = 4.0

# Magnitude range — log-spaced, in µm.
DELTA_MIN_UM = 0.005
DELTA_MAX_UM = 0.10
OFFGRID_SCALE = 0.1


def combined_kinds(enable_place: bool) -> tuple[str, ...]:
    return REPAIR_KINDS + PLACE_KINDS if enable_place else REPAIR_KINDS


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

    For REPAIR kinds the ``rid`` / ``edge`` / ``sign_*`` / ``mag`` fields
    carry the meaningful payload. For ``place_device`` the
    ``device_idx`` / ``x_um`` / ``y_um`` / ``orientation`` fields do.
    """
    kind:        str
    # REPAIR payload
    rid:         int   = -1
    edge:        str   = "left"
    sign_x:      int   = 1
    sign_y:      int   = 1
    mag:         float = 0.0
    # PLACE payload
    device_idx:  int   = -1
    x_um:        float = 0.0
    y_um:        float = 0.0
    orientation: str   = "R0"

    @property
    def is_place(self) -> bool:
        return self.kind in PLACE_KINDS


# ── Action space ─────────────────────────────────────────────────────────────

class ActionSpace:
    """Wraps a :class:`gymnasium.spaces.MultiDiscrete` with decoding helpers.

    Parameters
    ----------
    target_cap :
        Maximum number of polygons the policy can address.
    mag_bins :
        Number of magnitude bins.
    enable_place :
        When True, extends the action space with the PLACE kind and four
        PLACE-specific dims (device / x_bin / y_bin / orient).
    device_cap :
        Maximum number of placeable devices. Indices ≥ this are masked.
    x_bins, y_bins :
        Spatial-position discretisation grid for PLACE.
    cell_width_um, cell_height_um :
        Cell bbox used to map PLACE bins to µm coordinates. The bin
        ``i`` maps to ``(i + 0.5) / x_bins * cell_width_um`` (centre of
        the bin) — so the policy can never pick a coord exactly at 0
        or W (which is helpful for avoiding rail abutment by accident).
    """

    def __init__(self,
                 *,
                 target_cap:     int   = DEFAULT_TARGET_CAP,
                 mag_bins:       int   = DEFAULT_MAG_BINS,
                 enable_place:   bool  = False,
                 device_cap:     int   = DEFAULT_DEVICE_CAP,
                 x_bins:         int   = DEFAULT_POSITION_BINS,
                 y_bins:         int   = DEFAULT_POSITION_BINS,
                 cell_width_um:  float = DEFAULT_CELL_WIDTH_UM,
                 cell_height_um: float = DEFAULT_CELL_HEIGHT_UM,
                 ) -> None:
        self.target_cap     = target_cap
        self.mag_bins       = mag_bins
        self.enable_place   = enable_place
        self.device_cap     = device_cap
        self.x_bins         = x_bins
        self.y_bins         = y_bins
        self.cell_width_um  = cell_width_um
        self.cell_height_um = cell_height_um

        self._mag_table = magnitude_bins(mag_bins)
        self._kinds     = combined_kinds(enable_place)

        nvec = [len(self._kinds), target_cap, N_EDGES, 2, 2, mag_bins]
        if enable_place:
            nvec += [device_cap, x_bins, y_bins, N_ORIENTATIONS]
        self.nvec = tuple(nvec)
        self.gym_space = spaces.MultiDiscrete(self.nvec)

    @property
    def kinds(self) -> tuple[str, ...]:
        return self._kinds

    # ── Decoding ─────────────────────────────────────────────────────────────

    def decode(self,
               raw: Sequence[int],
               idx_to_rid: dict[int, int]) -> EnvAction:
        """Decode a raw MultiDiscrete sample into an :class:`EnvAction`."""
        raw = [int(x) for x in raw]
        kind_i, target_i, edge_i, sx_i, sy_i, mag_i = raw[:6]
        kind = self._kinds[kind_i] if 0 <= kind_i < len(self._kinds) else self._kinds[0]
        rid  = idx_to_rid.get(target_i, -1)
        edge = EDGE_NAMES[edge_i]
        sign_x = -1 if sx_i == 0 else 1
        sign_y = -1 if sy_i == 0 else 1
        mag  = float(self._mag_table[mag_i])
        if kind == "nudge_offgrid":
            mag *= OFFGRID_SCALE

        action = EnvAction(
            kind=kind, rid=rid, edge=edge,
            sign_x=sign_x, sign_y=sign_y, mag=mag,
        )

        if self.enable_place and len(raw) >= 10:
            dev_i, xb_i, yb_i, or_i = raw[6:10]
            action.device_idx = int(dev_i)
            action.x_um = self._bin_to_coord(xb_i, self.x_bins, self.cell_width_um)
            action.y_um = self._bin_to_coord(yb_i, self.y_bins, self.cell_height_um)
            action.orientation = orientation_from_index(
                or_i if 0 <= or_i < N_ORIENTATIONS else 0
            )

        return action

    @staticmethod
    def _bin_to_coord(bin_idx: int, n_bins: int, span_um: float) -> float:
        bin_idx = max(0, min(int(bin_idx), n_bins - 1))
        return (bin_idx + 0.5) / n_bins * span_um

    def to_perturb(self, env_action: EnvAction) -> PerturbAction | None:
        """Translate a REPAIR action to a :class:`PerturbAction`. Returns
        None for PLACE actions and for repair actions whose target rid
        is not present."""
        if env_action.is_place:
            return None
        if env_action.rid < 0:
            return None
        kind = env_action.kind
        if kind == "shift_edge":
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
    enable_place: bool = False,
    phase:        str = "repair",
    device_cap:   int = DEFAULT_DEVICE_CAP,
    n_devices:    int = 0,
    placed_mask:  np.ndarray | None = None,
    x_bins:       int = DEFAULT_POSITION_BINS,
    y_bins:       int = DEFAULT_POSITION_BINS,
) -> np.ndarray:
    """Produce the per-dim action mask consumed by MaskablePPO.

    Parameters
    ----------
    phase :
        ``"repair"`` (default) or ``"place"``. When the env is in
        ``"place"`` phase, kind is constrained to PLACE-only and the
        REPAIR-only kind slots are off.
    n_devices :
        Number of real devices in the topology graph (≥ this index in
        the device dim is always masked).
    placed_mask :
        Optional boolean array of length ``device_cap`` where True
        means *already placed* — those device indices are masked off
        during PLACE phase so the policy can't place the same device
        twice.
    """
    parts: list[np.ndarray] = []

    n_kinds_total = N_REPAIR_KINDS + (N_PLACE_KINDS if enable_place else 0)
    kind_m = np.ones(n_kinds_total, dtype=bool)

    if enable_place and phase == "place":
        # PLACE phase: only PLACE kinds allowed.
        for i in range(N_REPAIR_KINDS):
            kind_m[i] = False
    else:
        # REPAIR phase (or non-place env): PLACE kinds masked off.
        if enable_place:
            for i in range(N_REPAIR_KINDS, n_kinds_total):
                kind_m[i] = False
    for fk in forbid_kinds:
        if fk in REPAIR_KINDS:
            kind_m[REPAIR_KINDS.index(fk)] = False
    parts.append(kind_m)

    # Target dim: live polygons (only matters in REPAIR phase, but
    # mask must be valid in PLACE too — keep at least slot 0 selectable
    # so the action vector is well-formed).
    n_live = min(len(rid_to_idx), target_cap)
    tgt_m = np.zeros(target_cap, dtype=bool)
    if n_live > 0:
        tgt_m[:n_live] = True
    else:
        tgt_m[0] = True
    parts.append(tgt_m)

    # Edge / sign / mag: always all-True (specific kinds ignore them).
    parts.append(np.ones(N_EDGES, dtype=bool))
    parts.append(np.ones(2,        dtype=bool))
    parts.append(np.ones(2,        dtype=bool))
    parts.append(np.ones(mag_bins, dtype=bool))

    if enable_place:
        # Device dim: in PLACE phase, valid = (i < n_devices) and not
        # already placed. In REPAIR phase the head is unused, but the
        # mask must still pick at least one slot so the distribution
        # is well-defined — we just allow slot 0.
        dev_m = np.zeros(device_cap, dtype=bool)
        if phase == "place" and n_devices > 0:
            limit = min(n_devices, device_cap)
            dev_m[:limit] = True
            if placed_mask is not None:
                placed = np.asarray(placed_mask, dtype=bool)[:device_cap]
                # mask out the already-placed devices
                dev_m[: len(placed)] &= ~placed
            if not dev_m.any():
                dev_m[0] = True
        else:
            dev_m[0] = True
        parts.append(dev_m)
        parts.append(np.ones(x_bins, dtype=bool))
        parts.append(np.ones(y_bins, dtype=bool))
        parts.append(np.ones(N_ORIENTATIONS, dtype=bool))

    return np.concatenate(parts)


__all__ = [
    "REPAIR_KINDS", "PLACE_KINDS", "ACTION_KINDS",
    "N_REPAIR_KINDS", "N_PLACE_KINDS", "N_KINDS",
    "EDGE_NAMES", "N_EDGES",
    "DEFAULT_TARGET_CAP", "DEFAULT_MAG_BINS",
    "DEFAULT_DEVICE_CAP", "DEFAULT_POSITION_BINS",
    "DEFAULT_CELL_WIDTH_UM", "DEFAULT_CELL_HEIGHT_UM",
    "DELTA_MIN_UM", "DELTA_MAX_UM", "OFFGRID_SCALE",
    "combined_kinds", "magnitude_bins",
    "EnvAction", "ActionSpace",
    "action_mask_for",
]
