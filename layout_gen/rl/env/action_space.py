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
from layout_gen.rl.env.route_action import (
    DEFAULT_SIZE_BINS, N_ROUTE_LAYERS, ROUTE_LAYERS,
    layer_from_index, size_bins,
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

# Vocabulary (PLACE — Phase 4 part 2a)
PLACE_KINDS: tuple[str, ...] = ("place_device",)
N_PLACE_KINDS = len(PLACE_KINDS)

# Vocabulary (ROUTE — Phase 4 part 2c)
ROUTE_KINDS: tuple[str, ...] = ("route_segment",)
N_ROUTE_KINDS = len(ROUTE_KINDS)

EDGE_NAMES: tuple[str, ...] = ("left", "right", "bottom", "top")
N_EDGES  = len(EDGE_NAMES)

# Default per-dim sizes.
DEFAULT_TARGET_CAP    = 256
DEFAULT_MAG_BINS      = 16
DEFAULT_DEVICE_CAP    = 32
DEFAULT_NET_CAP       = 32
DEFAULT_POSITION_BINS = 16
DEFAULT_CELL_WIDTH_UM  = 4.0
DEFAULT_CELL_HEIGHT_UM = 4.0

# Magnitude range — log-spaced, in µm.
DELTA_MIN_UM = 0.005
DELTA_MAX_UM = 0.10
OFFGRID_SCALE = 0.1


def combined_kinds(enable_place: bool,
                    enable_route: bool = False) -> tuple[str, ...]:
    """Concatenate the active per-phase kind vocabularies.

    Order matters: REPAIR slots are first (so Phase 1–3 indices remain
    stable), PLACE next, then ROUTE. New phases append, never insert.
    """
    out = REPAIR_KINDS
    if enable_place:
        out = out + PLACE_KINDS
    if enable_route:
        out = out + ROUTE_KINDS
    return out


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

    Only the field block matching the action's :attr:`kind` carries a
    meaningful payload; the others stay at their dataclass defaults.
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
    # ROUTE payload
    net_idx:     int   = -1
    route_layer: str   = "met1"
    route_x_um:  float = 0.0
    route_y_um:  float = 0.0
    route_w_um:  float = 0.0
    route_h_um:  float = 0.0

    @property
    def is_place(self) -> bool:
        return self.kind in PLACE_KINDS

    @property
    def is_route(self) -> bool:
        return self.kind in ROUTE_KINDS


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
    poly_pitch_um :
        Optional. When supplied, the PLACE x_um coordinate is snapped
        to the nearest poly-pitch grid line. This makes gate alignment
        a hard constraint of the action space rather than a learned
        soft one (DTCO-style, MDPI 2025). Poly tracks are typically
        non-negotiable for digital layouts; analog floorplans may
        still want it off.
    metal_pitch_um_per_layer :
        Optional ``{layer_name: pitch_um}`` mapping (e.g.
        ``{"met1": 0.34, "met2": 0.46, "li1": 0.34}``). When the
        decoded ROUTE action targets a layer in this dict, the
        ``route_x_um`` and/or ``route_y_um`` are snapped to that
        pitch — which axes get snapped is governed by
        ``metal_direction_per_layer``.
    metal_direction_per_layer :
        Optional ``{layer_name: "horizontal" | "vertical" | ""}``
        mapping. Drives which axis a layer's pitch quantises:

        * ``"horizontal"`` — layer runs in horizontal tracks; **only
          the y coordinate** (cross-axis track index) snaps to pitch.
          x stays free, since routes extend along the horizontal.
        * ``"vertical"`` — only x snaps.
        * ``""`` or layer absent from the dict — both x and y snap
          (no directional preference; appropriate for li1).

        Together with ``metal_pitch_um_per_layer`` this gives the
        policy a maze-router-style track grid that respects the
        PDK's directional routing convention.
    """

    def __init__(self,
                 *,
                 target_cap:        int   = DEFAULT_TARGET_CAP,
                 mag_bins:          int   = DEFAULT_MAG_BINS,
                 enable_place:      bool  = False,
                 device_cap:        int   = DEFAULT_DEVICE_CAP,
                 x_bins:            int   = DEFAULT_POSITION_BINS,
                 y_bins:            int   = DEFAULT_POSITION_BINS,
                 cell_width_um:     float = DEFAULT_CELL_WIDTH_UM,
                 cell_height_um:    float = DEFAULT_CELL_HEIGHT_UM,
                 poly_pitch_um:     float | None = None,
                 metal_pitch_um_per_layer: dict[str, float] | None = None,
                 metal_direction_per_layer: dict[str, str] | None = None,
                 # ── ROUTE knobs ─────────────────────────────────────
                 enable_route:      bool  = False,
                 net_cap:           int   = DEFAULT_NET_CAP,
                 route_x_bins:      int   = DEFAULT_POSITION_BINS,
                 route_y_bins:      int   = DEFAULT_POSITION_BINS,
                 route_w_bins:      int   = DEFAULT_SIZE_BINS,
                 route_h_bins:      int   = DEFAULT_SIZE_BINS,
                 ) -> None:
        self.target_cap     = target_cap
        self.mag_bins       = mag_bins
        self.enable_place   = enable_place
        self.device_cap     = device_cap
        self.x_bins         = x_bins
        self.y_bins         = y_bins
        self.cell_width_um  = cell_width_um
        self.cell_height_um = cell_height_um
        self.poly_pitch_um  = (
            float(poly_pitch_um) if poly_pitch_um and poly_pitch_um > 0
            else None
        )
        self.metal_pitch_um_per_layer = {
            str(k): float(v)
            for k, v in (metal_pitch_um_per_layer or {}).items()
            if v and float(v) > 0
        } or None
        self.metal_direction_per_layer = {
            str(k): str(v).lower()
            for k, v in (metal_direction_per_layer or {}).items()
            if v
        } or None

        self.enable_route   = enable_route
        self.net_cap        = net_cap
        self.route_x_bins   = route_x_bins
        self.route_y_bins   = route_y_bins
        self.route_w_bins   = route_w_bins
        self.route_h_bins   = route_h_bins

        self._mag_table  = magnitude_bins(mag_bins)
        self._size_table = size_bins(max(route_w_bins, route_h_bins))
        self._kinds      = combined_kinds(enable_place, enable_route)

        nvec = [len(self._kinds), target_cap, N_EDGES, 2, 2, mag_bins]
        if enable_place:
            nvec += [device_cap, x_bins, y_bins, N_ORIENTATIONS]
        if enable_route:
            nvec += [net_cap, N_ROUTE_LAYERS,
                     route_x_bins, route_y_bins,
                     route_w_bins, route_h_bins]
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

        cursor = 6
        if self.enable_place and len(raw) >= cursor + 4:
            dev_i, xb_i, yb_i, or_i = raw[cursor:cursor + 4]
            cursor += 4
            action.device_idx = int(dev_i)
            action.x_um = self._bin_to_coord(xb_i, self.x_bins, self.cell_width_um)
            action.y_um = self._bin_to_coord(yb_i, self.y_bins, self.cell_height_um)
            if self.poly_pitch_um is not None:
                action.x_um = self._snap_to_pitch(
                    action.x_um, self.cell_width_um, self.poly_pitch_um,
                )
            action.orientation = orientation_from_index(
                or_i if 0 <= or_i < N_ORIENTATIONS else 0
            )

        if self.enable_route and len(raw) >= cursor + 6:
            net_i, lyr_i, rxb_i, ryb_i, wb_i, hb_i = raw[cursor:cursor + 6]
            action.net_idx     = int(net_i)
            action.route_layer = layer_from_index(
                lyr_i if 0 <= lyr_i < N_ROUTE_LAYERS else 0
            )
            action.route_x_um  = self._bin_to_coord(
                rxb_i, self.route_x_bins, self.cell_width_um,
            )
            action.route_y_um  = self._bin_to_coord(
                ryb_i, self.route_y_bins, self.cell_height_um,
            )
            action.route_w_um  = self._size_bin(wb_i, self.route_w_bins)
            action.route_h_um  = self._size_bin(hb_i, self.route_h_bins)
            if self.metal_pitch_um_per_layer is not None:
                pitch = self.metal_pitch_um_per_layer.get(action.route_layer)
                if pitch is not None:
                    direction = (
                        self.metal_direction_per_layer.get(action.route_layer, "")
                        if self.metal_direction_per_layer is not None else ""
                    )
                    # Snap only the cross-axis when the layer has a
                    # preferred direction; snap both axes when it
                    # doesn't (e.g. li1). "horizontal" → snap y;
                    # "vertical" → snap x.
                    snap_x = direction != "horizontal"
                    snap_y = direction != "vertical"
                    if snap_x:
                        action.route_x_um = self._snap_to_pitch(
                            action.route_x_um, self.cell_width_um, pitch,
                        )
                    if snap_y:
                        action.route_y_um = self._snap_to_pitch(
                            action.route_y_um, self.cell_height_um, pitch,
                        )

        return action

    @staticmethod
    def _snap_to_pitch(raw_um: float, span_um: float, pitch_um: float) -> float:
        """Snap ``raw_um`` to the nearest pitch-grid line, where grid
        lines are ``k * pitch + pitch/2`` for k = 0, 1, …. The half-
        pitch offset keeps the policy from ever landing on the cell
        boundary (x=0 or x=span); clamped into ``[pitch/2, span -
        pitch/2]`` so the snap never falls outside the cell."""
        if pitch_um <= 0:
            return float(raw_um)
        half = pitch_um / 2.0
        snapped = round((raw_um - half) / pitch_um) * pitch_um + half
        if snapped < half:
            snapped = half
        if span_um > pitch_um and snapped > span_um - half:
            snapped = span_um - half
        return float(snapped)

    def _size_bin(self, idx: int, n_bins: int) -> float:
        idx = max(0, min(int(idx), n_bins - 1))
        # _size_table is shared across w/h; both pull from the log-spaced
        # range [ROUTE_SIZE_MIN_UM, ROUTE_SIZE_MAX_UM].
        return float(self._size_table[idx])

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
    # ── Row-aware y_bin masking (audit fix B) ─────────────────────────
    # When ``strict_row_alignment`` is True and the *unplaced* device
    # set is unanimous on a single type (all-nmos or all-pmos), restrict
    # y_bins to that type's half-row so the policy can't waste an
    # attempt on a y the env will reject. We do not mask when types
    # are mixed because the mask is per-dim — we'd need to know which
    # device the policy will pick first.
    unplaced_device_types: list[str] | None = None,
    strict_row_alignment:  bool = False,
    # ── ROUTE-phase ────────────────────────────────────────────────────
    enable_route: bool = False,
    net_cap:      int = DEFAULT_NET_CAP,
    n_nets:       int = 0,
    route_x_bins: int = DEFAULT_POSITION_BINS,
    route_y_bins: int = DEFAULT_POSITION_BINS,
    route_w_bins: int = DEFAULT_SIZE_BINS,
    route_h_bins: int = DEFAULT_SIZE_BINS,
) -> np.ndarray:
    """Produce the per-dim action mask consumed by MaskablePPO.

    Parameters
    ----------
    phase :
        ``"repair"`` (default), ``"place"``, or ``"route"``. The kind
        dim is constrained to the kinds for the active phase; the
        per-phase blocks (PLACE, ROUTE) keep their dims masked-down
        when their phase is inactive.
    n_devices :
        Number of real devices in the topology graph (≥ this index in
        the device dim is always masked).
    placed_mask :
        Optional boolean array of length ``device_cap`` where True
        means *already placed* — masked off during PLACE phase.
    n_nets :
        Number of real nets in the topology graph (≥ this index in
        the net dim is always masked).
    """
    parts: list[np.ndarray] = []

    n_kinds_total = (
        N_REPAIR_KINDS
        + (N_PLACE_KINDS if enable_place else 0)
        + (N_ROUTE_KINDS if enable_route else 0)
    )
    kind_m = np.zeros(n_kinds_total, dtype=bool)
    place_offset = N_REPAIR_KINDS
    route_offset = place_offset + (N_PLACE_KINDS if enable_place else 0)

    if phase == "place":
        if enable_place:
            kind_m[place_offset:place_offset + N_PLACE_KINDS] = True
    elif phase == "route":
        if enable_route:
            kind_m[route_offset:route_offset + N_ROUTE_KINDS] = True
    else:
        # REPAIR phase (or no per-phase machine): only REPAIR kinds.
        kind_m[:N_REPAIR_KINDS] = True

    for fk in forbid_kinds:
        if fk in REPAIR_KINDS:
            kind_m[REPAIR_KINDS.index(fk)] = False

    if not kind_m.any():
        # Defensive: never emit a fully-False kind mask (the
        # categorical distribution would be undefined). The env will
        # treat this as a no-op anyway since the action's kind won't
        # match the active phase.
        kind_m[0] = True
    parts.append(kind_m)

    # Target dim: live polygons (REPAIR phase needs them; other phases
    # just keep slot 0 selectable so the vector is well-formed).
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
        dev_m = np.zeros(device_cap, dtype=bool)
        if phase == "place" and n_devices > 0:
            limit = min(n_devices, device_cap)
            dev_m[:limit] = True
            if placed_mask is not None:
                placed = np.asarray(placed_mask, dtype=bool)[:device_cap]
                dev_m[: len(placed)] &= ~placed
            if not dev_m.any():
                dev_m[0] = True
        else:
            dev_m[0] = True
        parts.append(dev_m)
        parts.append(np.ones(x_bins, dtype=bool))

        # y_bin row mask under strict_row_alignment when the unplaced
        # set is type-unanimous. Bottom half = NMOS, top half = PMOS
        # (matches LayoutEnv._apply_place's strict-row guard).
        y_m = np.ones(y_bins, dtype=bool)
        if (
            strict_row_alignment
            and phase == "place"
            and unplaced_device_types
        ):
            types = {t.lower() for t in unplaced_device_types}
            if types == {"nmos"}:
                y_m[y_bins // 2:] = False    # NMOS only → bottom half
            elif types == {"pmos"}:
                y_m[: y_bins // 2] = False   # PMOS only → top half
            if not y_m.any():
                y_m[0] = True                # never emit all-False
        parts.append(y_m)
        parts.append(np.ones(N_ORIENTATIONS, dtype=bool))

    if enable_route:
        # Net dim: in ROUTE phase, valid = i < n_nets. Otherwise just
        # slot 0 is selectable so the categorical is well-defined.
        net_m = np.zeros(net_cap, dtype=bool)
        if phase == "route" and n_nets > 0:
            limit = min(n_nets, net_cap)
            net_m[:limit] = True
        else:
            net_m[0] = True
        parts.append(net_m)
        parts.append(np.ones(N_ROUTE_LAYERS, dtype=bool))
        parts.append(np.ones(route_x_bins,   dtype=bool))
        parts.append(np.ones(route_y_bins,   dtype=bool))
        parts.append(np.ones(route_w_bins,   dtype=bool))
        parts.append(np.ones(route_h_bins,   dtype=bool))

    return np.concatenate(parts)


# ── Pitch derivation from PDK rules ──────────────────────────────────────────

def derive_poly_pitch_um(rules) -> float | None:
    """Best-effort poly pitch in µm derived from a :class:`PDKRules` object.

    Preferred source: ``rules.poly["pitch_um"]`` if the YAML defines it
    (e.g. sky130 CPP = 0.46 µm). Otherwise falls back to the minimum
    contacted poly pitch ``width_min_um + spacing_min_um`` which is the
    DRC lower bound. Returns None when neither is available.
    """
    poly = getattr(rules, "poly", None) or {}
    p = poly.get("pitch_um")
    if p:
        return float(p)
    w = poly.get("width_min_um")
    s = poly.get("spacing_min_um")
    if w and s:
        return float(w) + float(s)
    return None


def derive_metal_pitches_um(rules) -> dict[str, float]:
    """Per-layer metal pitch in µm derived from a :class:`PDKRules` object.

    Computes ``width_min_um + spacing_min_um`` for every metal/li layer
    that defines both fields. This is the *minimum* track pitch — real
    standard cells often use a wider, library-defined track pitch, but
    the DRC-minimum is a safe lower bound when no library is loaded.
    Empty dict when the PDK exposes no usable metal rule sections.
    """
    pitches: dict[str, float] = {}
    candidates = ["li1", "met1", "met2", "met3", "met4", "met5"]
    for name in candidates:
        section = getattr(rules, name, None) or {}
        w = section.get("width_min_um")
        s = section.get("spacing_min_um")
        if w and s:
            pitches[name] = float(w) + float(s)
    return pitches


def derive_metal_directions(rules) -> dict[str, str]:
    """Per-layer preferred routing direction from a :class:`PDKRules`.

    Reads ``rules.preferred_direction`` and normalises values to the
    set ``{"horizontal", "vertical", ""}``. Layers with an empty value
    (e.g. li1) keep the empty string — :class:`ActionSpace` treats
    those as "snap both axes" rather than "no snap"."""
    raw = getattr(rules, "preferred_direction", None) or {}
    out: dict[str, str] = {}
    for layer, direction in raw.items():
        v = str(direction).lower().strip()
        if v.startswith("h"):
            out[str(layer)] = "horizontal"
        elif v.startswith("v"):
            out[str(layer)] = "vertical"
        else:
            out[str(layer)] = ""
    return out


__all__ = [
    "REPAIR_KINDS", "PLACE_KINDS", "ROUTE_KINDS", "ACTION_KINDS",
    "N_REPAIR_KINDS", "N_PLACE_KINDS", "N_ROUTE_KINDS", "N_KINDS",
    "EDGE_NAMES", "N_EDGES",
    "DEFAULT_TARGET_CAP", "DEFAULT_MAG_BINS",
    "DEFAULT_DEVICE_CAP", "DEFAULT_NET_CAP", "DEFAULT_POSITION_BINS",
    "DEFAULT_CELL_WIDTH_UM", "DEFAULT_CELL_HEIGHT_UM",
    "DELTA_MIN_UM", "DELTA_MAX_UM", "OFFGRID_SCALE",
    "combined_kinds", "magnitude_bins",
    "derive_poly_pitch_um", "derive_metal_pitches_um",
    "derive_metal_directions",
    "EnvAction", "ActionSpace",
    "action_mask_for",
]
