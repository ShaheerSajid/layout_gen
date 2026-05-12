"""
layout_gen.rl.env.place_action — PLACE-phase action helpers.

Materialises a device into a :class:`LayoutState` given a target position
and orientation. Reuses :func:`layout_gen.transistor.draw_transistor`
under the hood so the PLACE action produces geometry consistent with
the rule-based synthesizer.

Two concerns are addressed here:

1. **Orientation math.** gdsfactory rotations apply around the
   component origin ``(0, 0)``; we mirror the same convention so the
   policy can predict standard symmetry flips (R0/MX/MY/R180) without
   knowing the underlying transistor's bbox.

2. **Caching.** ``draw_transistor`` leaks a gdsfactory ``Component``
   into the global PDK on every call. A PPO rollout placing tens of
   thousands of devices would accumulate substantial Component state
   that's never freed. :class:`TransistorCache` calls
   ``draw_transistor`` once per ``(device_type, w_um, l_um, fingers)``
   key, converts the result to a list of layer/coord tuples, and
   reuses that template for subsequent placements.

The cache holds **immutable rect templates** at the device's local
origin. :func:`place_device` clones them, applies the requested
orientation around ``(0, 0)``, translates to ``(target_x, target_y)``,
and adds them to the caller's ``LayoutState``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from layout_gen.synth.geo.state import LayoutState, Rect
from layout_gen.transistor import draw_transistor

from layout_gen.rl.topology.parser import DeviceNode


# ── Orientation ──────────────────────────────────────────────────────────────

ORIENTATIONS: tuple[str, ...] = ("R0", "MX", "MY", "R180")
N_ORIENTATIONS = len(ORIENTATIONS)
_ORIENT_INDEX  = {o: i for i, o in enumerate(ORIENTATIONS)}


def orient_rect(
    x0: float, y0: float, x1: float, y1: float,
    orientation: str,
) -> tuple[float, float, float, float]:
    """Transform a rect's bbox by *orientation* about the origin (0, 0).

    Returns the new ``(x0, y0, x1, y1)`` with ``x0 < x1`` and
    ``y0 < y1``. Translation is *not* applied here — that's done
    separately in :func:`place_device` so orientation and position are
    decoupled.
    """
    if orientation == "R0":
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    elif orientation == "MX":
        # Mirror across X axis: flip Y.
        nx0, ny0, nx1, ny1 = x0, -y1, x1, -y0
    elif orientation == "MY":
        # Mirror across Y axis: flip X.
        nx0, ny0, nx1, ny1 = -x1, y0, -x0, y1
    elif orientation == "R180":
        # 180° rotation: flip both.
        nx0, ny0, nx1, ny1 = -x1, -y1, -x0, -y0
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")
    # Canonicalise (the formulas above already produce x0<x1 / y0<y1
    # because the input rect is canonical; still cheap to be safe).
    if nx0 > nx1: nx0, nx1 = nx1, nx0
    if ny0 > ny1: ny0, ny1 = ny1, ny0
    return nx0, ny0, nx1, ny1


# ── Rect template ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _RectTemplate:
    """Cacheable, hashable shape — rid is assigned at placement time."""
    layer:      str
    x0:         float
    y0:         float
    x1:         float
    y1:         float
    net:        str = ""
    shape_type: str = ""


# ── Cache ────────────────────────────────────────────────────────────────────

class TransistorCache:
    """One-shot cache from ``(device_type, w, l, fingers)`` → template rects.

    Constructed once per env (per PDK); reused across episodes. Devices
    with the same sizing share a single cache entry, so per-cell PLACE
    rollouts effectively pay one ``draw_transistor`` call per *unique*
    transistor variant rather than per placement.

    Parameters
    ----------
    rules :
        PDK rules — handed to :func:`draw_transistor` and to
        :meth:`LayoutState.from_component` for layer-name resolution.
    """

    def __init__(self, rules) -> None:
        self._rules = rules
        self._cache: dict[tuple, list[_RectTemplate]] = {}

    def get(
        self,
        device_type: str,
        w_um:        float,
        l_um:        float,
        fingers:     int = 0,
    ) -> list[_RectTemplate]:
        key = (device_type, round(w_um, 6), round(l_um, 6), int(fingers))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        kwargs = {}
        if fingers:
            kwargs["n_fingers"] = fingers
        comp = draw_transistor(w_um, l_um, device_type, self._rules, **kwargs)

        # Convert the Component to plain rect tuples. We pass through
        # LayoutState.from_component to reuse the layer-mapping logic
        # already validated by the synth pipeline.
        tmp = LayoutState.from_component(comp, self._rules)
        templates = [
            _RectTemplate(
                layer=r.layer,
                x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1,
                net=r.net, shape_type=r.shape_type,
            )
            for r in tmp
        ]
        self._cache[key] = templates
        return templates


# ── Placement ────────────────────────────────────────────────────────────────

def place_device(
    state:       LayoutState,
    device:      DeviceNode,
    *,
    x_um:        float,
    y_um:        float,
    orientation: str,
    cache:       TransistorCache,
    net_lookup:  dict[str, str] | None = None,
) -> list[Rect]:
    """Materialise *device* into *state* at ``(x_um, y_um)`` with *orientation*.

    Parameters
    ----------
    state :
        Target :class:`LayoutState`. Rects are appended via
        :meth:`LayoutState.add`, which assigns fresh rids.
    device :
        :class:`DeviceNode` from the cell's topology graph. Provides
        ``device_type``, ``w_um``, ``l_um``, ``fingers``.
    x_um, y_um :
        Origin offset applied after orientation. The device's local
        ``(0, 0)`` lands at this point in cell coordinates.
    orientation :
        One of :data:`ORIENTATIONS`.
    cache :
        :class:`TransistorCache` (constructed once per env).
    net_lookup :
        Optional mapping ``terminal_name → net_name`` for tagging the
        new rects with their logical net. Currently passed-through to
        ``shape_type`` annotations in a Phase 4-part-2 stub manner —
        proper terminal-to-rect attribution lives in the routing phase.

    Returns
    -------
    list[Rect] :
        The newly-added rects (in insertion order). Useful for the
        env to record what was placed this step.
    """
    if orientation not in _ORIENT_INDEX:
        raise ValueError(f"Unknown orientation: {orientation!r}")

    templates = cache.get(
        device.device_type, device.w_um, device.l_um, device.fingers,
    )

    new_rects: list[Rect] = []
    for t in templates:
        ox0, oy0, ox1, oy1 = orient_rect(t.x0, t.y0, t.x1, t.y1, orientation)
        r = state.add(
            layer=t.layer,
            x0=ox0 + x_um, y0=oy0 + y_um,
            x1=ox1 + x_um, y1=oy1 + y_um,
            net=t.net,
            shape_type=t.shape_type,
        )
        new_rects.append(r)
    return new_rects


def orientation_index(orientation: str) -> int:
    return _ORIENT_INDEX[orientation]


def orientation_from_index(idx: int) -> str:
    return ORIENTATIONS[idx]


__all__ = [
    "ORIENTATIONS", "N_ORIENTATIONS",
    "orient_rect", "orientation_index", "orientation_from_index",
    "TransistorCache",
    "place_device",
]
