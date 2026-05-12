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
from layout_gen.transistor import draw_transistor, transistor_geom

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


def orient_point(x: float, y: float, orientation: str) -> tuple[float, float]:
    """Transform a single point by *orientation* about the origin.

    Used to map a transistor port's local coordinates into the
    post-orient frame; :func:`place_device` then translates to global.
    """
    if orientation == "R0":   return (x, y)
    if orientation == "MX":   return (x, -y)
    if orientation == "MY":   return (-x, y)
    if orientation == "R180": return (-x, -y)
    raise ValueError(f"Unknown orientation: {orientation!r}")


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


@dataclass(frozen=True)
class _PortTemplate:
    """Per-terminal access point in the device's local coordinate frame.

    ``layer`` is the logical layer the port lives on (li1 for source/drain,
    poly for the gate). Connectivity scoring uses these positions to
    decide whether a routing rect actually touches a terminal.
    """
    name:  str        # "G" / "S" / "D"
    x:     float
    y:     float
    layer: str


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
        # Each entry is (rect_templates, port_templates).
        self._cache: dict[tuple, tuple[list[_RectTemplate], list[_PortTemplate]]] = {}

    def get(
        self,
        device_type: str,
        w_um:        float,
        l_um:        float,
        fingers:     int = 0,
    ) -> list[_RectTemplate]:
        return self.get_full(device_type, w_um, l_um, fingers)[0]

    def get_full(
        self,
        device_type: str,
        w_um:        float,
        l_um:        float,
        fingers:     int = 0,
    ) -> tuple[list[_RectTemplate], list[_PortTemplate]]:
        """Returns (rect templates, port templates) for the device variant."""
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
        rect_templates = [
            _RectTemplate(
                layer=r.layer,
                x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1,
                net=r.net, shape_type=r.shape_type,
            )
            for r in tmp
        ]

        # Port positions are derivable from TransistorGeom; we don't try to
        # extract them from the gdsfactory Component (whose Port API has
        # changed shape across releases).
        geom = transistor_geom(w_um, l_um, device_type, self._rules)
        endcap = self._rules.poly["endcap_over_diff_um"]
        diff_y_mid = endcap + geom.w_finger_um / 2
        # First-finger gate centre.
        gate_x = geom.sd_length_um + geom.l_um / 2
        # Source = leftmost S/D; drain = rightmost.
        source_x = geom.sd_length_um / 2
        drain_x  = geom.n_fingers * (geom.sd_length_um + geom.l_um) \
                   + geom.sd_length_um / 2
        port_templates = [
            _PortTemplate(name="G", x=gate_x,   y=geom.total_y_um, layer="poly"),
            _PortTemplate(name="S", x=source_x, y=diff_y_mid,      layer="li1"),
            _PortTemplate(name="D", x=drain_x,  y=diff_y_mid,      layer="li1"),
        ]

        self._cache[key] = (rect_templates, port_templates)
        return rect_templates, port_templates


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

    Returns the newly-added rects. Use :func:`place_device_full` if you
    also need the per-terminal global positions (needed by the
    connectivity reward in :mod:`layout_gen.rl.env.connectivity`).
    """
    return place_device_full(
        state, device, x_um=x_um, y_um=y_um,
        orientation=orientation, cache=cache, net_lookup=net_lookup,
    )[0]


def place_device_full(
    state:       LayoutState,
    device:      DeviceNode,
    *,
    x_um:        float,
    y_um:        float,
    orientation: str,
    cache:       TransistorCache,
    net_lookup:  dict[str, str] | None = None,
) -> tuple[list[Rect], dict[str, tuple[float, float, str]]]:
    """Like :func:`place_device` but also returns per-terminal positions.

    Returns
    -------
    rects :
        Newly-added rectangles (insertion order).
    ports :
        ``{terminal_name: (x_um, y_um, layer)}`` in cell-global
        coordinates after orientation + translation.
    """
    if orientation not in _ORIENT_INDEX:
        raise ValueError(f"Unknown orientation: {orientation!r}")

    rect_templates, port_templates = cache.get_full(
        device.device_type, device.w_um, device.l_um, device.fingers,
    )

    new_rects: list[Rect] = []
    for t in rect_templates:
        ox0, oy0, ox1, oy1 = orient_rect(t.x0, t.y0, t.x1, t.y1, orientation)
        r = state.add(
            layer=t.layer,
            x0=ox0 + x_um, y0=oy0 + y_um,
            x1=ox1 + x_um, y1=oy1 + y_um,
            net=t.net,
            shape_type=t.shape_type,
        )
        new_rects.append(r)

    ports: dict[str, tuple[float, float, str]] = {}
    for p in port_templates:
        px, py = orient_point(p.x, p.y, orientation)
        ports[p.name] = (px + x_um, py + y_um, p.layer)
    return new_rects, ports


def orientation_index(orientation: str) -> int:
    return _ORIENT_INDEX[orientation]


def orientation_from_index(idx: int) -> str:
    return ORIENTATIONS[idx]


__all__ = [
    "ORIENTATIONS", "N_ORIENTATIONS",
    "orient_rect", "orient_point",
    "orientation_index", "orientation_from_index",
    "TransistorCache",
    "place_device", "place_device_full",
]
