"""
layout_gen.synth.placer — resolve floorplan constraints → placed devices.

Given a ``CellTemplate`` and ``PDKRules``, the placer:

1. Computes ``TransistorGeom`` for every device from the supplied params.
2. Evaluates named constraints (e.g. ``inter_cell_gap``).
3. Resolves symbolic ``y_offset_expr`` and ``x_spec`` into global (x, y)
   floats, respecting device dependencies.
4. Returns a ``{name: PlacedDevice}`` map.

The global coordinate system follows ``draw_transistor``:
- Origin (0, 0) = lower-left of the NMOS poly bounding box.
- X axis = channel length direction.
- Y axis = channel width direction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from layout_gen.pdk        import PDKRules
from layout_gen.transistor import draw_transistor, transistor_geom, TransistorGeom
from layout_gen.cells.standard import _gate_x, _sd_x, _diff_y
from layout_gen.synth.loader      import CellTemplate, DeviceSpec
from layout_gen.synth.constraints import eval_expr, resolve_named_constraints


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_W: dict[str, float] = {"nmos": 0.52, "pmos": 0.52}
_DEFAULT_L: float = 0.15


# ── PlacedDevice ──────────────────────────────────────────────────────────────

@dataclass
class PlacedDevice:
    """A device placed at a specific origin in global cell coordinates.

    Attributes
    ----------
    name :
        Device instance name (e.g. ``"N"``, ``"P"``).
    spec :
        Original ``DeviceSpec`` from the template.
    geom :
        Computed transistor geometry.
    x, y :
        Global origin offset.  Pass to ``ref.move((x, y))`` when adding
        to a ``gf.Component``.
    component :
        The drawn ``gf.Component``.  Set by the synthesizer after calling
        ``draw_transistor``; ``None`` until then.
    """
    name:      str
    spec:      DeviceSpec
    geom:      TransistorGeom
    x:         float
    y:         float
    component: Any = None   # gf.Component (avoid importing gf at module load)


# ── Terminal geometry ─────────────────────────────────────────────────────────

@dataclass
class TerminalGeom:
    """Global bounding box of a device terminal (G, D, or S)."""
    dev_name: str
    terminal: str    # "G", "D", or "S"
    x0: float
    x1: float
    y0: float
    y1: float
    layer: str       # logical layer name: "poly" or "li1"


def resolve_terminal(
    ref:    str,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> TerminalGeom:
    """Resolve a ``"DeviceName.Terminal"`` ref to global geometry.

    Parameters
    ----------
    ref :
        Terminal reference string, e.g. ``"N.G"``, ``"P.D"``.
    placed :
        Map of device name → ``PlacedDevice``.
    rules :
        PDK rules (needed for ``_diff_y``).

    Returns
    -------
    TerminalGeom
    """
    parts = ref.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid terminal reference {ref!r} (expected 'Dev.Term')")
    dev_name, term = parts
    if dev_name not in placed:
        raise KeyError(f"Device {dev_name!r} not found in placed devices")

    dev  = placed[dev_name]
    geom = dev.geom

    if term == "G":
        lx0, lx1 = _gate_x(0, geom)
        return TerminalGeom(
            dev_name, "G",
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=dev.y,        y1=dev.y + geom.total_y_um,
            layer="poly",
        )
    elif term == "D":
        j = geom.n_fingers   # rightmost S/D = drain for finger 0
        lx0, lx1 = _sd_x(j, geom)
        ly0, ly1 = _diff_y(geom, rules)
        return TerminalGeom(
            dev_name, "D",
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=ly0 + dev.y, y1=ly1 + dev.y,
            layer="li1",
        )
    elif term == "S":
        lx0, lx1 = _sd_x(0, geom)
        ly0, ly1 = _diff_y(geom, rules)
        return TerminalGeom(
            dev_name, "S",
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=ly0 + dev.y, y1=ly1 + dev.y,
            layer="li1",
        )
    else:
        raise ValueError(f"Unknown terminal {term!r} in reference {ref!r}")


# ── Global geometry helpers ────────────────────────────────────────────────────

def global_gate_x(dev: PlacedDevice, finger: int = 0) -> tuple[float, float]:
    """Global (x0, x1) of the *finger*-th gate poly finger."""
    lx0, lx1 = _gate_x(finger, dev.geom)
    return lx0 + dev.x, lx1 + dev.x


def global_sd_x(dev: PlacedDevice, j: int) -> tuple[float, float]:
    """Global (x0, x1) of the *j*-th source/drain region."""
    lx0, lx1 = _sd_x(j, dev.geom)
    return lx0 + dev.x, lx1 + dev.x


def global_diff_y(dev: PlacedDevice, rules: PDKRules) -> tuple[float, float]:
    """Global (y0, y1) of the diffusion region."""
    ly0, ly1 = _diff_y(dev.geom, rules)
    return ly0 + dev.y, ly1 + dev.y


def global_poly_top(dev: PlacedDevice) -> float:
    """Global Y of the poly top edge."""
    return dev.y + dev.geom.total_y_um


def global_poly_bottom(dev: PlacedDevice) -> float:
    """Global Y of the poly bottom edge (= device Y origin)."""
    return dev.y


# ── Placer ─────────────────────────────────────────────────────────────────────

class Placer:
    """Resolves device placements from a :class:`~layout_gen.synth.loader.CellTemplate`.

    Parameters
    ----------
    rules :
        PDK rules.
    params :
        Device sizing.  Keys: ``"w_<DevName>"`` (µm), ``"l"`` (gate length µm),
        ``"w"`` (fallback width for all devices).
        Example: ``{"w_N": 0.52, "w_P": 0.42, "l": 0.15}``.
    """

    def __init__(self, rules: PDKRules, params: dict[str, Any] | None = None):
        self.rules  = rules
        self.params = {k.lower(): v for k, v in (params or {}).items()}

    def place(self, template: CellTemplate) -> dict[str, PlacedDevice]:
        """Resolve all device positions.

        Returns
        -------
        dict[str, PlacedDevice]
            Ordered by placement evaluation order (bottom devices before top).
        """
        # Pass 1: compute transistor geometries
        geoms = self._compute_geoms(template)

        # Pass 2: resolve named scalar constraints
        named = resolve_named_constraints(
            template.named_constraints, self.rules, geoms
        )

        # Pass 3: place devices in topological order (bottom → top)
        return self._place_devices(template, geoms, named)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _w(self, dev_name: str, dev_type: str) -> float:
        p = self.params
        return (
            p.get(f"w_{dev_name.lower()}")
            or p.get("w")
            or _DEFAULT_W.get(dev_type, 0.52)
        )

    def _l(self, dev_name: str) -> float:
        return self.params.get(f"l_{dev_name.lower()}") or self.params.get("l") or _DEFAULT_L

    def _compute_geoms(self, template: CellTemplate) -> dict[str, TransistorGeom]:
        return {
            name: transistor_geom(
                self._w(name, spec.device_type),
                self._l(name),
                spec.device_type,
                self.rules,
            )
            for name, spec in template.devices.items()
        }

    def _place_devices(
        self,
        template: CellTemplate,
        geoms:    dict[str, TransistorGeom],
        named:    dict[str, float],
    ) -> dict[str, PlacedDevice]:
        placed: dict[str, PlacedDevice] = {}

        for dev_name in _topo_order(template.devices):
            spec = template.devices[dev_name]
            geom = geoms[dev_name]

            # Build namespace: rules + all geoms + named constraints + placed offsets
            placed_offsets: dict[str, float] = {}
            for pname, pd in placed.items():
                placed_offsets[f"{pname}_x"] = pd.x
                placed_offsets[f"{pname}_y"] = pd.y

            x = _resolve_x(spec, placed, geom, self.rules)
            y = eval_expr(
                spec.y_offset_expr,
                self.rules,
                geoms,
                named={**named, **placed_offsets},
            )

            placed[dev_name] = PlacedDevice(
                name=dev_name, spec=spec, geom=geom, x=x, y=y
            )

        return placed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _topo_order(devices: dict[str, DeviceSpec]) -> list[str]:
    """Return device names sorted so dependencies come first.

    Simple heuristic: devices with ``region`` containing ``"bottom"`` are
    placed before those with ``region == "top"``.  Within each tier the
    original YAML insertion order is preserved.
    """
    def _tier(spec: DeviceSpec) -> int:
        r = spec.region.lower()
        if "top" in r:
            return 1
        return 0   # "bottom", "bottom_only", default

    return sorted(devices.keys(), key=lambda n: _tier(devices[n]))


def _resolve_x(
    spec:   DeviceSpec,
    placed: dict[str, PlacedDevice],
    geom:   TransistorGeom,
    rules:  PDKRules,
) -> float:
    """Resolve the X origin for a device from its floorplan ``x_spec``."""
    x = spec.x_spec
    if x is None or x == "left":
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)

    xs = str(x).strip()

    # "right_of: DEV" or "right_of(DEV)"
    m = re.match(r"right_of\s*[:(]\s*(\w+)\s*\)?", xs, re.IGNORECASE)
    if m:
        ref = m.group(1)
        if ref in placed:
            p = placed[ref]
            sp = rules.diff["spacing_min_um"]
            return p.x + p.geom.total_x_um + sp

    # "between(DEV_A, DEV_B)"
    m = re.match(r"between\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", xs, re.IGNORECASE)
    if m:
        ra, rb = m.group(1), m.group(2)
        if ra in placed and rb in placed:
            pa, pb = placed[ra], placed[rb]
            gap = pb.x - (pa.x + pa.geom.total_x_um)
            return pa.x + pa.geom.total_x_um + (gap - geom.total_x_um) / 2

    # Fallback: treat as 0
    import warnings
    warnings.warn(
        f"Cannot resolve x_spec {xs!r}; defaulting to 0.0", stacklevel=4
    )
    return 0.0
