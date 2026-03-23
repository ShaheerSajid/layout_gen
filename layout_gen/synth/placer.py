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
from layout_gen.synth.loader      import CellTemplate, DeviceSpec, RowPairSpec
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

    # S/D flip: when a device is flipped for diffusion sharing, swap the
    # physical positions of S and D (left↔right).
    phys_term = term
    if dev.spec.sd_flip and term in ("S", "D"):
        phys_term = "D" if term == "S" else "S"

    if phys_term == "G":
        lx0, lx1 = _gate_x(0, geom)
        return TerminalGeom(
            dev_name, term,
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=dev.y,        y1=dev.y + geom.total_y_um,
            layer="poly",
        )
    elif phys_term == "D":
        j = geom.n_fingers   # rightmost S/D = drain for finger 0
        lx0, lx1 = _sd_x(j, geom)
        ly0, ly1 = _diff_y(geom, rules)
        return TerminalGeom(
            dev_name, term,
            x0=lx0 + dev.x, x1=lx1 + dev.x,
            y0=ly0 + dev.y, y1=ly1 + dev.y,
            layer="li1",
        )
    elif phys_term == "S":
        lx0, lx1 = _sd_x(0, geom)
        ly0, ly1 = _diff_y(geom, rules)
        return TerminalGeom(
            dev_name, term,
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


def global_sd_x(
    dev:   PlacedDevice,
    j:     int,
    rules: PDKRules | None = None,
) -> tuple[float, float]:
    """Global (x0, x1) of the *j*-th source/drain region.

    When *rules* is provided, li1 pullback is applied (matching the
    geometry produced by :func:`~layout_gen.transistor.draw_transistor`).
    """
    lx0, lx1 = _sd_x(j, dev.geom, rules)
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

        # Pass 3: place devices
        if template.layout_mode == "stacked":
            return self._place_stacked(template, geoms, named)
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
        from dataclasses import replace as _replace
        result: dict[str, TransistorGeom] = {}
        for name, spec in template.devices.items():
            geom = transistor_geom(
                self._w(name, spec.device_type),
                self._l(name),
                spec.device_type,
                self.rules,
            )
            n = int(spec.fingers)
            if n > 0 and n != geom.n_fingers:
                w_f    = geom.w_um / n
                endcap = self.rules.poly["endcap_over_diff_um"]
                geom   = _replace(
                    geom,
                    n_fingers    = n,
                    w_finger_um  = w_f,
                    total_x_um   = (n + 1) * geom.sd_length_um + n * geom.l_um,
                    total_y_um   = w_f + 2 * endcap,
                    n_contacts_y = self.rules.sd_contact_columns(w_f),
                )
            result[name] = geom
        return result

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

            full_named = {**named, **placed_offsets}
            x = _resolve_x(spec, placed, geom, self.rules, geoms, full_named)
            y = eval_expr(
                spec.y_offset_expr,
                self.rules,
                geoms,
                named=full_named,
            )

            placed[dev_name] = PlacedDevice(
                name=dev_name, spec=spec, geom=geom, x=x, y=y
            )

        return placed

    def _place_stacked(
        self,
        template: CellTemplate,
        geoms:    dict[str, TransistorGeom],
        named:    dict[str, float],
    ) -> dict[str, PlacedDevice]:
        """Place devices in a vertically stacked multi-row layout.

        Each :class:`RowPairSpec` produces an NMOS tier (bottom) and PMOS tier
        (top), stacked vertically.  Devices within a tier are placed
        left-to-right with shared-diffusion abutment.
        """
        placed: dict[str, PlacedDevice] = {}

        # Compute the inter-cell gap (within a row pair, between N and P tiers)
        endcap = self.rules.poly["endcap_over_diff_um"]
        ext_y  = self.rules.diff["extension_past_poly_um"]
        diff_sp = self.rules.diff["spacing_min_um"]
        icg = named.get(
            "inter_cell_gap",
            diff_sp - 2 * endcap + 2 * ext_y,
        )

        # Gap between adjacent row pairs (default: diff spacing)
        inter_row_gap = named.get("inter_row_gap", diff_sp)

        current_y = 0.0

        for rp_idx, rp in enumerate(template.row_pairs):
            n_list = [(n, geoms[n]) for n in rp.nmos_devices if n in geoms]
            p_list = [(n, geoms[n]) for n in rp.pmos_devices if n in geoms]

            nmos_h = max((g.total_y_um for _, g in n_list), default=0.0)

            # ── NMOS tier ────────────────────────────────────────────────
            x = 0.0
            for i, (name, geom) in enumerate(n_list):
                if i > 0:
                    # Abutment: overlap one shared S/D region
                    x -= geom.sd_length_um
                placed[name] = PlacedDevice(
                    name=name, spec=template.devices[name], geom=geom,
                    x=x, y=current_y,
                )
                x += geom.total_x_um

            # ── PMOS tier ────────────────────────────────────────────────
            if n_list:
                pmos_y = current_y + nmos_h + icg
            else:
                pmos_y = current_y  # no NMOS → PMOS at row pair base

            x = 0.0
            for i, (name, geom) in enumerate(p_list):
                if i > 0:
                    x -= geom.sd_length_um
                placed[name] = PlacedDevice(
                    name=name, spec=template.devices[name], geom=geom,
                    x=x, y=pmos_y,
                )
                x += geom.total_x_um

            # ── Advance Y for next row pair ──────────────────────────────
            pmos_h = max((g.total_y_um for _, g in p_list), default=0.0)
            if n_list and p_list:
                row_top = pmos_y + pmos_h
            elif n_list:
                row_top = current_y + nmos_h
            else:
                row_top = pmos_y + pmos_h

            if rp_idx < len(template.row_pairs) - 1:
                current_y = row_top + inter_row_gap

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
    geoms:  dict[str, TransistorGeom] | None = None,
    named:  dict[str, float]          | None = None,
) -> float:
    """Resolve the X origin for a device from its floorplan ``x_spec``.

    Supports four forms:

    * ``None`` / ``"left"`` — place at X = 0.
    * Number — use directly.
    * ``"right_of: DEV"`` or ``"right_of(DEV)"`` — right edge of *DEV* plus
      ``diff.spacing_min_um``.
    * ``"between(DEV_A, DEV_B)"`` — centre the device in the gap between
      *DEV_A* right edge and *DEV_B* left edge.
    * Any other string — evaluated as a constraint expression via
      ``eval_expr`` (same namespace as ``y_offset_expr``), giving access to
      ``rules.*``, device geom attributes, and placed offsets
      (``PD_L_x``, ``PD_R_x``, …).
    """
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

    # Fallback: arbitrary constraint expression (same namespace as y_offset_expr)
    if geoms is not None:
        try:
            return eval_expr(xs, rules, geoms, named=named)
        except ValueError:
            pass

    import warnings
    warnings.warn(
        f"Cannot resolve x_spec {xs!r}; defaulting to 0.0", stacklevel=4
    )
    return 0.0
