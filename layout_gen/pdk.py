"""
layout_gen.pdk — PDK layout rule loader.

Reads a layout PDK YAML (same concept as spice_gen's PDK YAML) and exposes
typed rule objects.  All geometry code imports from here — no layer numbers
or design-rule constants ever appear directly in cell or transistor code.

Typical use::

    from layout_gen.pdk import load_pdk, PDK_YAML

    rules = load_pdk()                        # default sky130A
    rules = load_pdk(Path("pdks/gf180.yaml")) # alternate technology

    layer = rules.layer("poly")               # → (66, 20)
    enc   = rules.contacts["enclosure_in_diff_um"]
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import List

# ── Repository-relative default paths ─────────────────────────────────────────

_HERE    = pathlib.Path(__file__).resolve().parent
PDK_YAML = _HERE / "pdks" / "sky130A.yaml"


# ── Internal loader ────────────────────────────────────────────────────────────

def _load_yaml(path: pathlib.Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── Via transition descriptor ──────────────────────────────────────────────────

@dataclass
class ViaTransition:
    """One via cut between two adjacent metals in the stack.

    Attributes
    ----------
    via_layer :
        Logical via layer name (e.g. ``"mcon"``, ``"via1"``).
    via_size :
        Via cut size in µm (square).
    lower_metal :
        Logical layer name of the metal below the via.
    upper_metal :
        Logical layer name of the metal above the via.
    enc_lower :
        Enclosure of the via in the lower metal (2-adj-edge, µm).
    enc_upper :
        Enclosure of the via in the upper metal (2-adj-edge, µm).
    """
    via_layer:    str
    via_size:     float
    lower_metal:  str
    upper_metal:  str
    enc_lower:    float
    enc_upper:    float


# ── Rule container ─────────────────────────────────────────────────────────────

@dataclass
class PDKRules:
    """All layout rules for one PDK, loaded from a YAML descriptor.

    Attributes
    ----------
    name:
        PDK name string (e.g. ``"sky130A"``).
    layers:
        Mapping of logical layer name → ``(gds_layer, gds_datatype)`` tuple.
    poly:
        Poly design rules dict (keys: width_min_um, spacing_min_um,
        endcap_over_diff_um).
    diff:
        Diffusion design rules dict (keys: width_min_um, spacing_min_um,
        extension_past_poly_um).
    contacts:
        Contact / licon rules dict (keys: size_um, spacing_um,
        enclosure_in_diff_um, enclosure_in_li1_um, space_to_poly_um).
    li1:
        Local interconnect rules dict (keys: width_min_um, spacing_min_um).
    implant:
        Implant rules dict (keys: enclosure_of_diff_um).
    nwell:
        N-well rules dict (keys: width_min_um, spacing_min_um,
        enclosure_of_pdiff_um).
    devices:
        Per-device-type rules dict.  Each entry keyed by logical device name
        (``"nmos"``, ``"pmos"``) contains: diff_layer, gate_layer,
        implant_layer, bulk_layer, nwell (bool), w_finger_max_um,
        sd_length_min_um.
    """

    name:     str
    layers:   dict[str, tuple[int, int]]
    poly:     dict
    diff:     dict
    contacts: dict
    li1:      dict
    met1:     dict
    met2:     dict
    met3:     dict
    met4:     dict
    met5:     dict
    via1:     dict
    via2:     dict
    via3:     dict
    via4:     dict
    implant:  dict
    nwell:    dict
    mcon:     dict
    tap:      dict
    npc:      dict
    devices:  dict
    drc:      dict    # DRC deck paths: {"klayout": "...", "magic": "..."}
    grid:     dict = field(default_factory=lambda: {
        "manufacturing_um": 0.005, "routing_um": 0.005,
    })
    preferred_direction: dict[str, str] = field(default_factory=dict)
    colors:   dict = field(default_factory=dict)
    _metal_stack_raw: list = field(default_factory=list)

    @property
    def li1_is_met1(self) -> bool:
        """True when li1 and met1 are the same GDS layer (e.g. GF180MCU)."""
        try:
            return self.layer("li1") == self.layer("met1")
        except KeyError:
            return False

    def layer(self, name: str) -> tuple[int, int]:
        """Return the ``(gds_layer, gds_datatype)`` pair for a logical layer name.

        Raises
        ------
        KeyError
            If *name* is not defined in the PDK YAML.
        """
        try:
            entry = self.layers[name]
        except KeyError:
            raise KeyError(
                f"Layer {name!r} not found in PDK {self.name!r}. "
                f"Available: {sorted(self.layers)}"
            )
        return (entry["layer"], entry["datatype"])

    def device(self, logical_name: str) -> dict:
        """Return the rule dict for a logical device type (e.g. ``'nmos'``).

        Raises
        ------
        KeyError
            If the device type is not defined in the PDK YAML.
        """
        try:
            return self.devices[logical_name]
        except KeyError:
            raise KeyError(
                f"Device {logical_name!r} not defined in PDK {self.name!r}. "
                f"Available: {sorted(self.devices)}"
            )

    # ── Grid helpers ──────────────────────────────────────────────────────────

    @property
    def mfg_grid(self) -> float:
        """Manufacturing grid quantum in µm."""
        return self.grid.get("manufacturing_um", 0.005)

    @property
    def routing_grid(self) -> float:
        """Routing grid quantum in µm."""
        return self.grid.get("routing_um", self.mfg_grid)

    def snap(self, value: float, grid: str = "mfg") -> float:
        """Snap *value* to the nearest grid point.

        Parameters
        ----------
        grid : ``"mfg"`` or ``"routing"``
        """
        g = self.mfg_grid if grid == "mfg" else self.routing_grid
        if g <= 0:
            return value
        return round(round(value / g) * g, 6)

    def direction(self, metal_layer: str) -> str:
        """Return preferred routing direction for *metal_layer*.

        Returns ``"horizontal"``, ``"vertical"``, or ``""`` (no preference).
        """
        return self.preferred_direction.get(metal_layer, "")

    def enclosure(self, section: str, key_prefix: str) -> tuple[float, float]:
        """Return ``(adj2, opp)`` enclosure values for an asymmetric rule.

        Enclosure rules follow a consistent YAML schema:

        - ``<key_prefix>_um``      — all-sides minimum (or opposite-side value
          when a ``_2adj_um`` variant exists).
        - ``<key_prefix>_2adj_um`` — 2-adjacent-edge minimum (optional; when
          present, the opposite 2 edges use ``_um``).

        When only ``_um`` exists the enclosure is symmetric: both values equal
        the ``_um`` value.

        Parameters
        ----------
        section :
            Rule-section name (``"contacts"``, ``"met1"``, ``"via1"``, …).
        key_prefix :
            Key prefix before the ``_um`` / ``_2adj_um`` suffix
            (e.g. ``"poly_enclosure"`` → reads ``poly_enclosure_um`` and
            ``poly_enclosure_2adj_um``).

        Returns
        -------
        (adj2, opp)
            ``adj2`` = enclosure on 2 adjacent edges,
            ``opp``  = enclosure on the 2 opposite edges.

        Example
        -------
        >>> rules.enclosure("contacts", "poly_enclosure")
        (0.05, 0.0)
        >>> rules.enclosure("contacts", "enclosure_in_diff")
        (0.04, 0.04)   # symmetric — no _2adj variant
        """
        sec = getattr(self, section, None)
        if sec is None:
            sec = {}
        all_val  = sec.get(f"{key_prefix}_um", 0.0)
        adj2_val = sec.get(f"{key_prefix}_2adj_um", None)
        if adj2_val is not None:
            return (float(adj2_val), float(all_val))
        return (float(all_val), float(all_val))

    # ── Metal-stack helpers ──────────────────────────────────────────────────

    def _resolve_metal(self, layer_name: str) -> str:
        """Map a logical layer name to its canonical metal-stack name.

        When li1 and met1 share the same GDS layer, ``"li1"`` resolves to
        the stack entry that carries that GDS layer (typically ``"met1"``).
        """
        try:
            gds = self.layer(layer_name)
        except KeyError:
            return layer_name
        for entry in self._metal_stack_raw:
            metal = entry["metal"]
            try:
                if self.layer(metal) == gds:
                    return metal
            except KeyError:
                continue
        return layer_name

    def via_stack_between(
        self, from_layer: str, to_layer: str,
    ) -> List[ViaTransition]:
        """Return the ordered via transitions needed to connect two metals.

        Parameters
        ----------
        from_layer, to_layer :
            Logical layer names (e.g. ``"li1"``, ``"met2"``).

        Returns
        -------
        list[ViaTransition]
            Empty if the layers resolve to the same stack position (no via
            needed).  Otherwise one ``ViaTransition`` per cut, ordered
            bottom-to-top.
        """
        stack = self._metal_stack_raw
        if not stack:
            return []

        # Resolve aliases (li1 → met1 when they share a GDS layer)
        src = self._resolve_metal(from_layer)
        dst = self._resolve_metal(to_layer)
        if src == dst:
            return []

        # Build ordered metal name list and locate src/dst
        metals = [e["metal"] for e in stack]
        try:
            i_src = metals.index(src)
            i_dst = metals.index(dst)
        except ValueError:
            return []

        if i_src > i_dst:
            i_src, i_dst = i_dst, i_src

        transitions: List[ViaTransition] = []
        for i in range(i_src + 1, i_dst + 1):
            entry = stack[i]
            via_name = entry.get("via")
            if via_name is None:
                continue  # no via between these adjacent metals

            # Read via size from the rules section named in via_rules
            via_rules_section = entry.get("via_rules", via_name)
            via_sec = getattr(self, via_rules_section, None) or {}
            via_size = via_sec.get("size_um", self.contacts.get("size_um", 0.17))

            # Read enclosures
            lower_enc_spec = entry.get("lower_enc", {})
            upper_enc_spec = entry.get("upper_enc", {})

            def _read_enc(spec: dict) -> float:
                sec = getattr(self, spec.get("section", ""), None) or {}
                return float(sec.get(spec.get("key", ""), 0.0))

            enc_lower = _read_enc(lower_enc_spec)
            enc_upper = _read_enc(upper_enc_spec)

            transitions.append(ViaTransition(
                via_layer=via_name,
                via_size=via_size,
                lower_metal=metals[i - 1],
                upper_metal=metals[i],
                enc_lower=enc_lower,
                enc_upper=enc_upper,
            ))

        return transitions

    def sd_contact_columns(self, w_finger_um: float) -> int:
        """Number of contact rows that fit in a source/drain of width *w_finger_um*.

        Contacts are arrayed along the channel-width (Y) direction.  Returns
        at least 1.
        """
        c_size  = self.contacts["size_um"]
        c_space = self.contacts["spacing_um"]
        # contacts fit when: n * c_size + (n-1) * c_space ≤ w_finger_um - 2*enc
        enc    = self.contacts["enclosure_in_diff_um"]
        usable = w_finger_um - 2 * enc
        if usable < c_size:
            return 1
        return max(1, int((usable + c_space) / (c_size + c_space)))


# ── Public API ─────────────────────────────────────────────────────────────────

def load_pdk(pdk_yaml: pathlib.Path | str = PDK_YAML) -> PDKRules:
    """Load and return a :class:`PDKRules` from a layout PDK YAML file.

    Parameters
    ----------
    pdk_yaml :
        Path to the layout PDK YAML.  Defaults to the bundled sky130A config.
    """
    pdk_yaml = pathlib.Path(pdk_yaml)
    data   = _load_yaml(pdk_yaml)
    rules  = data.get("rules", {})

    # Resolve DRC deck paths: expand env vars, resolve relative to YAML dir
    raw_drc = data.get("drc", {})
    drc: dict[str, str] = {}
    for tool, path_str in raw_drc.items():
        # Env-var override takes priority (e.g. DRC_DECK_KLAYOUT)
        env_key = f"DRC_DECK_{tool.upper()}"
        resolved = os.environ.get(env_key) or os.path.expandvars(path_str)
        p = pathlib.Path(resolved)
        if not p.is_absolute():
            p = pdk_yaml.parent / p
        drc[tool] = str(p.resolve())

    # Parse preferred_direction: YAML values may be unquoted strings or ""
    raw_pdir = data.get("preferred_direction", {})
    pdir: dict[str, str] = {}
    for layer_name, direction in raw_pdir.items():
        pdir[layer_name] = str(direction) if direction else ""

    return PDKRules(
        name     = data["name"],
        layers   = data.get("layers",  {}),
        poly     = rules.get("poly",    {}),
        diff     = rules.get("diff",    {}),
        contacts = rules.get("contacts", {}),
        li1      = rules.get("li1",     {}),
        met1     = rules.get("met1",    {}),
        met2     = rules.get("met2",    {}),
        met3     = rules.get("met3",    {}),
        met4     = rules.get("met4",    {}),
        met5     = rules.get("met5",    {}),
        via1     = rules.get("via1",    {}),
        via2     = rules.get("via2",    {}),
        via3     = rules.get("via3",    {}),
        via4     = rules.get("via4",    {}),
        implant  = rules.get("implant", {}),
        nwell    = rules.get("nwell",   {}),
        mcon     = rules.get("mcon",    {}),
        tap      = rules.get("tap",     {}),
        npc      = rules.get("npc",     {}),
        devices  = data.get("devices",  {}),
        drc      = drc,
        grid     = data.get("grid", {}),
        preferred_direction = pdir,
        colors   = data.get("colors", {}),
        _metal_stack_raw = data.get("metal_stack", []),
    )


# ── Module-level default (sky130A) ────────────────────────────────────────────
# Lazy-loaded: avoids import-time crash if the bundled PDK YAML is absent.
# First access via ``RULES`` or any cell/transistor function triggers loading.

class _LazyRules:
    """Proxy that loads PDKRules on first attribute access."""

    __slots__ = ('_rules',)

    def __init__(self):
        object.__setattr__(self, '_rules', None)

    def _load(self) -> PDKRules:
        r = object.__getattribute__(self, '_rules')
        if r is None:
            r = load_pdk()
            object.__setattr__(self, '_rules', r)
        return r

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        r = object.__getattribute__(self, '_rules')
        return repr(r) if r is not None else 'PDKRules(<lazy: not yet loaded>)'


RULES: PDKRules = _LazyRules()  # type: ignore[assignment]
