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

# ── Repository-relative default paths ─────────────────────────────────────────

_HERE    = pathlib.Path(__file__).resolve().parent
PDK_YAML = _HERE / "pdks" / "sky130A.yaml"


# ── Internal loader ────────────────────────────────────────────────────────────

def _load_yaml(path: pathlib.Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


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
    via1:     dict
    implant:  dict
    nwell:    dict
    mcon:     dict
    devices:  dict
    drc:      dict    # DRC deck paths: {"klayout": "...", "magic": "..."}

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

    return PDKRules(
        name     = data["name"],
        layers   = data.get("layers",  {}),
        poly     = rules.get("poly",    {}),
        diff     = rules.get("diff",    {}),
        contacts = rules.get("contacts", {}),
        li1      = rules.get("li1",     {}),
        met1     = rules.get("met1",    {}),
        met2     = rules.get("met2",    {}),
        via1     = rules.get("via1",    {}),
        implant  = rules.get("implant", {}),
        nwell    = rules.get("nwell",   {}),
        mcon     = rules.get("mcon",    {}),
        devices  = data.get("devices",  {}),
        drc      = drc,
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
