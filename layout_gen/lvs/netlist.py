"""
layout_gen.lvs.netlist — reference SPICE netlist from a CellTemplate.

The reference netlist is what LVS compares the extracted-from-GDS netlist
against.  We produce it via :mod:`spice_gen` when available so that the
generation logic stays in one place.

Bulk inference
--------------
Topology templates often omit the ``B`` (bulk) terminal because layout-side
bulk is implicit (substrate / well taps).  For LVS comparison we still need
each MOSFET to have a 4th terminal in the SPICE primitive line.  The rule:

- NMOS bulk → power net with ``rail: bottom`` (typically ``GND`` / ``VSS``)
- PMOS bulk → power net with ``rail: top`` (typically ``VDD``)

If the template has no matching power net, the runner will need to add one
to its setup mapping (e.g. global substrate).
"""
from __future__ import annotations

from typing import Iterable

from layout_gen.pdk        import PDKRules
from layout_gen.synth.loader import CellTemplate, DeviceSpec


# ── Bulk inference ───────────────────────────────────────────────────────────

def _infer_bulk_net(
    device_type: str,
    template:    CellTemplate,
) -> str:
    """Return the power net that supplies bulk for *device_type*.

    Falls back to common names (GND / VDD) when the template doesn't declare
    rails explicitly.
    """
    want_rail = "bottom" if device_type == "nmos" else "top"
    # Prefer explicit rail tagging
    for net in template.nets.values():
        if net.net_type == "power" and net.rail == want_rail:
            return net.name
    # Fall back to canonical names present in the template
    fallbacks = ("GND", "VSS") if device_type == "nmos" else ("VDD",)
    for n in fallbacks:
        if n in template.nets:
            return n
    # Last resort — pick any power net
    for net in template.nets.values():
        if net.net_type == "power":
            return net.name
    return "GND" if device_type == "nmos" else "VDD"


# ── Width / fingers resolution ───────────────────────────────────────────────

def _resolve_w_l(
    device:   DeviceSpec,
    params:   dict,
    rules:    PDKRules,
) -> tuple[float, float, int]:
    """Return (w_um, l_um, fingers) the synthesizer would draw.

    Mirrors the placer's resolution: per-device override > params w_<type>
    > params w > diff width minimum.
    """
    # Length
    l = float(device.l) if device.l > 0 else float(params.get("l", 0.0))
    if l <= 0:
        l = float(rules.poly.get("width_min_um", 0.15))

    # Width: per-device w > params w_<DevName> > params w_<type> > params w
    w = 0.0
    if device.w > 0:
        w = float(device.w)
    else:
        type_key = "w_N" if device.device_type == "nmos" else "w_P"
        for key in (f"w_{device.name}", type_key, "w"):
            if key in params and float(params[key]) > 0:
                w = float(params[key])
                break
    if w <= 0:
        w = float(rules.diff.get("width_min_um", 0.42))

    # Fingers: explicit > auto from w_finger_max
    if device.fingers > 0:
        nf = device.fingers
    else:
        wmax = float(rules.device(device.device_type).get("w_finger_max_um", 2.0))
        import math
        nf = max(1, math.ceil(w / wmax))
    return w, l, nf


# ── Public API ───────────────────────────────────────────────────────────────

def build_reference_netlist(
    template: CellTemplate,
    rules:    PDKRules,
    params:   dict | None = None,
    *,
    dialect:  str = "ngspice",
) -> str:
    """Generate the LVS reference SPICE netlist for *template*.

    Parameters
    ----------
    template :
        Parsed cell topology template.
    rules :
        PDK rules — supplies the ``lvs.model_<type>`` model name and bulk
        defaults.
    params :
        Same shape the synthesizer accepts (``w_N``, ``w_P``, ``l``, …).
        Used only to compute W / L on the SPICE side so the reference matches
        what the layout was drawn with.
    dialect :
        ``"ngspice"`` (default), ``"hspice"``, or ``"spice3"`` — passed to
        :mod:`spice_gen`.
    """
    params = dict(params or {})

    # PDK device model names — fall back to logical type if not in YAML
    lvs_cfg = getattr(rules, "lvs", {}) or {}
    model_map: dict[str, str] = {
        "nmos": str(lvs_cfg.get("model_nmos", "nmos")),
        "pmos": str(lvs_cfg.get("model_pmos", "pmos")),
    }
    # Per-device overrides on the device dict
    for tname in ("nmos", "pmos"):
        per_dev = rules.device(tname).get("lvs_model_name")
        if per_dev:
            model_map[tname] = str(per_dev)

    # Magic ext2spice emits each transistor as an X-instance of the PDK
    # device subcircuit (e.g. ``X0 D G S B sky130_fd_pr__nfet_01v8 w=… l=…``).
    # Mirror that on the reference side so netgen sees the same device class
    # — primitive M lines and X-subckt lines aren't equated by default.
    # Port order matches Magic's: D G S B.
    components = []
    for dname, dev in template.devices.items():
        w, l, nf = _resolve_w_l(dev, params, rules)
        terms = dict(dev.terminals)
        terms.setdefault("B", _infer_bulk_net(dev.device_type, template))
        components.append({
            "id":          dname,
            "type":        "subckt",
            "model":       model_map[dev.device_type],
            "connections": {"D": terms.get("D", ""),
                            "G": terms.get("G", ""),
                            "S": terms.get("S", ""),
                            "B": terms["B"]},
            "parameters":  {
                # Plain numeric µm to match Magic's ext2spice default
                # (which prints W / L unit-less in microns).  A trailing
                # ``u`` would cause netgen to convert to meters and a
                # spurious property-mismatch error.
                "w": f"{w}",
                "l": f"{l}",
                **({"m": str(nf)} if nf > 1 else {}),
            },
        })

    # Port order: declared ports first (in YAML order), bulk power last.
    declared = list(template.ports.keys())
    power_extras = [
        n for n, ns in template.nets.items()
        if ns.net_type == "power" and n not in declared
    ]
    ordered_ports = declared + power_extras
    if not ordered_ports:
        # Pathological: no ports — collect every net the devices touch
        seen: list[str] = []
        for c in components:
            for net in c["connections"].values():
                if net and net not in seen:
                    seen.append(net)
        ordered_ports = seen

    return _format_via_spice_gen(template.name, ordered_ports, components,
                                 dialect=dialect)


# ── spice_gen plumbing ───────────────────────────────────────────────────────

def _format_via_spice_gen(
    name:       str,
    ports:      Iterable[str],
    components: list[dict],
    *,
    dialect:    str,
) -> str:
    try:
        from spice_gen.schema.cell_schema import CellSchema, TopLevelSchema
        from spice_gen.parser.builder import build_subckt_def
        from spice_gen.model.netlist import Netlist
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "spice_gen is required for LVS reference netlist generation. "
            "Install it with `pip install -e vendor/spice_gen`."
        ) from exc

    cell = CellSchema(name=name, ports=list(ports), components=components)
    netlist = Netlist(subckt_defs=[build_subckt_def(cell)], top_cell=cell.name)

    if dialect == "ngspice":
        from spice_gen.generator.ngspice import NgspiceGenerator
        gen = NgspiceGenerator()
    elif dialect == "hspice":
        from spice_gen.generator.hspice import HspiceGenerator
        gen = HspiceGenerator()
    else:
        from spice_gen.generator.spice3 import Spice3Generator
        gen = Spice3Generator()

    return gen.generate(netlist)
