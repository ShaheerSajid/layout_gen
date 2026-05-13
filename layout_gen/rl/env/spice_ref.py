"""
layout_gen.rl.env.spice_ref — emit a reference SPICE netlist from a TopologyGraph.

LVS needs a reference netlist to compare the layout against. Rather than
require the user to hand-write one per cell, we synthesise a minimal
SPICE subckt directly from the topology graph: one device per
:class:`DeviceNode`, terminals wired to the named nets, w/l carried
through.

Bulks
-----
Magic / netgen need every transistor to have a bulk (B) terminal. For
the sky130-style flow we follow these conventions:

  * NMOS bulk → the cell's bulk-low net (typically ``VSS`` / ``GND``)
  * PMOS bulk → the cell's bulk-high net (typically ``VDD``)

If the topology declares power nets via ``net_type='power'`` we use
the ``rail='bottom'`` net for NMOS bulks and ``rail='top'`` for PMOS.
Failing that, we fall back to the literal names ``GND`` / ``VDD``.

Models
------
We emit ``sky130_fd_pr__nfet_01v8`` / ``sky130_fd_pr__pfet_01v8`` by
default — these are the standard sky130 device models that magic's
extraction will produce, so netgen's name comparison succeeds out of
the box. Override via ``nmos_model`` / ``pmos_model`` for other PDKs.
"""
from __future__ import annotations

from pathlib import Path

from layout_gen.rl.topology.parser import TopologyGraph


# ── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_bulks(graph: TopologyGraph) -> tuple[str, str]:
    """Return ``(nmos_bulk_net, pmos_bulk_net)`` from the topology."""
    bottom = next(
        (n.name for n in graph.nets
         if n.net_type == "power" and n.rail == "bottom"),
        None,
    )
    top = next(
        (n.name for n in graph.nets
         if n.net_type == "power" and n.rail == "top"),
        None,
    )
    return (bottom or "VSS"), (top or "VDD")


def _ports_for(graph: TopologyGraph) -> list[str]:
    """Cell-port nets — every power + signal net (skip ``internal`` ones).

    Magic's port list comes from the layout's labelled metals; for the
    reference netlist we want it to contain the same names so netgen
    can match. ``internal`` nets are not exposed at the cell boundary.
    """
    ports: list[str] = []
    for n in graph.nets:
        if n.net_type in ("power", "signal"):
            ports.append(n.name)
    return ports


# ── Emitter ─────────────────────────────────────────────────────────────────

def emit_spice_subckt(
    graph:        TopologyGraph,
    cell_name:    str,
    *,
    nmos_model:   str = "sky130_fd_pr__nfet_01v8",
    pmos_model:   str = "sky130_fd_pr__pfet_01v8",
) -> str:
    """Return a SPICE ``.subckt`` block as a string."""
    nmos_bulk, pmos_bulk = _resolve_bulks(graph)
    ports = _ports_for(graph)

    lines: list[str] = [
        f"* layout_gen.rl.env.spice_ref — auto-generated reference for {cell_name}",
        f".subckt {cell_name} " + " ".join(ports),
    ]
    for d in graph.devices:
        terms = d.terminal_nets
        if not all(k in terms for k in ("G", "D", "S")):
            # Skip devices that don't have full G/D/S declared.
            continue
        if d.device_type == "pmos":
            model, bulk = pmos_model, pmos_bulk
        else:
            model, bulk = nmos_model, nmos_bulk
        # SPICE subckt-instance card (X) referencing the PDK model:
        # X<inst> <D> <G> <S> <B> <model> w=.. l=..
        # Magic's extraction emits these as X cards too, so netgen
        # matches them by model name without aliasing.
        lines.append(
            f"X{d.name} {terms['D']} {terms['G']} {terms['S']} {bulk} "
            f"{model} w={d.w_um:.4g}u l={d.l_um:.4g}u"
        )
    lines.append(".ends")
    return "\n".join(lines) + "\n"


def write_spice_subckt(
    graph:     TopologyGraph,
    cell_name: str,
    out_path:  Path,
    **kwargs,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(emit_spice_subckt(graph, cell_name, **kwargs))
    return out_path


__all__ = ["emit_spice_subckt", "write_spice_subckt"]
