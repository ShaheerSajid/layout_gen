"""
layout_gen.synth.auto_router — algorithmic routing planner.

Replaces hand-specified routing directives with an automatic planner
that analyses the :class:`~layout_gen.synth.netlist.NetGraph` and
placed device positions to generate :class:`~layout_gen.synth.loader.RoutingSpec`
objects consumed by the existing style handlers.

The planner works in five phases, from most constrained to least:

A.  **Intra-pair local routing** — poly gate bridges, drain bridges,
    li1 bridges (deterministic from row-pair co-location).
B.  **Inter-pair routing** — cross-row gate/drain connections with
    automatic metal-layer and track-position selection.
C.  **Vertical buses** — multi-row S/D buses (bitlines).
D.  **Power rails** — VDD/VSS from net type declarations.
E.  **Terminal exposure** — ports for external access.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

from layout_gen.pdk import PDKRules
from layout_gen.synth.loader import CellTemplate, RoutingSpec
from layout_gen.synth.placer import PlacedDevice, resolve_terminal
from layout_gen.synth.netlist import NetGraph, NetInfo, TerminalRef


# ── AutoRouter ────────────────────────────────────────────────────────────────

class AutoRouter:
    """Plans routing for a v3 template, emitting RoutingSpec objects.

    Parameters
    ----------
    rules : PDKRules
        PDK design rules (spacing, width, enclosure).
    """

    def __init__(self, rules: PDKRules):
        self.rules = rules

    def plan(
        self,
        net_graph: NetGraph,
        placed:    dict[str, PlacedDevice],
        template:  CellTemplate,
    ) -> list[RoutingSpec]:
        """Generate all routing specs for the given placement.

        Returns a list of :class:`RoutingSpec` objects that can be passed
        directly to :meth:`Router.route`.
        """
        specs: list[RoutingSpec] = []

        # Build lookup: row_pair_id → set of device names
        row_devices = _build_row_device_map(placed)

        # Phase A: intra-pair local routing
        specs.extend(self._phase_a_intra_pair(net_graph, placed, row_devices))

        # Phase B: inter-pair cross-row routing
        specs.extend(self._phase_b_inter_pair(net_graph, placed, row_devices))

        # Phase C: vertical buses (multi-row S/D)
        specs.extend(self._phase_c_vertical_buses(net_graph, placed, row_devices))

        # Phase D: power rails
        specs.extend(self._phase_d_power_rails(net_graph))

        # Phase E: terminal exposure (handled by port_resolver, not here)
        # Port exposure specs are generated separately in port_resolver.py
        # to keep this module focused on metal routing.

        return specs

    # ── Phase A: intra-pair local routing ─────────────────────────────────

    def _phase_a_intra_pair(
        self,
        ng:          NetGraph,
        placed:      dict[str, PlacedDevice],
        row_devices: dict[int, set[str]],
    ) -> list[RoutingSpec]:
        """Generate poly bridges, drain bridges, and li1 bridges
        for terminals sharing a net within the same row pair."""
        specs: list[RoutingSpec] = []
        handled_pairs: set[tuple[str, str]] = set()   # (net, style_key) dedup

        for net_name, info in ng.nets.items():
            if info.is_power:
                continue

            # Group terminals by row pair
            by_row: dict[int, list[TerminalRef]] = defaultdict(list)
            for tref in info.terminals:
                dev = placed.get(tref.device)
                if dev is None:
                    continue
                by_row[dev.spec.row_pair].append(tref)

            for row_id, trefs in by_row.items():
                if row_id < 0:
                    # Standard mode: row_pair == -1 means single pair
                    row_id = -1

                nmos_terms = [t for t in trefs
                              if ng.device_types[t.device] == "nmos"]
                pmos_terms = [t for t in trefs
                              if ng.device_types[t.device] == "pmos"]

                # ── Shared gate poly bridge ──
                nmos_gates = [t for t in nmos_terms if t.terminal == "G"]
                pmos_gates = [t for t in pmos_terms if t.terminal == "G"]

                for ng_t in nmos_gates:
                    for pg_t in pmos_gates:
                        key = (net_name, f"gate_{ng_t.device}_{pg_t.device}")
                        if key in handled_pairs:
                            continue
                        handled_pairs.add(key)
                        specs.append(RoutingSpec(
                            net=net_name,
                            style="shared_gate_poly",
                            layer="poly",
                            path=[ng_t.ref, pg_t.ref],
                        ))

                # ── Drain bridge (vertical li1 N.D → P.D) ──
                nmos_drains = [t for t in nmos_terms if t.terminal == "D"]
                pmos_drains = [t for t in pmos_terms if t.terminal == "D"]

                for nd in nmos_drains:
                    for pd in pmos_drains:
                        # Only bridge if gates are column-aligned
                        # (same device X position implies vertical alignment)
                        nd_dev = placed[nd.device]
                        pd_dev = placed[pd.device]
                        if abs(nd_dev.x - pd_dev.x) < 0.01:
                            key = (net_name, f"drain_{nd.device}_{pd.device}")
                            if key in handled_pairs:
                                continue
                            handled_pairs.add(key)
                            specs.append(RoutingSpec(
                                net=net_name,
                                style="drain_bridge",
                                layer="li1",
                                path=[nd.ref, pd.ref],
                            ))

                # ── Li1 bridge (abutting S/D within same tier) ──
                # Find pairs of terminals on same net in same row pair
                # where both are S/D and on the same device type
                for dtype in ("nmos", "pmos"):
                    sd_terms = [t for t in trefs
                                if ng.device_types[t.device] == dtype
                                and t.terminal in ("S", "D")]
                    if len(sd_terms) < 2:
                        continue
                    # Sort by device X to find adjacent pairs
                    sd_terms.sort(key=lambda t: placed[t.device].x)
                    for i in range(len(sd_terms)):
                        for j in range(i + 1, len(sd_terms)):
                            a, b = sd_terms[i], sd_terms[j]
                            if a.device == b.device:
                                continue  # skip same device
                            key = (net_name, f"li1_{a.device}_{b.device}")
                            if key in handled_pairs:
                                continue
                            handled_pairs.add(key)
                            specs.append(RoutingSpec(
                                net=net_name,
                                style="li1_bridge",
                                layer="li1",
                                path=[a.ref, b.ref],
                            ))

        return specs

    # ── Phase B: inter-pair cross-row routing ─────────────────────────────

    def _phase_b_inter_pair(
        self,
        ng:          NetGraph,
        placed:      dict[str, PlacedDevice],
        row_devices: dict[int, set[str]],
    ) -> list[RoutingSpec]:
        """Generate cross_row_connect specs for nets that span row pairs.

        Selects met1 for short spans (adjacent rows) and met2 for long
        spans (2+ rows apart).  Uses a track allocator to assign X
        positions and avoid shorts.
        """
        if not row_devices or max(row_devices.keys(), default=-1) < 0:
            return []  # not a stacked layout

        specs: list[RoutingSpec] = []
        allocated_tracks: list[_TrackAllocation] = []

        # Collect all cross-row connections needed
        connections: list[_CrossRowConnection] = []

        for net_name, info in ng.nets.items():
            if info.is_power:
                continue

            # Find S/D source terminals and gate targets in different rows
            by_row: dict[int, list[TerminalRef]] = defaultdict(list)
            for tref in info.terminals:
                dev = placed.get(tref.device)
                if dev is None:
                    continue
                by_row[dev.spec.row_pair].append(tref)

            if len(by_row) < 2:
                continue  # all on same row, handled by Phase A

            # Find the "source" — a S/D terminal that drives cross-row gates
            # Source = first S/D terminal; targets = gates in other rows
            source = None
            targets: list[TerminalRef] = []

            # Prefer drain terminals as sources (they carry driven signals)
            all_sd = [(tref, placed[tref.device].spec.row_pair)
                      for tref in info.sd_terminals
                      if tref.device in placed]
            all_gates = [(tref, placed[tref.device].spec.row_pair)
                         for tref in info.gate_terminals
                         if tref.device in placed]

            if not all_sd or not all_gates:
                continue  # no cross-row connection possible

            # Source = S/D terminal with drain bridge already placed
            # (Phase A handles drain bridge, so the S/D terminal is
            # accessible via li1)
            for sd_tref, sd_row in all_sd:
                cross_gates = [g for g, g_row in all_gates if g_row != sd_row]
                if cross_gates:
                    source = sd_tref
                    targets = cross_gates
                    break

            if source is None or not targets:
                continue

            src_row = placed[source.device].spec.row_pair
            tgt_rows = {placed[t.device].spec.row_pair for t in targets}
            max_span = max(abs(r - src_row) for r in tgt_rows)

            connections.append(_CrossRowConnection(
                net=net_name,
                source=source,
                targets=targets,
                src_row=src_row,
                max_span=max_span,
            ))

        # Sort by span (longest first) — long spans get met2 priority
        connections.sort(key=lambda c: c.max_span, reverse=True)

        # Track allocator
        met1_w = self.rules.met1.get("width_min_um", 0.14)
        met1_sp = self.rules.met1.get("spacing_min_um", 0.14)
        _met2 = self.rules.met2 if self.rules.met2 else self.rules.met1
        met2_w = _met2.get("width_min_um", 0.14)
        met2_sp = _met2.get("spacing_min_um", 0.14)
        _via1 = self.rules.via1 if self.rules.via1 else {}
        via1_sz = _via1.get("size_um", 0.15)
        enc_m2_v1 = _met2.get("enclosure_of_via1_2adj_um", 0.085)
        landing_half = via1_sz / 2 + enc_m2_v1

        # Cell X extent
        cell_x0 = min(d.x for d in placed.values())
        cell_x1 = max(d.x + d.geom.total_x_um for d in placed.values())

        for conn in connections:
            src_cx = _terminal_cx(conn.source, placed, self.rules)
            src_cy = _terminal_cy(conn.source, placed, self.rules)
            tgt_cy_min = min(_terminal_cy(t, placed, self.rules) for t in conn.targets)
            tgt_cy_max = max(_terminal_cy(t, placed, self.rules) for t in conn.targets)

            y_min = min(src_cy, tgt_cy_min)
            y_max = max(src_cy, tgt_cy_max)

            # Decide metal level: adjacent rows → met1, else → met2
            if conn.max_span <= 1:
                via_level = 1
                track_x = src_cx  # use source X as track
            else:
                via_level = 2
                # Find best track_x: try source X first, then allocate
                track_x = _allocate_track(
                    src_cx, y_min, y_max,
                    landing_half, met2_w / 2, met2_sp,
                    cell_x0, cell_x1,
                    allocated_tracks, via_level,
                )
                allocated_tracks.append(_TrackAllocation(
                    x=track_x, y_min=y_min, y_max=y_max, level=via_level,
                ))

            path = [conn.source.ref] + [t.ref for t in conn.targets]
            extra: dict = {"via_level": via_level}
            if via_level >= 2:
                extra["track_x"] = round(track_x, 3)

            layer = "met2" if via_level >= 2 else "met1"
            specs.append(RoutingSpec(
                net=conn.net,
                style="cross_row_connect",
                layer=layer,
                path=path,
                extra=extra,
            ))

        return specs

    # ── Phase C: vertical buses ───────────────────────────────────────────

    def _phase_c_vertical_buses(
        self,
        ng:          NetGraph,
        placed:      dict[str, PlacedDevice],
        row_devices: dict[int, set[str]],
    ) -> list[RoutingSpec]:
        """Generate vertical_bus specs for S/D nets spanning 3+ row pairs.

        Buses like BL/BL_ need vertical metal runs connecting multiple
        S/D terminals across the cell height.
        """
        if not row_devices or max(row_devices.keys(), default=-1) < 0:
            return []

        specs: list[RoutingSpec] = []

        for net_name, info in ng.nets.items():
            if info.is_power:
                continue

            # Find S/D terminals across different row pairs
            sd_terms = [t for t in info.sd_terminals if t.device in placed]
            if len(sd_terms) < 2:
                continue

            rows_with_sd = {placed[t.device].spec.row_pair for t in sd_terms}
            if len(rows_with_sd) < 2:
                continue

            # Skip if this net already has cross_row_connect coverage
            # (Phase B handles nets that drive gates across rows)
            has_cross_row_gates = any(
                placed[t.device].spec.row_pair != placed[sd_terms[0].device].spec.row_pair
                for t in info.gate_terminals
                if t.device in placed
            )
            if has_cross_row_gates:
                continue  # Phase B handles this

            # This is a bus net (like BL/BL_): S/D terminals only across rows
            path = [t.ref for t in sd_terms]

            # Compute bus_x from terminal centroid
            tap_xs = [_terminal_cx(t, placed, self.rules) for t in sd_terms]
            bus_x = sum(tap_xs) / len(tap_xs)

            met1_w = self.rules.met1.get("width_min_um", 0.14)
            extra: dict = {"bus_x": round(bus_x, 3)}

            specs.append(RoutingSpec(
                net=net_name,
                style="vertical_bus",
                layer="met1",
                path=path,
                extra=extra,
            ))

        return specs

    # ── Phase D: power rails ──────────────────────────────────────────────

    def _phase_d_power_rails(self, ng: NetGraph) -> list[RoutingSpec]:
        """Generate horizontal_power_rail specs for power nets."""
        specs: list[RoutingSpec] = []

        for net_name, info in ng.nets.items():
            if not info.is_power:
                continue

            edge = ""
            if info.rail == "top":
                edge = "top"
            elif info.rail == "bottom":
                edge = "bottom"
            elif net_name in ("VDD",):
                edge = "top"
            elif net_name in ("VSS", "GND"):
                edge = "bottom"
            else:
                continue  # unknown power net without rail spec

            specs.append(RoutingSpec(
                net=net_name,
                style="horizontal_power_rail",
                layer="met1",
                edge=edge,
            ))

        return specs


# ── Track allocation helpers ──────────────────────────────────────────────────

@dataclass
class _TrackAllocation:
    """A reserved vertical track on a specific metal level."""
    x:     float
    y_min: float
    y_max: float
    level: int       # 1=met1, 2=met2


@dataclass
class _CrossRowConnection:
    """A cross-row connection to be routed."""
    net:      str
    source:   TerminalRef
    targets:  list[TerminalRef]
    src_row:  int
    max_span: int


def _allocate_track(
    preferred_x: float,
    y_min: float,
    y_max: float,
    landing_half: float,
    wire_half: float,
    spacing: float,
    cell_x0: float,
    cell_x1: float,
    existing: list[_TrackAllocation],
    level: int,
) -> float:
    """Find a track X position that doesn't conflict with existing tracks.

    Tries *preferred_x* first, then offsets in both directions.
    """
    # Check if preferred_x works
    if _track_is_clear(preferred_x, y_min, y_max,
                       landing_half, wire_half, spacing,
                       existing, level):
        return preferred_x

    # Try offsets in both directions
    pitch = landing_half + spacing + wire_half
    for i in range(1, 20):
        for sign in (+1, -1):
            candidate = preferred_x + sign * i * pitch
            if candidate < cell_x0 or candidate > cell_x1:
                continue
            if _track_is_clear(candidate, y_min, y_max,
                               landing_half, wire_half, spacing,
                               existing, level):
                return candidate

    # Fallback: just use preferred
    warnings.warn(
        f"Track allocator: could not find clear track near x={preferred_x:.3f}; "
        f"using preferred position (may cause shorts).",
        stacklevel=3,
    )
    return preferred_x


def _track_is_clear(
    x: float,
    y_min: float,
    y_max: float,
    landing_half: float,
    wire_half: float,
    spacing: float,
    existing: list[_TrackAllocation],
    level: int,
) -> bool:
    """Check if a track at *x* spanning [y_min, y_max] is clear of conflicts."""
    for alloc in existing:
        if alloc.level != level:
            continue
        # Check Y overlap
        if y_max <= alloc.y_min or y_min >= alloc.y_max:
            continue  # no Y overlap, tracks don't conflict
        # Y overlaps — check X separation
        # Need spacing between landing pads
        min_sep = landing_half + spacing + wire_half
        if abs(x - alloc.x) < min_sep:
            return False
    return True


def _terminal_cx(
    tref:   TerminalRef,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> float:
    """X center of a terminal."""
    t = resolve_terminal(tref.ref, placed, rules)
    return (t.x0 + t.x1) / 2


def _terminal_cy(
    tref:   TerminalRef,
    placed: dict[str, PlacedDevice],
    rules:  PDKRules,
) -> float:
    """Y center of a terminal."""
    t = resolve_terminal(tref.ref, placed, rules)
    return (t.y0 + t.y1) / 2


# ── Row-device map ────────────────────────────────────────────────────────────

def _build_row_device_map(
    placed: dict[str, PlacedDevice],
) -> dict[int, set[str]]:
    """Build map: row_pair_id → set of device names."""
    m: dict[int, set[str]] = defaultdict(set)
    for dev_name, dev in placed.items():
        m[dev.spec.row_pair].add(dev_name)
    return dict(m)
