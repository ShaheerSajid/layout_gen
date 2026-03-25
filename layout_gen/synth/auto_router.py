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
from layout_gen.synth.loader import CellTemplate, RoutingHint, RoutingSpec
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
        hints = template.routing_hints   # dict[str, RoutingHint]

        # Build lookup: row_pair_id → set of device names
        row_devices = _build_row_device_map(placed)

        # Phase A: intra-pair local routing
        specs.extend(self._phase_a_intra_pair(net_graph, placed, row_devices))

        # Phase B: inter-pair cross-row routing
        specs.extend(self._phase_b_inter_pair(net_graph, placed, row_devices))

        # Phase C: vertical buses (multi-row S/D)
        specs.extend(self._phase_c_vertical_buses(net_graph, placed, row_devices, hints))

        # Phase D: power rails
        specs.extend(self._phase_d_power_rails(net_graph, placed, template))

        # Phase F: gate-to-drain connections (cross-couple, stage-to-stage)
        gtd_specs = self._phase_f_gate_to_drain(net_graph, placed, template)
        gtd_nets = {s.net for s in gtd_specs}
        # Suppress drain_bridge on nets that have gate_to_drain (they overlap)
        specs = [s for s in specs
                 if not (s.style == "drain_bridge" and s.net in gtd_nets)]
        specs.extend(gtd_specs)

        # Phase H: hint-driven routing (full_width WL, full_height BL, etc.)
        specs.extend(self._phase_h_hint_routing(net_graph, placed, template, hints))

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

                # Pair each NMOS drain with each PMOS drain on the
                # same net.  Gate-aligned pairs produce straight
                # vertical bridges; offset pairs (e.g. mirrored PMOS)
                # produce L-shapes routed through the NMOS-PMOS gap.
                for nd in nmos_drains:
                    for pd in pmos_drains:
                        key = (net_name, f"drain_{nd.device}_{pd.device}")
                        if key in handled_pairs:
                            continue
                        handled_pairs.add(key)
                        specs.append(RoutingSpec(
                            net=net_name,
                            style="drain_bridge",
                            layer=info.layer or "li1",
                            path=[nd.ref, pd.ref],
                        ))

                # ── Li1 bridge (abutting S/D within same tier) ──
                # Only bridge terminals that share diffusion at an
                # abutment boundary (adjacent S/D strips).  Terminals
                # separated by other devices with different-net S/D
                # would short if bridged on li1 (e.g. PMOS drains
                # separated by VDD sources in a NAND gate).
                for dtype in ("nmos", "pmos"):
                    sd_terms = [t for t in trefs
                                if ng.device_types[t.device] == dtype
                                and t.terminal in ("S", "D")]
                    if len(sd_terms) < 2:
                        continue
                    # Sort by terminal X centre to find truly adjacent pairs
                    sd_terms.sort(
                        key=lambda t: _terminal_cx(t, placed, self.rules))
                    for i in range(len(sd_terms) - 1):
                        a, b = sd_terms[i], sd_terms[i + 1]
                        if a.device == b.device:
                            continue  # skip same device
                        # Only bridge if terminal X centres are within
                        # one S/D length — shared diffusion at abutment
                        a_cx = _terminal_cx(a, placed, self.rules)
                        b_cx = _terminal_cx(b, placed, self.rules)
                        sd_len = placed[a.device].geom.sd_length_um
                        if abs(a_cx - b_cx) > sd_len + 0.01:
                            continue
                        # Skip li1 bridge for shared-diffusion abutments:
                        # terminals at the same X share continuous
                        # diffusion, so no li1 or contacts are needed.
                        if abs(a_cx - b_cx) < 0.05:
                            continue
                        key = (net_name, f"li1_{a.device}_{b.device}")
                        if key in handled_pairs:
                            continue
                        handled_pairs.add(key)
                        specs.append(RoutingSpec(
                            net=net_name,
                            style="li1_bridge",
                            layer=info.layer or "li1",
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
        hints:       dict[str, RoutingHint] | None = None,
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

            extra: dict = {"bus_x": round(bus_x, 3)}

            # Use routing hint layer preference if available
            hint = (hints or {}).get(net_name)
            bus_layer = "met1"
            if hint and hint.layer:
                bus_layer = hint.layer
                if bus_layer in ("met2",):
                    extra["via_level"] = 2

            specs.append(RoutingSpec(
                net=net_name,
                style="vertical_bus",
                layer=bus_layer,
                path=path,
                extra=extra,
            ))

        return specs

    # ── Phase D: power rails ──────────────────────────────────────────────

    def _phase_d_power_rails(
        self,
        ng:       NetGraph,
        placed:   dict[str, PlacedDevice],
        template: CellTemplate,
    ) -> list[RoutingSpec]:
        """Generate horizontal_power_rail specs for power nets.

        For stacked layouts with ``rail_top``/``rail_bottom`` on row pairs,
        emits intermediate rails at the computed Y boundaries between pairs.
        """
        specs: list[RoutingSpec] = []
        emitted_edges: set[str] = set()  # track top/bottom to avoid duplicates

        # ── Fixed cell width (if specified in template) ───────────────
        cell_width = template.cell_dimensions.width if template.cell_dimensions.width > 0 else 0.0

        # ── Intermediate rails from row pair declarations ──────────────
        if template.layout_mode == "stacked" and template.row_pairs:
            # Compute Y-extent of each row pair from placed devices
            rp_bounds: dict[int, tuple[float, float]] = {}  # id → (y_min, y_max)
            for rp in template.row_pairs:
                all_devs = rp.nmos_devices + rp.pmos_devices
                ys = []
                for dname in all_devs:
                    if dname in placed:
                        d = placed[dname]
                        ys.extend([d.y, d.y + d.geom.total_y_um])
                if ys:
                    rp_bounds[rp.id] = (min(ys), max(ys))

            extra_base: dict = {}
            if cell_width > 0:
                extra_base["cell_width"] = cell_width

            for rp in template.row_pairs:
                if rp.id not in rp_bounds:
                    continue
                rp_ymin, rp_ymax = rp_bounds[rp.id]

                if rp.rail_top:
                    top_net = ng.nets.get(rp.rail_top)
                    specs.append(RoutingSpec(
                        net=rp.rail_top,
                        style="horizontal_power_rail",
                        layer=(top_net.layer if top_net and top_net.layer else "met1"),
                        extra={**extra_base, "y_pos": rp_ymax},
                    ))
                    emitted_edges.add(f"{rp.rail_top}_top_{rp.id}")

                if rp.rail_bottom:
                    bot_net = ng.nets.get(rp.rail_bottom)
                    specs.append(RoutingSpec(
                        net=rp.rail_bottom,
                        style="horizontal_power_rail",
                        layer=(bot_net.layer if bot_net and bot_net.layer else "met1"),
                        extra={**extra_base, "y_pos": rp_ymin},
                    ))
                    emitted_edges.add(f"{rp.rail_bottom}_bottom_{rp.id}")

        # ── Outer edge rails (default behaviour) ──────────────────────
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
                continue

            extra_edge: dict = {}
            if cell_width > 0:
                extra_edge["cell_width"] = cell_width
            specs.append(RoutingSpec(
                net=net_name,
                style="horizontal_power_rail",
                layer=info.layer or "met1",
                edge=edge,
                extra=extra_edge,
            ))

            # Connect device source terminals to the rail
            # source_to_rail needs the rail layer to drop vias from li1 up
            rail_lyr = info.layer or "met1"
            terminals = [t.ref for t in info.terminals if t.terminal == "S"]
            if terminals:
                specs.append(RoutingSpec(
                    net=net_name,
                    style="source_to_rail",
                    path=terminals,
                    layer=rail_lyr,
                    edge=edge,
                ))

        return specs


    # ── Phase F: gate-to-drain connections ───────────────────────────────

    def _phase_f_gate_to_drain(
        self,
        ng:       NetGraph,
        placed:   dict[str, PlacedDevice],
        template: CellTemplate,
    ) -> list[RoutingSpec]:
        """Detect nets with both gate and drain terminals on different
        devices and emit ``gate_to_drain`` specs.

        This covers:
        - SRAM cross-couple (gate of one inverter ↔ drain of the other)
        - Stage-to-stage (NAND output drain ↔ inverter gate input)

        When two gate_to_drain routes on the same row pair cross in X,
        they are assigned different layers to avoid shorting.
        """
        specs: list[RoutingSpec] = []
        handled: set[str] = set()

        # For each gate terminal, find the closest same-type drain on
        # the same net (NMOS gate → closest NMOS drain, PMOS gate →
        # closest PMOS drain).  The drain bridge already connects N/P
        # drain pairs vertically, so one gate_to_drain per gate suffices.
        from layout_gen.synth.placer import global_gate_x, global_sd_x
        gtd_pairs: list[tuple[str, TerminalRef, TerminalRef]] = []

        for net_name, info in ng.nets.items():
            if info.is_power:
                continue

            gates  = [t for t in info.terminals if t.terminal == "G"]
            drains = [t for t in info.terminals if t.terminal == "D"]

            for g in gates:
                gdev = placed.get(g.device)
                if gdev is None:
                    continue
                g_type = ng.device_types[g.device]
                gx0, gx1 = global_gate_x(gdev, 0)
                g_cx = (gx0 + gx1) / 2

                # Find closest drain on same net, same type, different device
                best_d = None
                best_dist = float("inf")
                for d in drains:
                    if d.device == g.device:
                        continue
                    ddev = placed.get(d.device)
                    if ddev is None:
                        continue
                    if ng.device_types[d.device] != g_type:
                        continue
                    if gdev.spec.row_pair != ddev.spec.row_pair:
                        continue
                    j_d = 0 if ddev.spec.sd_flip else ddev.geom.n_fingers
                    dx0, dx1 = global_sd_x(ddev, j_d, self.rules)
                    d_cx = (dx0 + dx1) / 2
                    dist = abs(g_cx - d_cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_d = d

                if best_d is not None:
                    key = f"gtd_{net_name}_{g.device}"
                    if key not in handled:
                        handled.add(key)
                        gtd_pairs.append((net_name, g, best_d))

        if not gtd_pairs:
            return specs

        # Detect crossing: two routes cross if their gate and drain X
        # positions are on opposite sides (one goes L→R, the other R→L).
        # Group by row pair and detect crossings.
        from layout_gen.synth.placer import global_gate_x, global_sd_x

        # Build route descriptors with X positions
        routes: list[dict] = []
        for net_name, g, d in gtd_pairs:
            gdev = placed[g.device]
            ddev = placed[d.device]
            gx0, gx1 = global_gate_x(gdev, 0)
            gate_cx = (gx0 + gx1) / 2
            j_d = 0 if ddev.spec.sd_flip else ddev.geom.n_fingers
            dx0, dx1 = global_sd_x(ddev, j_d, self.rules)
            drain_cx = (dx0 + dx1) / 2
            row_pair = gdev.spec.row_pair
            routes.append({
                "net": net_name, "gate": g, "drain": d,
                "gate_cx": gate_cx, "drain_cx": drain_cx,
                "row_pair": row_pair, "layer_idx": 0,
            })

        # Assign layers: check each pair of routes for crossing
        # Build layer stack: li1 → met1 → met2 ...
        layer_stack = ["li1"]
        try:
            transitions = self.rules.via_stack_between("li1", "met2")
            for t in transitions:
                layer_stack.append(t.upper_metal)
        except (KeyError, IndexError, AttributeError):
            layer_stack.append("met1")

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                ri, rj = routes[i], routes[j]
                if ri["row_pair"] != rj["row_pair"]:
                    continue
                # Routes cross if one goes left→right and other right→left
                dir_i = ri["drain_cx"] - ri["gate_cx"]
                dir_j = rj["drain_cx"] - rj["gate_cx"]
                if dir_i * dir_j < 0:  # opposite directions = crossing
                    rj["layer_idx"] = max(rj["layer_idx"], ri["layer_idx"] + 1)

        # Emit specs — prefer net-level layer if set, else use crossing logic
        for r in routes:
            net_layer = ng.nets[r["net"]].layer
            if net_layer:
                route_lyr = net_layer
            else:
                idx = min(r["layer_idx"], len(layer_stack) - 1)
                route_lyr = layer_stack[idx]
            specs.append(RoutingSpec(
                net=r["net"],
                style="gate_to_drain",
                layer=route_lyr,
                path=[r["gate"].ref, r["drain"].ref],
            ))

        return specs


    # ── Phase H: hint-driven routing ─────────────────────────────────────

    def _phase_h_hint_routing(
        self,
        ng:       NetGraph,
        placed:   dict[str, PlacedDevice],
        template: CellTemplate,
        hints:    dict[str, RoutingHint],
    ) -> list[RoutingSpec]:
        """Generate routing specs from explicit per-net hints in the template.

        Handles:
        - ``style: full_width`` → WL-style met1 bus spanning cell width
          (via ``poly_stub_met1_bus`` style handler)
        - ``style: full_height`` → BL-style met2 bus spanning cell height
          (via ``expose_terminal`` + vertical bus on specified layer)
        """
        if not hints:
            return []

        specs: list[RoutingSpec] = []

        for net_name, hint in hints.items():
            info = ng.nets.get(net_name)
            if info is None:
                continue

            # Cell bounds for full-span routing
            cell_x0 = min(d.x for d in placed.values())
            cell_x1 = max(d.x + d.geom.total_x_um for d in placed.values())
            cell_y0 = min(d.y for d in placed.values())
            cell_y1 = max(d.y + d.geom.total_y_um for d in placed.values())
            cell_w  = template.cell_dimensions.width if template.cell_dimensions.width else 0
            if cell_w > 0:
                cx = (cell_x0 + cell_x1) / 2
                cell_x0 = cx - cell_w / 2
                cell_x1 = cx + cell_w / 2

            if hint.style == "full_width":
                # WL-style: poly-contact stub + met1 horizontal bus
                # Find gate terminals on this net
                gate_refs = [t.ref for t in info.gate_terminals
                             if t.device in placed]
                if gate_refs:
                    specs.append(RoutingSpec(
                        net=net_name,
                        style="poly_stub_met1_bus",
                        layer=hint.layer or "met1",
                        path=gate_refs,
                        extra={"cell_x0": cell_x0, "cell_x1": cell_x1},
                    ))

            elif hint.style == "full_height":
                # BL-style: full-height met2 stripe at S/D terminal
                sd_refs = [t.ref for t in info.sd_terminals
                           if t.device in placed]
                for ref in sd_refs:
                    layer = hint.layer or "met2"
                    specs.append(RoutingSpec(
                        net=net_name,
                        style="vertical_met2_bus",
                        layer=layer,
                        path=[ref],
                        extra={
                            "cell_y0": cell_y0, "cell_y1": cell_y1,
                        },
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
