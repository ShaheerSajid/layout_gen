"""
layout_gen.repair.primitives — registry of canonical memory primitives.

A primitive is a logical cell type the repair engine knows about (6T
bitcell, sense amp, decoder, …).  The registry is *declarative* — pure
data describing each primitive's shape, what zones it tends to have,
and which fix skills apply.  No PDK rule values; the registry is fully
PDK-agnostic.

The recogniser (Phase 2) consumes this registry to identify a layout
region as one of these primitives, given the connectivity graph.  The
skill library (Phase 2) consumes it to scope each skill to the
primitives it applies to.

Primitives are intentionally coarse-grained — same registry entry
covers a 6T bitcell at any process node.  Tech-specific dimensions are
the synthesizer's problem; the repair engine reasons about *structure*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ── Net role ─────────────────────────────────────────────────────────────────
#
# Logical role of a net within a primitive.  Lets the recogniser match
# templates whose net names differ across PDKs (VSS vs GND vs VPWR).

NET_ROLES = (
    "vdd", "vss",                # power
    "input", "output",           # combinational data
    "internal",                  # internal state (Q, Q_, intermediate)
    "wordline", "bitline",       # SRAM-specific
    "clock", "scan_in", "scan_enable", "set", "reset",
)


# ── Device role ──────────────────────────────────────────────────────────────

DEVICE_ROLES = (
    "pull_down",   "pull_up",
    "pass_gate",
    "tg_n", "tg_p",              # transmission-gate halves
    "feedback_inv_n", "feedback_inv_p",   # in latches/flops
    "input_inv_n", "input_inv_p",
    "output_inv_n", "output_inv_p",
    "row_decoder_n", "row_decoder_p",
    "precharge",
    "sense_amp_cross_couple_n", "sense_amp_cross_couple_p",
    "sense_amp_isolation",
    "write_driver",
    "tap",                       # well/substrate tap
)


# ── Zone archetype ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneArchetype:
    """A typical fix zone within a primitive.

    Used by the recogniser to predict, given a region of a layout,
    *which* archetypal zone it is — that lets us pick the right skill.
    """
    name:               str          # human-readable: "polycontact_npc"
    likely_rules:       tuple[str, ...]  # rule categories common in this zone
    likely_layers:      tuple[str, ...]  # layer roles involved
    description:        str = ""


# ── Primitive ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Primitive:
    """One canonical memory primitive.

    Attributes
    ----------
    name :
        Identifier, e.g. ``"bitcell_6t"``.
    family :
        Coarse grouping: ``"logic"`` | ``"sequential"`` | ``"memory"`` |
        ``"periphery"``.
    n_devices :
        (min, max) device count.  E.g. NAND2 = (4, 4); buffer = (4, 4);
        AOI21 = (6, 6); 6T bitcell = (6, 6); sense amp = (8, 12) etc.
    nets :
        Required net roles.  The recogniser confirms a candidate region
        contains all of these.
    devices :
        Required device-role multiset.
    zones :
        Common fix-zone archetypes inside this primitive.
    description :
        Free-text intent.
    """
    name:        str
    family:      str
    n_devices:   tuple[int, int]
    nets:        tuple[str, ...]
    devices:     tuple[str, ...]
    zones:       tuple[ZoneArchetype, ...] = field(default_factory=tuple)
    description: str = ""


# ── Common zone archetypes ──────────────────────────────────────────────────
#
# Pulled from inspection of the catalog + the cells we synthesize.  Each
# is tagged with the *category* of rules likely to fire there (not
# specific rule names — those vary across PDKs).

POLYCONTACT_NPC = ZoneArchetype(
    name="polycontact_npc",
    likely_rules=("merge", "spacing_cross", "enclosure"),
    likely_layers=("poly", "licon", "li1", "npc"),
    description="Region around a poly contact (gate licon) where NPC,"
                " licon, poly pad, and li1 pad must coexist.  Most"
                " licon-13 / npc-2 violations appear here.",
)

GATE_DIFF_INTERFACE = ZoneArchetype(
    name="gate_diff_interface",
    likely_rules=("extension", "spacing_same", "enclosure"),
    likely_layers=("poly", "diff", "implant"),
    description="Where a gate finger meets diffusion: poly endcap and"
                " diff-extension-past-gate rules.",
)

POWER_RAIL = ZoneArchetype(
    name="power_rail",
    likely_rules=("width", "area", "spacing_same"),
    likely_layers=("met1", "li1", "tap", "implant"),
    description="VDD/GND rail band along cell edges.",
)

CROSS_COUPLE = ZoneArchetype(
    name="cross_couple",
    likely_rules=("spacing_cross", "spacing_same", "merge"),
    likely_layers=("li1", "met1", "met2", "poly"),
    description="Cross-coupled gate/drain wiring (bitcells, latches,"
                " sense amps).  High wire density, multiple coupled"
                " nets in close proximity.",
)

PASS_GATE_BL = ZoneArchetype(
    name="pass_gate_bl",
    likely_rules=("spacing_cross", "enclosure"),
    likely_layers=("diff", "li1", "met2", "poly"),
    description="Bitline contact at a pass gate (bitcell, write/read"
                " driver).",
)

DIFFUSION_ABUTMENT = ZoneArchetype(
    name="diffusion_abutment",
    likely_rules=("spacing_same", "extension", "implant"),
    likely_layers=("diff", "implant", "nwell"),
    description="Where two same-type devices share an S/D diffusion.",
)


# ── Primitive registry ───────────────────────────────────────────────────────

REGISTRY: dict[str, Primitive] = {}


def _reg(p: Primitive) -> None:
    REGISTRY[p.name] = p


# ── Combinational logic ─────────────────────────────────────────────────────

_reg(Primitive(
    name="inv", family="logic",
    n_devices=(2, 2),
    nets=("input", "output", "vdd", "vss"),
    devices=("pull_down", "pull_up"),
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, POWER_RAIL),
    description="Single-stage inverter.",
))

_reg(Primitive(
    name="buf", family="logic",
    n_devices=(4, 4),
    nets=("input", "output", "vdd", "vss"),
    devices=("pull_down", "pull_up", "pull_down", "pull_up"),
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, POWER_RAIL),
    description="Two-stage non-inverting buffer.",
))

_reg(Primitive(
    name="nand2", family="logic",
    n_devices=(4, 4),
    nets=("input", "input", "output", "vdd", "vss"),
    devices=("pull_down", "pull_down", "pull_up", "pull_up"),
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, DIFFUSION_ABUTMENT, POWER_RAIL),
    description="2-input NAND.  NMOS series + PMOS parallel.",
))

_reg(Primitive(
    name="nand3", family="logic", n_devices=(6, 6),
    nets=("input", "input", "input", "output", "vdd", "vss"),
    devices=("pull_down",) * 3 + ("pull_up",) * 3,
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, DIFFUSION_ABUTMENT, POWER_RAIL),
))

_reg(Primitive(
    name="nor2", family="logic", n_devices=(4, 4),
    nets=("input", "input", "output", "vdd", "vss"),
    devices=("pull_down", "pull_down", "pull_up", "pull_up"),
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, DIFFUSION_ABUTMENT, POWER_RAIL),
    description="2-input NOR.  NMOS parallel + PMOS series.",
))

_reg(Primitive(
    name="nor3", family="logic", n_devices=(6, 6),
    nets=("input", "input", "input", "output", "vdd", "vss"),
    devices=("pull_down",) * 3 + ("pull_up",) * 3,
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, DIFFUSION_ABUTMENT, POWER_RAIL),
))

_reg(Primitive(
    name="aoi21", family="logic", n_devices=(6, 6),
    nets=("input", "input", "input", "output", "vdd", "vss"),
    devices=("pull_down",) * 3 + ("pull_up",) * 3,
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, CROSS_COUPLE, POWER_RAIL),
    description="And-Or-Invert: NOT((A & B) | C).",
))

_reg(Primitive(
    name="oai21", family="logic", n_devices=(6, 6),
    nets=("input", "input", "input", "output", "vdd", "vss"),
    devices=("pull_down",) * 3 + ("pull_up",) * 3,
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, CROSS_COUPLE, POWER_RAIL),
    description="Or-And-Invert: NOT((A | B) & C).",
))

_reg(Primitive(
    name="mux2", family="logic", n_devices=(8, 12),
    nets=("input", "input", "input", "output", "vdd", "vss"),
    devices=("tg_n",) * 2 + ("tg_p",) * 2 + ("pull_down",) + ("pull_up",),
    zones=(POLYCONTACT_NPC, CROSS_COUPLE, POWER_RAIL),
    description="2:1 transmission-gate multiplexer.",
))

_reg(Primitive(
    name="xor2", family="logic", n_devices=(8, 12),
    nets=("input", "input", "output", "vdd", "vss"),
    devices=("tg_n",) * 2 + ("tg_p",) * 2 + ("pull_down",) * 2 + ("pull_up",) * 2,
    zones=(POLYCONTACT_NPC, CROSS_COUPLE, POWER_RAIL),
))

# ── Sequential logic ────────────────────────────────────────────────────────

_reg(Primitive(
    name="dff", family="sequential", n_devices=(16, 24),
    nets=("input", "output", "clock", "vdd", "vss"),
    devices=("tg_n",) * 4 + ("tg_p",) * 4
            + ("feedback_inv_n",) * 2 + ("feedback_inv_p",) * 2
            + ("input_inv_n",) + ("input_inv_p",)
            + ("output_inv_n",) + ("output_inv_p",),
    zones=(POLYCONTACT_NPC, CROSS_COUPLE, POWER_RAIL),
    description="Master-slave D flip-flop (TG-based).",
))

_reg(Primitive(
    name="dlatch", family="sequential", n_devices=(8, 12),
    nets=("input", "output", "clock", "vdd", "vss"),
    devices=("tg_n",) * 2 + ("tg_p",) * 2
            + ("feedback_inv_n",) + ("feedback_inv_p",)
            + ("output_inv_n",) + ("output_inv_p",),
    zones=(POLYCONTACT_NPC, CROSS_COUPLE, POWER_RAIL),
    description="D latch (TG-based).",
))

# ── Memory cells ────────────────────────────────────────────────────────────

_reg(Primitive(
    name="bitcell_6t", family="memory", n_devices=(6, 6),
    nets=("wordline", "bitline", "bitline",
          "internal", "internal", "vdd", "vss"),
    devices=("pull_down",) * 2 + ("pull_up",) * 2 + ("pass_gate",) * 2,
    zones=(POLYCONTACT_NPC, CROSS_COUPLE, PASS_GATE_BL,
           DIFFUSION_ABUTMENT, POWER_RAIL),
    description="6T single-port SRAM bitcell.",
))

# ── Memory periphery ─────────────────────────────────────────────────────────

_reg(Primitive(
    name="sense_amp", family="periphery", n_devices=(8, 14),
    nets=("bitline", "bitline", "output", "output",
          "clock", "vdd", "vss"),
    devices=("sense_amp_cross_couple_n",) * 2
            + ("sense_amp_cross_couple_p",) * 2
            + ("sense_amp_isolation",) * 2
            + ("precharge",) * 2,
    zones=(POLYCONTACT_NPC, CROSS_COUPLE, PASS_GATE_BL, POWER_RAIL),
    description="Latch-style sense amplifier with isolation transistors.",
))

_reg(Primitive(
    name="write_driver", family="periphery", n_devices=(6, 12),
    nets=("input", "input", "bitline", "bitline",
          "clock", "vdd", "vss"),
    devices=("write_driver",) * 4 + ("pull_up", "pull_up", "pull_down", "pull_down"),
    zones=(POLYCONTACT_NPC, PASS_GATE_BL, POWER_RAIL),
))

_reg(Primitive(
    name="precharge", family="periphery", n_devices=(2, 4),
    nets=("bitline", "bitline", "clock", "vdd"),
    devices=("precharge",) * 2 + ("pull_up",) * 2,
    zones=(POLYCONTACT_NPC, POWER_RAIL),
))

_reg(Primitive(
    name="row_driver", family="periphery", n_devices=(6, 12),
    nets=("input", "input", "wordline", "vdd", "vss"),
    devices=("pull_down", "pull_down", "pull_up", "pull_up",
             "row_decoder_n", "row_decoder_p"),
    zones=(POLYCONTACT_NPC, GATE_DIFF_INTERFACE, POWER_RAIL),
    description="NAND2 + drive inverter feeding the wordline.",
))

_reg(Primitive(
    name="col_mux", family="periphery", n_devices=(4, 16),
    nets=("input", "bitline", "bitline", "vdd", "vss"),
    devices=("tg_n",) * 4 + ("tg_p",) * 4,
    zones=(PASS_GATE_BL, POWER_RAIL),
    description="N:1 column multiplexer (typically transmission gates).",
))

# ── Tap / filler ─────────────────────────────────────────────────────────────

_reg(Primitive(
    name="tap_cell", family="memory", n_devices=(0, 0),
    nets=("vdd", "vss"),
    devices=("tap", "tap"),
    zones=(POWER_RAIL,),
    description="Well/substrate tap with no transistors.",
))


# ── Public helpers ───────────────────────────────────────────────────────────

def all_primitives() -> Iterable[Primitive]:
    """Iterate every registered primitive."""
    return REGISTRY.values()


def get(name: str) -> Primitive:
    return REGISTRY[name]


def primitives_by_family(family: str) -> list[Primitive]:
    return [p for p in REGISTRY.values() if p.family == family]


__all__ = [
    "Primitive", "ZoneArchetype",
    "REGISTRY", "all_primitives", "get", "primitives_by_family",
    "NET_ROLES", "DEVICE_ROLES",
    # Common archetypes (re-exported for tests / docs)
    "POLYCONTACT_NPC", "GATE_DIFF_INTERFACE", "POWER_RAIL",
    "CROSS_COUPLE", "PASS_GATE_BL", "DIFFUSION_ABUTMENT",
]
