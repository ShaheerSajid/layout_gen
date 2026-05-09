"""
layout_gen.lvs — tool-agnostic LVS (Layout-vs-Schematic) framework.

Quick start::

    from layout_gen          import load_pdk
    from layout_gen.synth    import load_template
    from layout_gen.lvs      import build_reference_netlist, run_lvs

    rules = load_pdk()
    tmpl  = load_template("inverter")
    spice = build_reference_netlist(tmpl, rules, {"w_N": 0.52, "w_P": 0.42, "l": 0.15})

    # After writing the synthesized GDS:
    result = run_lvs(gds_path, spice, "cmos_inverter", rules)
    print("clean" if result.clean else result.mismatches)
"""
from __future__ import annotations

from pathlib import Path

from layout_gen.lvs.base    import LVSRunner, LVSResult, LVSMismatch
from layout_gen.lvs.netlist import build_reference_netlist
from layout_gen.lvs         import registry

# Register built-in backends
from layout_gen.lvs.magic_runner import MagicNetgenLVSRunner
registry.register("magic_netgen", MagicNetgenLVSRunner)


def run_lvs(
    gds_path,
    ref_netlist,
    cell_name: str,
    rules,
    tool: str = "magic_netgen",
    **runner_kwargs,
) -> LVSResult:
    """Run LVS on *gds_path* against *ref_netlist*.

    Parameters
    ----------
    gds_path :
        Path to the layout GDS.
    ref_netlist :
        Either a path to a SPICE file or a SPICE string (auto-detected).
    cell_name :
        Top-cell name (must match the ``.subckt`` in *ref_netlist*).
    rules :
        :class:`~layout_gen.pdk.PDKRules`.
    tool :
        Backend identifier — currently ``"magic_netgen"``.
    """
    import tempfile

    runner = registry.get(tool, rules=rules, **runner_kwargs)

    if isinstance(ref_netlist, (str, Path)) and Path(str(ref_netlist)).is_file():
        ref_path = Path(ref_netlist)
        return runner.run(Path(gds_path), ref_path, cell_name)

    # Treat as SPICE text — write to a tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".spice", delete=False, encoding="utf-8",
    ) as f:
        f.write(str(ref_netlist))
        tmp = Path(f.name)
    try:
        return runner.run(Path(gds_path), tmp, cell_name)
    finally:
        tmp.unlink(missing_ok=True)


def available_tools() -> list[str]:
    """Tool names with a registered LVS backend."""
    return registry.available()


def get_runner(rules=None) -> "LVSRunner | None":
    """Return the first available LVS runner, or ``None``."""
    if rules is None:
        from layout_gen import load_pdk
        rules = load_pdk()
    for name in registry.available():
        try:
            r = registry.get(name, rules=rules)
            if r.is_available():
                return r
        except Exception:
            continue
    return None


__all__ = [
    "LVSRunner",
    "LVSResult",
    "LVSMismatch",
    "build_reference_netlist",
    "run_lvs",
    "available_tools",
    "get_runner",
    "MagicNetgenLVSRunner",
]
