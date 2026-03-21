"""
layout_gen.synth.constraints — safe symbolic expression evaluator.

Expressions in YAML templates reference PDK rules and device geometry using
dot notation::

    rules.diff.spacing_min_um - 2*rules.poly.endcap_over_diff_um
    N.total_y + inter_cell_gap

This module wraps ``PDKRules`` and ``TransistorGeom`` objects in recursive
attribute-access namespaces and evaluates expressions with a restricted
``eval()`` (``__builtins__`` disabled).
"""
from __future__ import annotations

import math
from typing import Any

from layout_gen.pdk        import PDKRules
from layout_gen.transistor import TransistorGeom


# ── Recursive attribute namespace ─────────────────────────────────────────────

class _NS:
    """Wraps a dict so keys are accessible as attributes, recursively.

    >>> ns = _NS({"poly": {"width_min_um": 0.15}})
    >>> ns.poly.width_min_um
    0.15
    """
    __slots__ = ("_d",)

    def __init__(self, d: dict):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str):
        d = object.__getattribute__(self, "_d")
        if name not in d:
            raise AttributeError(
                f"No attribute {name!r} in namespace {list(d)}"
            )
        val = d[name]
        return _NS(val) if isinstance(val, dict) else val

    def __repr__(self) -> str:
        d = object.__getattribute__(self, "_d")
        return f"_NS({list(d)})"


# ── Namespace construction ─────────────────────────────────────────────────────

def build_namespace(
    rules:  PDKRules,
    geoms:  dict[str, TransistorGeom] | None = None,
    named:  dict[str, float]          | None = None,
) -> dict[str, Any]:
    """Build an eval namespace for constraint expressions.

    Parameters
    ----------
    rules :
        PDK rules.  Exposed as ``rules.poly.width_min_um`` etc.
    geoms :
        Device name → ``TransistorGeom`` map.  Each device is exposed by
        name so ``N.total_y`` resolves to ``geoms["N"].total_y_um``.
        Short aliases (``total_y``, ``total_x``) are added alongside the
        canonical ``_um`` names.
    named :
        Pre-computed scalar constraints (e.g. ``inter_cell_gap = 0.14``).
        Added as plain float names in the namespace.

    Returns
    -------
    dict
        Namespace for ``eval(expr, {"__builtins__": {}}, ns)``.
    """
    rules_ns = _NS({
        "poly":     rules.poly,
        "diff":     rules.diff,
        "contacts": rules.contacts,
        "li1":      rules.li1,
        "met1":     rules.met1 or {},
        "mcon":     rules.mcon or {},
        "nwell":    rules.nwell,
        "implant":  rules.implant,
    })

    ns: dict[str, Any] = {
        "__builtins__": {},
        "rules": rules_ns,
        "math":  math,
        "max":   max,
        "min":   min,
        "abs":   abs,
    }

    if geoms:
        for dev_name, g in geoms.items():
            # Expose both canonical (_um suffix) and short-alias names
            geom_dict = {
                **g.__dict__,
                "total_y": g.total_y_um,
                "total_x": g.total_x_um,
                "sd":      g.sd_length_um,
                "l":       g.l_um,
                "w":       g.w_um,
                "n":       g.n_fingers,
            }
            ns[dev_name] = _NS(geom_dict)

    if named:
        ns.update(named)

    return ns


# ── Expression evaluator ───────────────────────────────────────────────────────

def eval_expr(
    expr:  Any,
    rules: PDKRules,
    geoms: dict[str, TransistorGeom] | None = None,
    named: dict[str, float]          | None = None,
) -> float:
    """Evaluate a symbolic constraint expression.

    Parameters
    ----------
    expr :
        The expression.  If already a number, returned as-is.  If a string,
        evaluated in the constructed namespace.
    rules, geoms, named :
        See :func:`build_namespace`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the expression cannot be evaluated.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    ns = build_namespace(rules, geoms, named)
    try:
        return float(eval(str(expr), ns))
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate constraint expression {expr!r}: {exc}"
        ) from exc


# ── Named constraint resolution ────────────────────────────────────────────────

def resolve_named_constraints(
    named_constraints: dict[str, Any],
    rules:             PDKRules,
    geoms:             dict[str, TransistorGeom],
) -> dict[str, float]:
    """Evaluate all named constraints from a template's ``named_constraints`` dict.

    Named constraints reference only ``rules.*`` and device geometry — not
    each other (to avoid cycles).

    Parameters
    ----------
    named_constraints :
        From :attr:`CellTemplate.named_constraints`.
    rules :
        PDK rules.
    geoms :
        Device geometry objects (for any cross-device references).

    Returns
    -------
    dict[str, float]
        Resolved scalar values, e.g. ``{"inter_cell_gap": 0.14}``.
    """
    resolved: dict[str, float] = {}
    for name, spec in named_constraints.items():
        if isinstance(spec, dict):
            if "min" in spec:
                val = eval_expr(spec["min"], rules, geoms, named={})
                resolved[name] = val
            # other sub-keys (like "note") are ignored
        else:
            resolved[name] = eval_expr(spec, rules, geoms, named={})
    return resolved
