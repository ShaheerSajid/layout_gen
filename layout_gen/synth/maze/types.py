"""Data types for the maze router."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GridPoint:
    """A point on the 3-D routing grid (layer, x-index, y-index)."""
    layer: int
    gx: int
    gy: int


@dataclass
class RouteSegment:
    """One wire segment on a single layer."""
    layer: str
    x0: float
    y0: float
    x1: float
    y1: float
    width: float


@dataclass
class ViaLocation:
    """Via between two adjacent layers."""
    cx: float
    cy: float
    from_layer: str
    to_layer: str


@dataclass
class NetRoute:
    """Result of routing one net through the maze."""
    net: str
    segments: list[RouteSegment] = field(default_factory=list)
    vias: list[ViaLocation] = field(default_factory=list)


@dataclass
class RoutingProblem:
    """One net to route through the grid."""
    net: str
    sources: list[GridPoint]         # seed cells (already connected)
    targets: list[GridPoint]         # cells to reach
    allowed_layers: list[int] | None = None  # None = all layers
