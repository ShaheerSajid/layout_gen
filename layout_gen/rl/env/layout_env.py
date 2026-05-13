"""
layout_gen.rl.env.layout_env — gymnasium.Env wrapping layout repair + place.

Phase 1 implements the **repair** episode: reset to a (broken) layout,
let the policy issue per-step edits, terminate when DRC-clean.

Phase 4 part 2 adds the **place** episode: reset to an *empty* layout
plus a topology graph, let the policy place each device once, then
transition into REPAIR until DRC-clean.

Episode lifecycle (REPAIR-only)
-------------------------------
1. ``reset(options={"state": LayoutState, ...})`` — env loads a
   ready-made (typically broken) layout.
2. ``step(action)`` decodes the action, applies the perturb primitive,
   re-runs DRC, computes reward.
3. ``terminated=True`` on DRC-clean; ``truncated=True`` on max_steps.

Episode lifecycle (PLACE → REPAIR)
----------------------------------
1. ``reset()`` starts with an empty :class:`LayoutState` and the env's
   bound :class:`TopologyGraph`. Phase = ``"place"``.
2. Each ``step`` consumes a PLACE action; the env materialises the
   chosen device via :func:`place_device` and marks that device done.
3. When all devices have been placed (or PLACE phase truncated by a
   ``max_place_steps`` budget), the env transitions to phase
   ``"repair"`` and the policy plays repair primitives.
4. ``terminated=True`` on DRC-clean post-repair.

Action masking
--------------
``info["action_mask"]`` is the flat boolean mask consumed by
sb3-contrib MaskablePPO. The mask is **phase-aware**: PLACE-phase masks
suppress all REPAIR kinds (and vice versa), and the device dim is
limited to topology-known + not-yet-placed devices.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from layout_gen.repair import perturb as perturb_lib
from layout_gen.synth.geo.state import LayoutState

from layout_gen.rl.env.action_space import (
    ActionSpace, EnvAction, action_mask_for,
)
from layout_gen.rl.env.observation  import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP,
    build_observation, make_observation_space,
)
from layout_gen.rl.env.place_action import (
    TransistorCache, place_device_full,
)
from layout_gen.rl.env.connectivity import (
    compute_connectivity_score, compute_electrical_score,
    compute_hpwl_score, compute_short_count,
)
from layout_gen.rl.env.placement_intent import (
    compute_row_score, score_alignment,
)
from layout_gen.rl.env.reward       import (
    RewardConfig, RewardBreakdown, compute_reward,
)
from layout_gen.rl.env.route_action import add_route_segment
from layout_gen.rl.env.runner       import CachedDRC, geometry_key
from layout_gen.rl.topology.parser  import TopologyGraph

from layout_gen.rl.env.action_space import (
    DEFAULT_CELL_HEIGHT_UM, DEFAULT_CELL_WIDTH_UM,
    DEFAULT_DEVICE_CAP, DEFAULT_NET_CAP, DEFAULT_POSITION_BINS,
)
from layout_gen.rl.env.route_action import DEFAULT_SIZE_BINS


@dataclass
class EpisodeConfig:
    """Per-reset configuration. Pass via ``options=`` in :meth:`reset`."""
    state:         LayoutState | None = None
    state_factory: Callable[[], LayoutState] | None = None
    cell_bbox:     tuple[float, float, float, float] | None = None
    forbid_kinds:  frozenset[str] = frozenset()
    start_phase:   str | None = None    # override env's default start phase


class LayoutEnv(gym.Env):
    """Gymnasium env for DRC-repair (and, optionally, place) RL.

    Parameters
    ----------
    drc :
        :class:`CachedDRC` (or any object with
        ``run(state) -> list[DRCViolation]``).
    enable_place :
        Turn on the PLACE action vocabulary. Required when the policy
        will issue ``place_device`` actions. Default False keeps the
        Phase 1–3 action space.
    topology_graph :
        Required when ``enable_place=True`` — provides the device list
        the policy is choosing from.
    transistor_cache :
        Required when ``enable_place=True`` — caches the rect templates
        produced by :func:`layout_gen.transistor.draw_transistor`.
    cell_width_um, cell_height_um :
        Cell bbox used to map PLACE x/y bins to µm coordinates.
    max_place_steps :
        PLACE-phase truncation budget. ``0`` = no separate budget; the
        phase still ends when every device has been placed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        drc: CachedDRC,
        *,
        poly_cap:      int = DEFAULT_POLY_CAP,
        viol_cap:      int = DEFAULT_VIOL_CAP,
        target_cap:    int | None = None,
        mag_bins:      int = 16,
        max_steps:     int = 32,
        reward_config: RewardConfig | None = None,
        default_state_factory: Callable[[], LayoutState] | None = None,
        default_cell_bbox: tuple[float, float, float, float] | None = None,
        topology_global: np.ndarray | None = None,
        # ── PLACE-phase additions ──────────────────────────────────────
        enable_place:     bool = False,
        topology_graph:   TopologyGraph | None = None,
        transistor_cache: TransistorCache | None = None,
        device_cap:       int   = DEFAULT_DEVICE_CAP,
        x_bins:           int   = DEFAULT_POSITION_BINS,
        y_bins:           int   = DEFAULT_POSITION_BINS,
        cell_width_um:    float = DEFAULT_CELL_WIDTH_UM,
        cell_height_um:   float = DEFAULT_CELL_HEIGHT_UM,
        max_place_steps:  int   = 0,
        start_phase:      str   = "auto",
        # ── ROUTE-phase additions ──────────────────────────────────────
        enable_route:     bool = False,
        net_cap:          int  = DEFAULT_NET_CAP,
        route_x_bins:     int  = DEFAULT_POSITION_BINS,
        route_y_bins:     int  = DEFAULT_POSITION_BINS,
        route_w_bins:     int  = DEFAULT_SIZE_BINS,
        route_h_bins:     int  = DEFAULT_SIZE_BINS,
        max_route_steps:  int  = 0,
        # ── Placement-intent reward ────────────────────────────────────
        placement_directives: list | None = None,
        # ── Pitch quantisation (track-aligned action space) ───────────
        poly_pitch_um:    float | None = None,
        metal_pitch_um_per_layer: dict[str, float] | None = None,
        metal_direction_per_layer: dict[str, str] | None = None,
        # ── LVS reward (truth signal via magic + netgen) ──────────────
        lvs = None,   # CachedLVS-shaped: .run(state) → LVSResult
        # ── Wiremask-style proximity channel ──────────────────────────
        proximity_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._drc = drc
        self._lvs = lvs
        self.poly_cap   = poly_cap
        self.viol_cap   = viol_cap
        self.target_cap = target_cap or poly_cap
        self.max_steps  = max_steps
        self.reward_cfg = reward_config or RewardConfig()
        self._default_state_factory = default_state_factory
        self._default_cell_bbox     = default_cell_bbox

        # ── Topology conditioning ─────────────────────────────────────
        self._topology_global: np.ndarray | None = None
        topology_dim_for_space: int | None = None
        if topology_global is not None:
            self._topology_global = np.asarray(
                topology_global, dtype=np.float32,
            ).reshape(-1)
            topology_dim_for_space = int(self._topology_global.shape[0])

        # ── PLACE phase wiring ────────────────────────────────────────
        self.enable_place    = enable_place
        self._topology_graph = topology_graph
        self._tx_cache       = transistor_cache
        self.device_cap      = device_cap
        self.x_bins          = x_bins
        self.y_bins          = y_bins
        self.cell_width_um   = cell_width_um
        self.cell_height_um  = cell_height_um
        self.max_place_steps = max_place_steps

        if enable_place:
            if topology_graph is None or transistor_cache is None:
                raise ValueError(
                    "enable_place=True requires both topology_graph and "
                    "transistor_cache."
                )
            if topology_graph.n_devices > device_cap:
                raise ValueError(
                    f"topology has {topology_graph.n_devices} devices but "
                    f"device_cap is {device_cap}."
                )

        if enable_route:
            if topology_graph is None:
                raise ValueError(
                    "enable_route=True requires topology_graph (need the "
                    "net list to address)."
                )
            if topology_graph.n_nets > net_cap:
                raise ValueError(
                    f"topology has {topology_graph.n_nets} nets but "
                    f"net_cap is {net_cap}."
                )

        # ── ROUTE phase wiring (cache config used in mask + decode) ────
        self.enable_route   = enable_route
        self.net_cap        = net_cap
        self.route_x_bins   = route_x_bins
        self.route_y_bins   = route_y_bins
        self.route_w_bins   = route_w_bins
        self.route_h_bins   = route_h_bins
        self.max_route_steps = max_route_steps

        if start_phase not in ("auto", "place", "route", "repair"):
            raise ValueError(
                f"start_phase must be auto/place/route/repair, "
                f"got {start_phase!r}"
            )
        if start_phase == "auto":
            if enable_place:
                start_phase = "place"
            elif enable_route:
                start_phase = "route"
            else:
                start_phase = "repair"
        if start_phase == "place" and not enable_place:
            raise ValueError("start_phase='place' requires enable_place=True.")
        if start_phase == "route" and not enable_route:
            raise ValueError("start_phase='route' requires enable_route=True.")
        self._default_start_phase = start_phase

        # ── Action / observation spaces ───────────────────────────────
        self._action_helper = ActionSpace(
            target_cap=self.target_cap,
            mag_bins=mag_bins,
            enable_place=enable_place,
            device_cap=device_cap,
            x_bins=x_bins,
            y_bins=y_bins,
            cell_width_um=cell_width_um,
            cell_height_um=cell_height_um,
            poly_pitch_um=poly_pitch_um,
            metal_pitch_um_per_layer=metal_pitch_um_per_layer,
            metal_direction_per_layer=metal_direction_per_layer,
            enable_route=enable_route,
            net_cap=net_cap,
            route_x_bins=route_x_bins,
            route_y_bins=route_y_bins,
            route_w_bins=route_w_bins,
            route_h_bins=route_h_bins,
        )
        self.action_space      = self._action_helper.gym_space
        self._proximity_shape  = proximity_shape
        self.observation_space = make_observation_space(
            poly_cap=poly_cap, viol_cap=viol_cap,
            topology_dim=topology_dim_for_space,
            proximity_shape=proximity_shape,
        )

        # ── Mutable per-episode state ─────────────────────────────────
        self._state:        LayoutState | None = None
        self._cell_bbox:    tuple[float, float, float, float] | None = None
        self._violations:   list = []
        self._step_count:   int  = 0
        self._place_step_count: int = 0
        self._last_obs:     dict | None = None
        self._last_rid_map: dict[int, int] = {}
        self._forbid_kinds: frozenset[str] = frozenset()
        self._phase: str = self._default_start_phase
        self._placed_mask:  np.ndarray = np.zeros(device_cap, dtype=bool)
        # Per-device placement origins (x_um, y_um) populated by
        # _apply_place. Used to reject placements that collide with an
        # already-placed device — closes the "stack two devices at the
        # same X bin" failure mode that the trained multi-cell policy
        # falls into otherwise.
        self._placed_origins: dict[int, tuple[float, float]] = {}
        # Minimum distance between two placed devices' origins, in µm.
        # Smaller than the smallest transistor width we emit (~0.7 µm
        # for w=0.5 nmos) so adjacent gate-aligned devices in a real
        # bitcell are not falsely flagged.
        self._origin_separation_um: float = 0.20
        self._route_step_count: int = 0
        self._routed_mask: np.ndarray = np.zeros(net_cap, dtype=bool)
        # Per-terminal global positions populated by PLACE actions:
        #   {(device_idx, terminal_name): (x_um, y_um, layer)}
        # Used by the connectivity reward to score whether route segments
        # actually touch the terminals of the net they claim.
        self._terminals: dict[tuple[int, str],
                              tuple[float, float, str]] = {}
        # YAML placement_logic directives — drive the alignment_delta
        # reward term so PLACE learns the cell's intended axis (e.g.
        # gates aligned vertically for an inverter).
        self._placement_directives: list = list(placement_directives or [])

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self,
              *,
              seed:    int | None = None,
              options: dict[str, Any] | EpisodeConfig | None = None,
              ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        cfg = self._coerce_options(options)
        start_phase = cfg.start_phase or self._default_start_phase
        if start_phase == "place" and not self.enable_place:
            raise ValueError("Cannot start in PLACE phase: enable_place=False.")
        if start_phase == "route" and not self.enable_route:
            raise ValueError("Cannot start in ROUTE phase: enable_route=False.")
        self._phase = start_phase
        self._placed_mask = np.zeros(self.device_cap, dtype=bool)
        self._placed_origins = {}
        self._routed_mask = np.zeros(self.net_cap, dtype=bool)
        self._place_step_count = 0
        self._route_step_count = 0
        self._terminals = {}

        if self._phase in ("place", "route"):
            # Generative episodes start with an empty layout. PLACE
            # populates devices; ROUTE populates wires. (When the env
            # was configured route-only, the caller is expected to
            # have pre-placed devices via options['state'].)
            if self._phase == "place":
                self._state = LayoutState()
            elif cfg.state is not None:
                self._state = deepcopy(cfg.state)
            elif cfg.state_factory is not None:
                self._state = cfg.state_factory()
            elif self._default_state_factory is not None:
                self._state = self._default_state_factory()
            else:
                self._state = LayoutState()
        elif cfg.state is not None:
            self._state = deepcopy(cfg.state)
        elif cfg.state_factory is not None:
            self._state = cfg.state_factory()
        elif self._default_state_factory is not None:
            self._state = self._default_state_factory()
        else:
            raise ValueError(
                "LayoutEnv.reset (REPAIR phase) requires options['state'], "
                "options['state_factory'], or default_state_factory passed "
                "to __init__."
            )

        self._cell_bbox    = cfg.cell_bbox or self._default_cell_bbox
        self._forbid_kinds = cfg.forbid_kinds
        self._step_count   = 0

        self._violations = list(self._drc.run(self._state))
        obs, info = self._build_step_output()
        return obs, info

    def step(self,
             action: np.ndarray | list[int] | tuple[int, ...]
             ) -> tuple[dict, float, bool, bool, dict]:
        if self._state is None:
            raise RuntimeError("LayoutEnv.step called before reset().")

        self._step_count += 1
        before = self._violations
        connectivity_before = self._connectivity_score()
        alignment_before    = self._alignment_score()
        electrical_before   = self._electrical_score()
        hpwl_before         = self._hpwl_score()
        row_before          = self._row_score()
        short_before        = compute_short_count(self._state)
        lvs_before = self._lvs_mismatch_count()

        env_action = self._action_helper.decode(action, self._last_rid_map)

        action_valid = False
        state_changed = False
        before_key = geometry_key(self._state)

        if env_action.is_place and self._phase == "place":
            action_valid = self._apply_place(env_action)
            self._place_step_count += 1
            if action_valid:
                state_changed = geometry_key(self._state) != before_key
        elif env_action.is_route and self._phase == "route":
            action_valid = self._apply_route(env_action)
            self._route_step_count += 1
            if action_valid:
                state_changed = geometry_key(self._state) != before_key
        elif (
            (not env_action.is_place) and (not env_action.is_route)
            and self._phase == "repair"
        ):
            perturb_action = self._action_helper.to_perturb(env_action)
            action_valid = perturb_action is not None
            if action_valid:
                try:
                    perturb_lib.apply(self._state, perturb_action)
                except Exception:
                    action_valid = False
                else:
                    state_changed = geometry_key(self._state) != before_key
        # else: action kind doesn't match current phase → action_valid
        # stays False, state unchanged. Reward will apply the invalid
        # penalty.

        # Re-run DRC. PLACE-phase DRC matters too: placing a device may
        # introduce immediate violations (e.g. overlapping with prior
        # placements), and the reward signal should pick that up.
        after = list(self._drc.run(self._state))
        self._violations = after

        connectivity_after = self._connectivity_score()
        alignment_after    = self._alignment_score()
        electrical_after   = self._electrical_score()
        hpwl_after         = self._hpwl_score()
        row_after          = self._row_score()
        short_after        = compute_short_count(self._state)
        lvs_after = self._lvs_mismatch_count()

        rb = compute_reward(
            violations_before=before,
            violations_after=after,
            state_changed=state_changed,
            action_valid=action_valid,
            phase=self._phase,
            config=self.reward_cfg,
            connectivity_before=connectivity_before,
            connectivity_after=connectivity_after,
            alignment_before=alignment_before,
            alignment_after=alignment_after,
            electrical_before=electrical_before,
            electrical_after=electrical_after,
            hpwl_before=hpwl_before,
            hpwl_after=hpwl_after,
            row_before=row_before,
            row_after=row_after,
            short_before=short_before,
            short_after=short_after,
            lvs_mismatches_before=lvs_before,
            lvs_mismatches_after=lvs_after,
        )

        # Phase transitions: PLACE → ROUTE (or → REPAIR if route disabled),
        # then ROUTE → REPAIR. A phase ends when its work is done OR its
        # per-phase budget is exhausted.
        phase_transitioned = False
        if self._phase == "place":
            if self._all_devices_placed() or (
                self.max_place_steps > 0
                and self._place_step_count >= self.max_place_steps
            ):
                self._phase = "route" if self.enable_route else "repair"
                phase_transitioned = True

        if self._phase == "route":
            # Routing phase has no "all-nets-routed" oracle in v1
            # (LVS check would be needed); phase ends only on the
            # explicit step budget. If max_route_steps==0 the user
            # intended ROUTE-only experimentation — leave it open.
            if (
                self.max_route_steps > 0
                and self._route_step_count >= self.max_route_steps
            ):
                self._phase = "repair"
                phase_transitioned = True

        # Termination: DRC-clean only counts once we're in REPAIR phase
        # (otherwise empty / partial layouts trivially satisfy DRC).
        terminated = (
            self._phase == "repair"
            and len(after) == 0
            and len(before) > 0
        )
        truncated = (self.max_steps > 0 and self._step_count >= self.max_steps)

        obs, info = self._build_step_output(
            reward_breakdown=rb,
            env_action=env_action,
            action_valid=action_valid,
            state_changed=state_changed,
            phase_transitioned=phase_transitioned,
        )
        return obs, rb.total, terminated, truncated, info

    # ── PLACE helpers ────────────────────────────────────────────────────────

    def _apply_place(self, env_action: EnvAction) -> bool:
        """Materialise the chosen device. Returns True iff valid."""
        if self._topology_graph is None or self._tx_cache is None:
            return False
        d_idx = env_action.device_idx
        if d_idx < 0 or d_idx >= self._topology_graph.n_devices:
            return False
        if d_idx >= self.device_cap or self._placed_mask[d_idx]:
            return False

        device = self._topology_graph.devices[d_idx]
        # Sizing fallback: a device with zero w/l shouldn't be sent to
        # draw_transistor — treat as invalid so the reward penalises it.
        if device.w_um <= 0 or device.l_um <= 0:
            return False

        # No-stacking guard: a placement within ε of an already-placed
        # device's origin is rejected (treated as invalid). Stops the
        # multi-cell failure mode where the trained policy puts both
        # NMOSes of a NAND2 at the same (x_bin, y_bin), which the old
        # inspector hid because the merged cluster still had every
        # expected layer. The reward layer charges the invalid-action
        # penalty so PPO learns to avoid these collisions.
        nx, ny = float(env_action.x_um), float(env_action.y_um)
        eps = self._origin_separation_um
        for (ox, oy) in self._placed_origins.values():
            if abs(nx - ox) < eps and abs(ny - oy) < eps:
                return False

        try:
            _, ports = place_device_full(
                self._state, device,
                x_um=nx, y_um=ny,
                orientation=env_action.orientation,
                cache=self._tx_cache,
            )
        except Exception:
            return False
        self._placed_mask[d_idx] = True
        self._placed_origins[d_idx] = (nx, ny)
        # Record terminal global positions for the connectivity reward.
        for term_name, (px, py, layer) in ports.items():
            self._terminals[(d_idx, term_name)] = (px, py, layer)
        return True

    def _connectivity_score(self) -> float:
        """Per-net connectivity score against the cached topology.

        Returns 0 when the env has no topology graph (e.g. REPAIR-only
        runs); otherwise sums the per-net "fraction of terminals
        touched by a wire on the right net" — bounded by
        ``topology.n_nets``.
        """
        if self._topology_graph is None:
            return 0.0
        return compute_connectivity_score(
            self._state, self._topology_graph, self._terminals,
        )

    def _electrical_score(self) -> float:
        """Transitive per-net electrical connectivity (union-find).
        Stricter than ``_connectivity_score`` — only counts nets whose
        terminals are all in one connected component."""
        if self._topology_graph is None:
            return 0.0
        return compute_electrical_score(
            self._state, self._topology_graph, self._terminals,
        )

    def _alignment_score(self) -> float:
        """Sum-of-clipped-linears score against the YAML's
        ``placement_logic`` directives. Zero when the env has no
        topology / no directives loaded."""
        if self._topology_graph is None or not self._placement_directives:
            return 0.0
        return score_alignment(
            self._topology_graph,
            self._placement_directives,
            self._terminals,
        )

    def _hpwl_score(self) -> float:
        """Negated sum of per-net HPWL (placement-quality signal).
        Zero when no topology is bound. Mostly negative once two or
        more terminals on the same net have been placed."""
        if self._topology_graph is None:
            return 0.0
        return compute_hpwl_score(
            self._state, self._topology_graph, self._terminals,
        )

    def _row_score(self) -> float:
        """Per-device row-alignment score (NMOS→bottom, PMOS→top).
        Zero when no topology is bound or no devices have been placed."""
        if self._topology_graph is None or not self._placed_origins:
            return 0.0
        return compute_row_score(
            self._topology_graph,
            self._placed_origins,
            self.cell_height_um,
        )

    def _lvs_mismatch_count(self) -> int | None:
        """Number of LVS mismatches reported by the magic+netgen runner.

        Returns ``None`` when the env was constructed without an
        ``lvs`` runner — the reward layer treats ``None`` as "skip the
        LVS terms for this step" so envs without LVS pay no overhead
        and emit no spurious LVS signal.
        """
        if self._lvs is None:
            return None
        try:
            result = self._lvs.run(self._state)
        except Exception:
            return None
        return len(result.mismatches)

    def _all_devices_placed(self) -> bool:
        if self._topology_graph is None:
            return True
        n = self._topology_graph.n_devices
        return bool(self._placed_mask[:n].all())

    # ── ROUTE helpers ────────────────────────────────────────────────────────

    def _apply_route(self, env_action: EnvAction) -> bool:
        """Materialise one routing segment. Returns True iff valid."""
        if self._topology_graph is None:
            return False
        n_idx = env_action.net_idx
        if n_idx < 0 or n_idx >= self._topology_graph.n_nets:
            return False
        if n_idx >= self.net_cap:
            return False
        net_name = self._topology_graph.nets[n_idx].name
        try:
            add_route_segment(
                self._state,
                layer=env_action.route_layer,
                x_um=env_action.route_x_um,
                y_um=env_action.route_y_um,
                w_um=env_action.route_w_um,
                h_um=env_action.route_h_um,
                net_name=net_name,
            )
        except Exception:
            return False
        self._routed_mask[n_idx] = True
        return True

    # ── Internals ────────────────────────────────────────────────────────────

    def _coerce_options(self,
                        options: dict[str, Any] | EpisodeConfig | None
                        ) -> EpisodeConfig:
        if options is None:
            return EpisodeConfig()
        if isinstance(options, EpisodeConfig):
            return options
        return EpisodeConfig(
            state=options.get("state"),
            state_factory=options.get("state_factory"),
            cell_bbox=options.get("cell_bbox"),
            forbid_kinds=frozenset(options.get("forbid_kinds", frozenset())),
            start_phase=options.get("start_phase"),
        )

    def _build_step_output(
        self,
        *,
        reward_breakdown: RewardBreakdown | None = None,
        env_action:       EnvAction | None = None,
        action_valid:     bool = True,
        state_changed:    bool = False,
        phase_transitioned: bool = False,
    ) -> tuple[dict, dict]:
        progress = (self._step_count / self.max_steps
                    if self.max_steps > 0 else 0.0)
        # Per-terminal global positions for the proximity map
        # (only computed when the env was built with proximity_shape).
        terminal_points: list[tuple[float, float]] | None = None
        cell_dims: tuple[float, float] | None = None
        if self._proximity_shape is not None:
            terminal_points = [(x, y) for (x, y, _layer)
                               in self._terminals.values()]
            cell_dims = (self.cell_width_um, self.cell_height_um)
        obs_struct = build_observation(
            self._state, self._violations,
            poly_cap=self.poly_cap,
            viol_cap=self.viol_cap,
            cell_bbox=self._cell_bbox,
            step_progress=progress,
            topology_global=self._topology_global,
            proximity_shape=self._proximity_shape,
            terminal_positions=terminal_points,
            cell_dimensions=cell_dims,
        )
        self._last_obs     = obs_struct.to_dict()
        self._last_rid_map = obs_struct.rid_to_idx

        mask = self._compute_action_mask()

        info: dict[str, Any] = {
            "n_violations":     len(self._violations),
            "n_polygons":       len(self._state),
            "step":             self._step_count,
            "phase":            self._phase,
            "n_devices_placed": int(self._placed_mask.sum()),
            "n_nets_routed":    int(self._routed_mask.sum()),
            "connectivity":     self._connectivity_score(),
            "alignment":        self._alignment_score(),
            "electrical":       self._electrical_score(),
            "hpwl":             self._hpwl_score(),
            "row":              self._row_score(),
            "shorts":           compute_short_count(self._state),
            "lvs_mismatches":   self._lvs_mismatch_count(),
            "action_mask":      mask,
            "drc_cache_stats":  self._drc.stats(),
        }
        if phase_transitioned:
            info["phase_transitioned"] = True
        if reward_breakdown is not None:
            info["reward"] = reward_breakdown.to_dict()
        if env_action is not None:
            info["action"] = {
                "kind": env_action.kind, "rid": env_action.rid,
                "edge": env_action.edge, "sign_x": env_action.sign_x,
                "sign_y": env_action.sign_y, "mag": env_action.mag,
                "device_idx":  env_action.device_idx,
                "x_um":        env_action.x_um,
                "y_um":        env_action.y_um,
                "orientation": env_action.orientation,
                "net_idx":     env_action.net_idx,
                "route_layer": env_action.route_layer,
                "route_x_um":  env_action.route_x_um,
                "route_y_um":  env_action.route_y_um,
                "route_w_um":  env_action.route_w_um,
                "route_h_um":  env_action.route_h_um,
                "valid": action_valid, "state_changed": state_changed,
            }
        return self._last_obs, info

    def _compute_action_mask(self) -> np.ndarray:
        n_devices = (
            self._topology_graph.n_devices
            if self._topology_graph is not None else 0
        )
        n_nets = (
            self._topology_graph.n_nets
            if self._topology_graph is not None else 0
        )
        return action_mask_for(
            self._state, self._last_rid_map,
            target_cap=self.target_cap,
            mag_bins=self._action_helper.mag_bins,
            forbid_kinds=self._forbid_kinds,
            enable_place=self.enable_place,
            phase=self._phase,
            device_cap=self.device_cap,
            n_devices=n_devices,
            placed_mask=self._placed_mask,
            x_bins=self.x_bins,
            y_bins=self.y_bins,
            enable_route=self.enable_route,
            net_cap=self.net_cap,
            n_nets=n_nets,
            route_x_bins=self.route_x_bins,
            route_y_bins=self.route_y_bins,
            route_w_bins=self.route_w_bins,
            route_h_bins=self.route_h_bins,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> LayoutState | None:
        return self._state

    @property
    def violations(self):
        return self._violations

    @property
    def phase(self) -> str:
        return self._phase

    def action_mask(self) -> np.ndarray:
        """sb3-contrib MaskablePPO calls a method named ``action_masks()``;
        we expose both names for convenience."""
        return self._compute_action_mask()

    # MaskablePPO standard hook
    def action_masks(self) -> np.ndarray:  # noqa: D401
        return self.action_mask()


__all__ = ["LayoutEnv", "EpisodeConfig"]
