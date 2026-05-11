"""
layout_gen.rl.env.layout_env — gymnasium.Env wrapping layout repair.

Phase 1 implements the **repair** episode: reset to a (broken) layout,
let the policy issue per-step edits, terminate when DRC-clean or when
``max_steps`` is exhausted. Place / route episodes (Phase 4) are
gated by an episode mode flag and re-use the same observation /
reward plumbing.

Episode lifecycle
-----------------
1. ``reset(options={"state": LayoutState, "cell_bbox": (x0,y0,x1,y1)})``
   – the env accepts a ready-made state. A factory callable is also
   accepted via ``options["state_factory"]`` for vectorised training
   where each env instance perturbs a fresh seed every episode.
2. ``step(action)`` – decode the MultiDiscrete action, apply the
   :class:`PerturbAction` (or no-op if masked out), re-run cached DRC,
   compute the reward.
3. Termination – ``terminated=True`` on DRC-clean. ``truncated=True``
   when ``max_steps`` reached.

Action masking
--------------
``info["action_mask"]`` carries the flat boolean mask consumed by
sb3-contrib MaskablePPO. Polices that ignore it will still produce
syntactically valid actions (the env tolerates invalid rids), but they
pay an ``invalid`` reward penalty.
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
from layout_gen.rl.env.reward       import (
    RewardConfig, RewardBreakdown, compute_reward,
)
from layout_gen.rl.env.runner       import CachedDRC, geometry_key


@dataclass
class EpisodeConfig:
    """Per-reset configuration. Pass via ``options=`` in :meth:`reset`."""
    state:         LayoutState | None = None
    state_factory: Callable[[], LayoutState] | None = None
    cell_bbox:     tuple[float, float, float, float] | None = None
    forbid_kinds:  frozenset[str] = frozenset()


class LayoutEnv(gym.Env):
    """Gymnasium env for DRC-repair RL.

    Parameters
    ----------
    drc :
        :class:`CachedDRC` (or any object with ``run(state) -> list[DRCViolation]``).
    poly_cap, viol_cap :
        Padding caps for polygons and violations.
    target_cap :
        Action-space target dim. Defaults to ``poly_cap`` so every
        addressable polygon has an action slot.
    mag_bins :
        Magnitude bin count for the action space.
    max_steps :
        Truncation budget. Zero or negative = unbounded.
    reward_config :
        Reward weights.
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
    ) -> None:
        super().__init__()
        self._drc = drc
        self.poly_cap   = poly_cap
        self.viol_cap   = viol_cap
        self.target_cap = target_cap or poly_cap
        self.max_steps  = max_steps
        self.reward_cfg = reward_config or RewardConfig()
        self._default_state_factory = default_state_factory
        self._default_cell_bbox     = default_cell_bbox

        self._action_helper = ActionSpace(target_cap=self.target_cap,
                                          mag_bins=mag_bins)
        self.action_space      = self._action_helper.gym_space
        self.observation_space = make_observation_space(poly_cap=poly_cap,
                                                         viol_cap=viol_cap)

        # Mutable per-episode state
        self._state:        LayoutState | None = None
        self._cell_bbox:    tuple[float, float, float, float] | None = None
        self._violations:   list = []
        self._step_count:   int  = 0
        self._last_obs:     dict | None = None
        self._last_rid_map: dict[int, int] = {}
        self._forbid_kinds: frozenset[str] = frozenset()

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self,
              *,
              seed:    int | None = None,
              options: dict[str, Any] | EpisodeConfig | None = None,
              ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        cfg = self._coerce_options(options)
        if cfg.state is not None:
            self._state = deepcopy(cfg.state)
        elif cfg.state_factory is not None:
            self._state = cfg.state_factory()
        elif self._default_state_factory is not None:
            self._state = self._default_state_factory()
        else:
            raise ValueError(
                "LayoutEnv.reset requires options['state'], options['state_factory'], "
                "or default_state_factory passed to __init__."
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

        # Decode + apply
        env_action = self._action_helper.decode(action, self._last_rid_map)
        perturb_action = self._action_helper.to_perturb(env_action)

        action_valid = perturb_action is not None
        state_changed = False
        before_key = geometry_key(self._state)
        if action_valid:
            try:
                perturb_lib.apply(self._state, perturb_action)
            except Exception:
                # Malformed apply (e.g. invalid rid in a stale rid_map) —
                # treat as invalid action. State is unchanged because
                # apply mutates only on success of its dispatch lookups.
                action_valid = False
            else:
                state_changed = geometry_key(self._state) != before_key

        # Re-run DRC
        after = list(self._drc.run(self._state))
        self._violations = after

        rb = compute_reward(
            violations_before=before,
            violations_after=after,
            state_changed=state_changed,
            action_valid=action_valid,
            config=self.reward_cfg,
        )

        terminated = (len(after) == 0 and len(before) > 0)
        truncated  = (self.max_steps > 0 and self._step_count >= self.max_steps)

        obs, info = self._build_step_output(
            reward_breakdown=rb,
            env_action=env_action,
            action_valid=action_valid,
            state_changed=state_changed,
        )
        return obs, rb.total, terminated, truncated, info

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
        )

    def _build_step_output(
        self,
        *,
        reward_breakdown: RewardBreakdown | None = None,
        env_action:       EnvAction | None = None,
        action_valid:     bool = True,
        state_changed:    bool = False,
    ) -> tuple[dict, dict]:
        progress = (self._step_count / self.max_steps
                    if self.max_steps > 0 else 0.0)
        obs_struct = build_observation(
            self._state, self._violations,
            poly_cap=self.poly_cap,
            viol_cap=self.viol_cap,
            cell_bbox=self._cell_bbox,
            step_progress=progress,
        )
        self._last_obs     = obs_struct.to_dict()
        self._last_rid_map = obs_struct.rid_to_idx

        mask = action_mask_for(
            self._state, self._last_rid_map,
            target_cap=self.target_cap,
            mag_bins=self._action_helper.mag_bins,
            forbid_kinds=self._forbid_kinds,
        )

        info: dict[str, Any] = {
            "n_violations":   len(self._violations),
            "n_polygons":     len(self._state),
            "step":           self._step_count,
            "action_mask":    mask,
            "drc_cache_stats": self._drc.stats(),
        }
        if reward_breakdown is not None:
            info["reward"] = reward_breakdown.to_dict()
        if env_action is not None:
            info["action"] = {
                "kind": env_action.kind, "rid": env_action.rid,
                "edge": env_action.edge, "sign_x": env_action.sign_x,
                "sign_y": env_action.sign_y, "mag": env_action.mag,
                "valid": action_valid, "state_changed": state_changed,
            }
        return self._last_obs, info

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> LayoutState | None:
        return self._state

    @property
    def violations(self):
        return self._violations

    def action_mask(self) -> np.ndarray:
        """sb3-contrib MaskablePPO calls a method named ``action_masks()``;
        we expose both names for convenience."""
        return action_mask_for(
            self._state, self._last_rid_map,
            target_cap=self.target_cap,
            mag_bins=self._action_helper.mag_bins,
            forbid_kinds=self._forbid_kinds,
        )

    # MaskablePPO standard hook
    def action_masks(self) -> np.ndarray:  # noqa: D401
        return self.action_mask()


__all__ = ["LayoutEnv", "EpisodeConfig"]
