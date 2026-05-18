"""
layout_gen.rl.tests.test_coupled_place — auto-regressive PLACE coupling.

Verifies the ``couple_device_position`` flag (RL_GUIDE §9.1 option C):

  * The conditioned heads only exist when the flag is on; the
    unconditioned heads only exist when it's off.
  * For a fixed observation, ``heads(device_idx=a)`` and
    ``heads(device_idx=b)`` produce different position logits — i.e.
    the device actually changes the position distribution. This is the
    structural test that closes the factored-action coupling failure
    mode (the same (x, y) cannot be the argmax for two different
    devices unless it just happens to be).
  * ``MaskableLayoutPolicy.forward`` autoregressively samples a device
    first, then position dims conditioned on it. The returned action's
    device dim agrees with the device that was used to condition.
  * ``evaluate_actions`` reproduces ``forward``'s log-prob in
    deterministic mode (this catches subtle bugs in which logits the
    sampling vs. eval paths use).
  * BC training with the flag on runs without errors and saves a
    checkpoint that can be reloaded.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from layout_gen.synth.geo.state import LayoutState

from layout_gen.repair.features import POLY_FEAT_DIM
from layout_gen.rl.env.action_space import (
    PLACE_KINDS, REPAIR_KINDS, action_mask_for,
)
from layout_gen.rl.env.observation import (
    DEFAULT_POLY_CAP, DEFAULT_VIOL_CAP, N_GLOBAL, V_FEAT_DIM,
    build_observation,
)
from layout_gen.rl.env.place_action import N_ORIENTATIONS
from layout_gen.rl.policy import (
    LayoutPolicy, LayoutPolicyConfig, MaskableLayoutPolicy,
)
from layout_gen.rl.training import (
    BCTrainer, BCTrainerConfig, TrajectoryDataset,
    mine_synthetic_trajectories,
)
from layout_gen.rl.training.synthetic import SyntheticMineConfig


# ── Fixtures / helpers ───────────────────────────────────────────────────────

def _coupled_cfg() -> LayoutPolicyConfig:
    return LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        enable_place=True, device_cap=4, x_bins=8, y_bins=8,
        couple_device_position=True,
    )


def _make_obs(cfg: LayoutPolicyConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    poly_feats = torch.zeros(batch, cfg.poly_cap, POLY_FEAT_DIM)
    viol_feats = torch.zeros(batch, cfg.viol_cap, V_FEAT_DIM)
    poly_feats.normal_(generator=torch.Generator().manual_seed(0))
    viol_feats.normal_(generator=torch.Generator().manual_seed(1))
    poly_mask = torch.zeros(batch, cfg.poly_cap)
    viol_mask = torch.zeros(batch, cfg.viol_cap)
    poly_mask[:, :3] = 1.0
    viol_mask[:, :1] = 1.0
    global_feats = torch.zeros(batch, N_GLOBAL).uniform_(
        0.0, 1.0, generator=torch.Generator().manual_seed(2),
    )
    return {
        "poly_feats":   poly_feats,
        "poly_mask":    poly_mask,
        "viol_feats":   viol_feats,
        "viol_mask":    viol_mask,
        "global_feats": global_feats,
    }


def _build_sb3_policy(cfg: LayoutPolicyConfig) -> MaskableLayoutPolicy:
    obs_space = spaces.Dict({
        "poly_feats":   spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(cfg.poly_cap, POLY_FEAT_DIM),
                                    dtype=np.float32),
        "poly_mask":    spaces.Box(0.0, 1.0, (cfg.poly_cap,), np.float32),
        "viol_feats":   spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(cfg.viol_cap, V_FEAT_DIM),
                                    dtype=np.float32),
        "viol_mask":    spaces.Box(0.0, 1.0, (cfg.viol_cap,), np.float32),
        "global_feats": spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(N_GLOBAL,), dtype=np.float32),
    })
    nvec = [
        len(REPAIR_KINDS) + len(PLACE_KINDS),
        cfg.target_cap, 4, 2, 2, cfg.mag_bins,
        cfg.device_cap, cfg.x_bins, cfg.y_bins, N_ORIENTATIONS,
    ]
    act_space = spaces.MultiDiscrete(nvec)
    return MaskableLayoutPolicy(
        observation_space=obs_space, action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        layout_config=cfg,
    )


def _make_place_mask(cfg: LayoutPolicyConfig, n_devices: int,
                     batch: int = 2) -> torch.Tensor:
    s = LayoutState()
    obs_struct = build_observation(
        s, [], poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap,
    )
    mask = action_mask_for(
        s, obs_struct.rid_to_idx,
        target_cap=cfg.target_cap, mag_bins=cfg.mag_bins,
        enable_place=True, phase="place",
        device_cap=cfg.device_cap, n_devices=n_devices,
        x_bins=cfg.x_bins, y_bins=cfg.y_bins,
    )
    return torch.from_numpy(np.stack([mask] * batch))


# ── Tests ────────────────────────────────────────────────────────────────────

def test_flag_swaps_head_modules():
    """Coupled config has only ``*_cond_head``; default config has only
    the unconditioned heads. This guarantees no wasted parameters and
    no silent fallback path between the two regimes."""
    coupled = LayoutPolicy(_coupled_cfg())
    assert hasattr(coupled, "x_bin_cond_head")
    assert hasattr(coupled, "y_bin_cond_head")
    assert hasattr(coupled, "orient_cond_head")
    assert not hasattr(coupled, "x_bin_head")
    assert not hasattr(coupled, "y_bin_head")
    assert not hasattr(coupled, "orient_head")

    default_cfg = LayoutPolicyConfig(enable_place=True, device_cap=4,
                                      x_bins=4, y_bins=4)
    factored = LayoutPolicy(default_cfg)
    assert hasattr(factored, "x_bin_head")
    assert not hasattr(factored, "x_bin_cond_head")


def test_position_logits_change_with_device():
    """The structural test: feeding two different devices through the
    *same* observation must produce different position logits. If they
    were equal, coupling would be vacuous and the original
    same-(x,y)-for-two-devices failure mode would persist."""
    cfg = _coupled_cfg()
    policy = LayoutPolicy(cfg).eval()
    obs = _make_obs(cfg, batch=1)

    with torch.no_grad():
        ctx, poly_emb, poly_pad = policy.encode_state(obs)
        out_a = policy.heads(ctx, poly_emb, poly_pad,
                              device_idx=torch.tensor([0]))
        out_b = policy.heads(ctx, poly_emb, poly_pad,
                              device_idx=torch.tensor([1]))

    # Device head doesn't depend on device_idx, so it should be identical.
    assert torch.allclose(out_a.device, out_b.device, atol=1e-6)
    # Position heads should differ — different conditioning input.
    assert not torch.allclose(out_a.x_bin, out_b.x_bin, atol=1e-6)
    assert not torch.allclose(out_a.y_bin, out_b.y_bin, atol=1e-6)
    assert not torch.allclose(out_a.orient, out_b.orient, atol=1e-6)


def test_default_device_when_idx_omitted():
    """Calling ``heads()`` without ``device_idx`` on a coupled policy
    should fall back to argmax(device_head) and still emit position
    logits — this keeps single-pass eval callers functional."""
    cfg = _coupled_cfg()
    policy = LayoutPolicy(cfg).eval()
    obs = _make_obs(cfg, batch=2)
    with torch.no_grad():
        out = policy(obs)
    assert torch.isfinite(out.x_bin).all()
    assert torch.isfinite(out.y_bin).all()
    assert torch.isfinite(out.orient).all()


def test_sb3_forward_returns_consistent_action_shape():
    """The two-pass forward must still produce a (B, 10) action vector
    (kind, target, edge, sx, sy, mag + device, x_bin, y_bin, orient)."""
    cfg = _coupled_cfg()
    policy = _build_sb3_policy(cfg)
    obs = _make_obs(cfg, batch=3)
    mask = _make_place_mask(cfg, n_devices=cfg.device_cap, batch=3)
    actions, values, log_probs = policy.forward(obs, action_masks=mask)
    assert actions.shape == (3, 10)
    assert values.shape  == (3,)
    assert log_probs.shape == (3,)
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_probs).all()


def test_sb3_forward_device_matches_conditioning():
    """In deterministic mode, the device dim of the returned action
    must equal the device the position heads were conditioned on
    (otherwise log_prob factorisation would mix the wrong terms)."""
    cfg = _coupled_cfg()
    torch.manual_seed(7)
    policy = _build_sb3_policy(cfg)
    obs = _make_obs(cfg, batch=4)
    mask = _make_place_mask(cfg, n_devices=cfg.device_cap, batch=4)

    actions_a, _, _ = policy.forward(obs, action_masks=mask, deterministic=True)
    actions_b, _, _ = policy.forward(obs, action_masks=mask, deterministic=True)
    # Deterministic forward should be reproducible.
    assert torch.equal(actions_a, actions_b)

    device_dim = 6
    # Re-run argmax-of-device by hand to verify the device dim agrees.
    ctx, _, _ = policy.layout_policy.encode_state(obs)
    expected_dev = policy.layout_policy.device_head(ctx).argmax(dim=-1)
    assert torch.equal(actions_a[:, device_dim], expected_dev)


def test_sb3_forward_accepts_numpy_action_masks():
    """sb3-contrib MaskablePPO passes ``action_masks`` as ``np.ndarray``
    during rollout collection (only ``evaluate_actions`` sees a torch
    tensor). The autoregressive coupling path used to call ``.bool()``
    directly on the input — fine for tensors, AttributeError for numpy.
    This regression test asserts both forms work."""
    cfg = _coupled_cfg()
    policy = _build_sb3_policy(cfg)
    obs = _make_obs(cfg, batch=2)
    mask_t = _make_place_mask(cfg, n_devices=cfg.device_cap, batch=2)
    mask_np = mask_t.numpy()

    a_t, _, _ = policy.forward(obs, action_masks=mask_t, deterministic=True)
    a_np, _, _ = policy.forward(obs, action_masks=mask_np, deterministic=True)
    # Same input mask in two dtypes should yield the same deterministic
    # action — proves the np→tensor coercion didn't change semantics.
    assert torch.equal(a_t, a_np)


def test_sb3_evaluate_actions_consistent_with_forward():
    """Forward(deterministic) and evaluate_actions on the resulting
    action must yield the same log-prob — this catches inconsistent
    logit sources between sampling and training."""
    cfg = _coupled_cfg()
    torch.manual_seed(11)
    policy = _build_sb3_policy(cfg)
    obs = _make_obs(cfg, batch=3)
    mask = _make_place_mask(cfg, n_devices=cfg.device_cap, batch=3)

    actions, _, log_probs_fwd = policy.forward(
        obs, action_masks=mask, deterministic=True,
    )
    _, log_probs_eval, entropy = policy.evaluate_actions(
        obs, actions, action_masks=mask,
    )
    assert torch.allclose(log_probs_fwd, log_probs_eval, atol=1e-5)
    assert torch.isfinite(entropy).all()


def test_global_feats_carry_progress_and_phase():
    """N_GLOBAL was bumped from 4 to 8 to expose ``n_placed/n_devices``
    and a (place/route/repair) one-hot, addressing the audit's
    finding #5 (the policy had no observation signal for "how many
    more devices to place" or "which phase am I in")."""
    from layout_gen.rl.env.observation import N_GLOBAL, build_observation
    assert N_GLOBAL == 8

    s = LayoutState()
    s.add(layer="met1", x0=0.0, y0=0.0, x1=0.10, y1=0.10)

    obs_place = build_observation(
        s, [], n_placed=2, n_devices_total=4, phase="place",
    )
    obs_route = build_observation(
        s, [], n_placed=4, n_devices_total=4, phase="route",
    )
    obs_repair = build_observation(
        s, [], n_placed=4, n_devices_total=4, phase="repair",
    )

    # n_placed_norm
    assert obs_place.global_feats[4] == pytest.approx(0.5)
    assert obs_route.global_feats[4] == pytest.approx(1.0)
    # phase one-hot exclusivity
    assert tuple(obs_place.global_feats[5:8].tolist())  == (1.0, 0.0, 0.0)
    assert tuple(obs_route.global_feats[5:8].tolist())  == (0.0, 1.0, 0.0)
    assert tuple(obs_repair.global_feats[5:8].tolist()) == (0.0, 0.0, 1.0)

    # REPAIR-only env (no topology, no phase) → no progress, no one-hot.
    obs_legacy = build_observation(s, [])
    assert tuple(obs_legacy.global_feats[4:8].tolist()) == (0.0, 0.0, 0.0, 0.0)


def test_place_progress_reward_scales_per_cell_size():
    """Completion-fraction reward: per-device contribution is
    ``cfg.place_progress / n_devices_total``, so the per-episode total
    of place_progress stays equal across cells of different sizes.
    Closes the audit's finding #3 (no long-horizon completion signal)."""
    from layout_gen.rl.env.reward import RewardConfig, compute_reward

    cfg = RewardConfig()  # uses new default place_progress=4.0

    # Place the 1st of 2 devices.
    rb_inv = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True, phase="place",
        config=cfg,
        n_placed_before=0, n_placed_after=1, n_devices_total=2,
    )
    # Place the 1st of 4 devices.
    rb_nand = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=True, action_valid=True, phase="place",
        config=cfg,
        n_placed_before=0, n_placed_after=1, n_devices_total=4,
    )
    # Per-device contribution scales with 1/n_devices_total.
    assert rb_inv.place_progress  == pytest.approx(cfg.place_progress / 2)
    assert rb_nand.place_progress == pytest.approx(cfg.place_progress / 4)

    # Per-episode total is invariant: 2 × (place_progress/2) == 4 × (place_progress/4).
    inv_total  = 2 * rb_inv.place_progress
    nand_total = 4 * rb_nand.place_progress
    assert inv_total == pytest.approx(nand_total)

    # Invalid placement: no progress reward.
    rb_invalid = compute_reward(
        violations_before=[], violations_after=[],
        state_changed=False, action_valid=False, phase="place",
        config=cfg,
        n_placed_before=1, n_placed_after=1, n_devices_total=4,
    )
    assert rb_invalid.place_progress == 0.0


def test_row_mask_prunes_y_bins_when_unplaced_is_type_unanimous():
    """Audit fix B: under strict_row_alignment, when the unplaced
    device set is unanimous on type (all-NMOS or all-PMOS), the
    y_bin mask must restrict to the half-row the env's strict guard
    would accept. Mixed-type sets leave y_bins fully open."""
    from layout_gen.rl.env.action_space import action_mask_for
    from layout_gen.rl.env.observation import build_observation

    s = LayoutState()
    obs = build_observation(s, [], poly_cap=32, viol_cap=8)
    common = dict(
        target_cap=32, mag_bins=8,
        enable_place=True, phase="place",
        device_cap=4, n_devices=4,
        x_bins=16, y_bins=16,
        strict_row_alignment=True,
    )
    # nvec for the (kind, target, edge, sx, sy, mag, device, x, y, orient) layout.
    y_start = (len(REPAIR_KINDS) + len(PLACE_KINDS)) + 32 + 4 + 2 + 2 + 8 + 4 + 16
    y_stop  = y_start + 16

    # All unplaced are NMOS → bottom half of y allowed.
    m_nmos = action_mask_for(
        s, obs.rid_to_idx,
        unplaced_device_types=["nmos", "nmos"], **common,
    )
    assert m_nmos[y_start:y_start + 8].all()
    assert not m_nmos[y_start + 8:y_stop].any()

    # All unplaced are PMOS → top half allowed.
    m_pmos = action_mask_for(
        s, obs.rid_to_idx,
        unplaced_device_types=["pmos", "pmos"], **common,
    )
    assert not m_pmos[y_start:y_start + 8].any()
    assert m_pmos[y_start + 8:y_stop].all()

    # Mixed types → both halves open (pre-fix behaviour preserved).
    m_mixed = action_mask_for(
        s, obs.rid_to_idx,
        unplaced_device_types=["nmos", "pmos"], **common,
    )
    assert m_mixed[y_start:y_stop].all()

    # strict_row_alignment off → no pruning regardless of unanimity.
    m_loose = action_mask_for(
        s, obs.rid_to_idx,
        unplaced_device_types=["nmos"], **{**common,
                                            "strict_row_alignment": False},
    )
    assert m_loose[y_start:y_stop].all()


def test_bc_training_with_coupling(tmp_path: Path):
    """End-to-end: BC-train a coupled policy on synthetic trajectories
    and verify the saved checkpoint roundtrips. The synthetic corpus
    is REPAIR-only, so device targets are absent — the coupled trainer
    must handle that gracefully (validity mask zeroes those samples)."""
    cfg = LayoutPolicyConfig(
        poly_cap=32, viol_cap=8, target_cap=32, mag_bins=8,
        d_token=16, d_trunk=32, n_layers=1, n_heads=4, dim_ff=32,
        enable_place=True, device_cap=4, x_bins=4, y_bins=4,
        couple_device_position=True,
    )

    rng = random.Random(0)
    traj_dir = tmp_path / "trajs"

    def _seed():
        s = LayoutState()
        for k in range(4):
            x0 = 0.25 * k + rng.uniform(-0.005, 0.005)
            s.add(layer="met1", x0=x0, y0=0.0, x1=x0 + 0.10, y1=0.10)
        return s

    counts = mine_synthetic_trajectories(
        state_factory=_seed, out_dir=traj_dir,
        config=SyntheticMineConfig(
            n_trajectories=64, depths=(1,),
            forbid_kinds=frozenset({"delete_rect", "shrink_rect", "grow_rect"}),
        ),
        rng=rng,
    )
    if counts["kept"] < 4:
        pytest.skip(f"insufficient synthetic trajectories: {counts}")

    dataset = TrajectoryDataset(
        traj_dir,
        poly_cap=cfg.poly_cap, viol_cap=cfg.viol_cap,
        target_cap=cfg.target_cap, mag_bins=cfg.mag_bins,
    )
    policy = LayoutPolicy(cfg)
    trainer = BCTrainer(policy, BCTrainerConfig(epochs=1, batch_size=8))
    trainer.fit(dataset)

    bc_path = tmp_path / "bc.pt"
    trainer.save(bc_path)

    # Round-trip the checkpoint.
    raw = torch.load(bc_path, weights_only=False, map_location="cpu")
    cfg2 = LayoutPolicyConfig(**raw["config"])
    assert cfg2.couple_device_position is True
    policy2 = LayoutPolicy(cfg2)
    policy2.load_state_dict(raw["state_dict"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
