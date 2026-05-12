# layout_gen RL — Plan & Status

This document is the portable handoff for the RL-based layout generator
under `layout_gen/rl/`. It's tracked in the repo so picking up work on a
new machine just needs `git pull`.

End goal: a **PDK-agnostic generator** that takes a topology YAML
(devices + nets + placement intent) and produces a DRC-clean,
LVS-meaningful GDS. SRAM cells are the starting point.

Current branch: `drc-repair-engine`.

---

## Quick start (after `git clone`)

```bash
# 1. Set up venv + install deps (CPU torch).
sudo apt install -y python3.12-venv      # one-time, Debian/Ubuntu
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -e ".[rl,test]"

# 2. (Optional) install klayout for real DRC. Already installed on the
#    primary dev machine. Magic also works.

# 3. Run the suite (must pass — currently 106 tests).
.venv/bin/python -m pytest layout_gen/rl/tests/

# 4. End-to-end inverter pipeline (the demo we already ran).
mkdir -p demos checkpoints out
.venv/bin/python -m layout_gen.rl.scripts.extract_demos \
    --templates inverter --out demos/

.venv/bin/python -m layout_gen.rl.scripts.train_bc \
    --demos demos/ --epochs 30 --batch-size 16 --lr 1e-3 \
    --enable-place --enable-route --use-topology --topology-dim 64 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --poly-cap 128 --viol-cap 32 --target-cap 128 \
    --out checkpoints/bc_inv.pt

.venv/bin/python -m layout_gen.rl.scripts.train_ppo \
    --topology inverter --enable-place --enable-route --no-drc \
    --bc-init checkpoints/bc_inv.pt \
    --total-timesteps 5000 \
    --max-place-steps 4 --max-route-steps 6 --max-steps 16 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --ent-coef 0.005 \
    --out checkpoints/ppo_inv_full.zip

.venv/bin/python -m layout_gen.rl.scripts.generate \
    --topology inverter --no-drc \
    --checkpoint checkpoints/ppo_inv_full.zip \
    --cell-name inv_full --out out/inv_full.gds \
    --max-place-steps 4 --max-route-steps 6 --max-steps 16 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --poly-cap 128 --viol-cap 32 --target-cap 128

.venv/bin/python -m layout_gen.rl.scripts.inspect_gds \
    out/inv_full.gds --ascii --strict
```

For real-DRC training drop `--no-drc` (klayout takes ~0.5 s/step).

---

## Architecture (one screen)

```
                     ┌──────────────────────────────────────────────┐
                     │  layout_gen/templates/cells/<name>.yaml      │
                     │   devices, nets, placement_logic, routing    │
                     │       (PDK-agnostic input contract)          │
                     └──────────────────────┬───────────────────────┘
                                            │
                                ┌───────────┴────────────┐
                                ▼                        ▼
       ┌───────────────────────────────────┐   ┌──────────────────────┐
       │ rl/topology/parser.py             │   │ rl/training/         │
       │   CellTemplate → TopologyGraph    │   │   demo_extract.py    │
       │   (devices=nodes, nets=hyperedges)│   │   (synth → PLACE     │
       └───────────────┬───────────────────┘   │    actions, BC)      │
                       │                        └────────┬─────────────┘
                       ▼                                 │
       ┌───────────────────────────────────┐             ▼
       │ rl/topology/encoder.py            │  ┌──────────────────────┐
       │   bipartite GNN                   │  │ rl/training/         │
       │   per-device + global embedding   │  │   bc_pretrain.py     │
       └───────────────┬───────────────────┘  │   PlacementDemoDataset│
                       │                       └────────┬─────────────┘
                       ▼                                │
       ┌──────────────────────────────────────┐         │
       │ rl/env/                              │◀────────┘  BC checkpoint
       │   layout_env.py    (PLACE→ROUTE→     │            (warm-start)
       │     REPAIR machine)                  │
       │   action_space.py  (MultiDiscrete)   │            ┌────────────┐
       │   place_action.py  (transistor cache,│◀───────────│ rl/policy/ │
       │     terminal positions)              │            │  network   │
       │   route_action.py  (metal segments,  │            │  + sb3     │
       │     net tagging)                     │────────────▶│ MaskableLayoutPolicy
       │   reward.py        (phase-aware,     │            └─────┬──────┘
       │     +alignment +connectivity         │                  │
       │     +electrical)                     │                  ▼
       │   connectivity.py  (per-net           │      ┌──────────────────────┐
       │     touched + transitive)            │◀─────│ rl/training/         │
       │   placement_intent.py (align_gate /  │      │   ppo_train.py       │
       │     abut_x / origin satisfaction)    │      │   MaskablePPO + vec  │
       │   runner.py        (CachedDRC over   │      │     env + Monitor    │
       │     klayout / magic)                 │      └──────────────────────┘
       └──────────────────┬───────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────┐
       │ rl/scripts/                          │
       │   train_bc / train_ppo / generate /  │
       │   extract_demos / inspect_gds        │
       └──────────────────────────────────────┘
```

PDK-specific knowledge stays behind the DRC/LVS runner interface +
the layer→`LAYER_ROLES` mapping in `repair/features.py`. No rule µm
constants in `rl/`.

---

## Reward shaping (`rl/env/reward.py`)

Phase-aware composite. Per-step:

| Term | Default weight | Notes |
|---|---|---|
| `drc_delta` (per-phase) | 0.05 / 0.20 / 1.00 (place / route / repair) | PLACE intrinsically adds violations until layout is complete; full weight only in REPAIR |
| `value_delta` (per-phase) | 0 / 0.05 / 0.05 | Soft Δ in measured-µm-deficit |
| `step` | −0.05 | Discourages stalling |
| `terminal` | +5.0 | Only fires in REPAIR when DRC reaches clean |
| `invalid` | −0.5 | Action structurally invalid |
| `no_change` | −0.2 | Action ran but geometry unchanged |
| `place_success` | +1.0 | Valid PLACE that changed state |
| `route_success` | +0.5 | Valid ROUTE that changed state |
| `connectivity_delta` | ×2.0 | Δ(per-net fraction of terminals touched by same-net wire) |
| `alignment_delta` | ×1.5 | Δ(YAML directive satisfaction: align_gate / abut_x / origin) |
| `electrical_delta` | ×3.0 | Δ(per-net "all terminals in one connected component") via union-find |

The combination empirically takes the inverter from `ep_rew_mean ≈ 3.3`
(naïve) to `≈ 12.5` (BC + all rewards), and produces canonical
gate-aligned layouts.

---

## Where we are (Phase 5 complete)

| Phase | Status | Commits |
|---|---|---|
| 1: env + repair primitives | ✅ | `23cb778` |
| 2: BC trainer on perturb trajectories | ✅ | `23cb778` |
| 3: MaskablePPO wrapper | ✅ | `23cb778` |
| 4.1: topology GNN + global conditioning | ✅ | `521ff8e` |
| 4.2a: PLACE action vocabulary + phase machine | ✅ | `485112e` |
| 4.2b: `generate.py` end-to-end | ✅ | `df79d19` |
| 4.2c: ROUTE action vocabulary | ✅ | `39e6cd7` |
| 4.3: inspect tool + bug fixes | ✅ | `8223f75`, `bfa7f62`, `559be8a` |
| 4.4: phase-aware reward + topology train CLI | ✅ | `69c8cd8` |
| 4.5: connectivity reward | ✅ | `442fe9c` |
| 4.6: alignment reward + real-DRC default | ✅ | `9e96bdc` |
| 4.7: electrical (transitive) connectivity | ✅ | `fac638d` |
| 5: BC demos from synth pipeline | ✅ | `660b69a` |
| 5.1: train_bc wired to demo dataset | ✅ | this commit |

**Test count:** 106 passing.

**End-to-end demo:** inverter via demo → BC → PPO → generate produces a
gate-aligned layout (NMOS at (0.615, 0.505), PMOS at (0.615, 1.755)).

---

## What's next

In rough decreasing order of impact. Pick one per session.

### 1. Real-DRC PPO training run (~hours, no new code)

```bash
.venv/bin/python -m layout_gen.rl.scripts.train_ppo \
    --topology inverter --enable-place --enable-route \
    --bc-init checkpoints/bc_inv.pt \
    --total-timesteps 20000 \
    --max-place-steps 4 --max-route-steps 6 --max-steps 16 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --ent-coef 0.005 \
    --out checkpoints/ppo_inv_realdrc.zip
```

20k steps × ~0.5 s/step ≈ 3 hours. Run overnight, then `generate.py`
+ `inspect_gds.py --strict` for the verdict.

### 2. ROUTE demos from the synth router (~1 day)

Currently `demo_extract.py` only emits PLACE actions. The synth pipeline
also produces routing geometry; we can attribute each metal rect to a
net (it overlaps that net's terminals) and emit ROUTE actions in the
demo JSON. This would let BC pretrain the ROUTE heads as well.

Files to touch:
- `rl/training/demo_extract.py` — extend to walk `result.component`,
  identify routing rects, attribute by terminal-overlap, output
  `kind: route_segment` actions.
- `rl/training/demo_dataset.py` — encode ROUTE labels (currently only
  PLACE samples are emitted).

### 3. Multi-cell training (~1 day)

Right now `train_ppo --topology X` trains on one cell. Add
`--topologies X,Y,Z` and a vec-env that rotates cells per episode.
Tests the policy's generalisation; should also let the topology GNN's
conditioning vector pull its weight (today the policy sees one
topology so the GNN is effectively a constant).

Files to touch:
- `rl/scripts/train_ppo.py` — accept comma-separated list, build one
  env factory per cell, alternate.
- Maybe `rl/training/ppo_train.py` — the trainer is already vec-env
  capable; just need to pass a list of factories.

### 4. LVS reward via magic (~1 day)

Replace the geometric heuristics in `connectivity.py` with calls to a
real LVS extractor. We have `layout_gen/lvs/magic_runner.py` from the
pre-RL work.

Files to touch:
- `rl/env/runner.py` — add `CachedLVS` analogous to `CachedDRC`.
- `rl/env/reward.py` — new term `lvs_delta` weighted on (clean - dirty).

### 5. Eval harness (~half day)

`rl/scripts/eval.py` that runs N episodes, reports mean ep_rew,
DRC-clean rate, alignment score, electrical score, inspector pass
rate. Useful for tracking progress quantitatively.

### 6. Decommission rule-based `synth/placer.py` + `synth/router.py`

Only after RL reaches parity on all template cells. Long-term goal.

---

## CLI reference (one-liners)

```bash
# Extract synth-derived BC demos for a list of cell templates.
.venv/bin/python -m layout_gen.rl.scripts.extract_demos \
    --templates inverter,nand2,nor2,bit_cell_6t,row_driver \
    --out demos/

# BC pretrain on demos. enable_place/enable_route/use_topology must
# match the PPO env that consumes the checkpoint.
.venv/bin/python -m layout_gen.rl.scripts.train_bc \
    --demos demos/ --epochs 30 --batch-size 16 --lr 1e-3 \
    --enable-place --enable-route --use-topology \
    --device-cap 16 --net-cap 16 --position-bins 8 \
    --out checkpoints/bc.pt

# PPO. --no-drc for fast iteration; drop it for real klayout.
.venv/bin/python -m layout_gen.rl.scripts.train_ppo \
    --topology inverter --enable-place --enable-route \
    --bc-init checkpoints/bc.pt \
    --total-timesteps 10000 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 \
    --out checkpoints/ppo.zip
# Add --no-drc for fast iteration; default is real klayout/magic.

# Generate from a checkpoint. Action-space caps MUST match training.
.venv/bin/python -m layout_gen.rl.scripts.generate \
    --topology inverter --checkpoint checkpoints/ppo.zip \
    --cell-name out_cell --out out/cell.gds \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --poly-cap 128 --viol-cap 32 --target-cap 128
# Add --no-drc for fast iteration; default is real klayout/magic.

# Inspect any GDS — reports per-device cluster classification, missing
# layers, ASCII top-down sketch. --strict exits non-zero on issues.
.venv/bin/python -m layout_gen.rl.scripts.inspect_gds \
    out/cell.gds --ascii --strict
```

---

## Architecture invariants (don't break these)

1. **PDK-agnostic env**: no rule µm constants in `rl/`. Anything PDK-
   specific lives behind the DRC/LVS runner interface or in
   `repair/features.py:LAYER_ROLES`.
2. **Input contract**: cell topology YAMLs in
   `layout_gen/templates/cells/*.yaml`. Don't invent a new format.
3. **Additive action space**: `enable_place` / `enable_route` /
   `use_topology` are flags. Older checkpoints stay loadable as long as
   the matching flags are off in the env they're loaded into.
4. **Real DRC by default** in CLIs. Fake DRC is a unit-test convenience
   only — see `feedback_real_drc_only.md` in user memory and the
   `--no-drc` flag.
5. **Inspector is the verification ground truth** for any generated
   cell. If a new device type or layer is added, extend
   `_NMOS_EXPECTED` / `_PMOS_EXPECTED` in `inspect_gds.py`.

---

## Files I touch most often

```
layout_gen/rl/
├── env/
│   ├── action_space.py     ← MultiDiscrete vocabulary, masking
│   ├── connectivity.py     ← per-net + transitive scores
│   ├── layout_env.py       ← gymnasium.Env, phase machine
│   ├── observation.py      ← padded poly+viol+global obs
│   ├── place_action.py     ← TransistorCache, place_device_full
│   ├── placement_intent.py ← align_gate / abut_x / origin
│   ├── reward.py           ← phase-aware composite
│   ├── route_action.py     ← add_route_segment
│   └── runner.py           ← CachedDRC
├── policy/
│   ├── network.py          ← LayoutPolicy + heads
│   └── sb3.py              ← MaskableLayoutPolicy
├── topology/
│   ├── parser.py           ← TopologyGraph
│   └── encoder.py          ← bipartite GNN
├── training/
│   ├── bc_pretrain.py      ← BCTrainer
│   ├── dataset.py          ← TrajectoryDataset (REPAIR-style)
│   ├── demo_dataset.py     ← PlacementDemoDataset (PLACE BC)
│   ├── demo_extract.py     ← synth → PLACE demos
│   └── ppo_train.py        ← PPOTrainer
├── scripts/
│   ├── extract_demos.py    ← bulk demo extraction CLI
│   ├── generate.py         ← topology YAML → GDS
│   ├── inspect_gds.py      ← verification CLI
│   ├── train_bc.py         ← BC training CLI
│   └── train_ppo.py        ← PPO training CLI
└── tests/                  ← 106 tests; keep green
```

---

*Last updated by Claude Code session 2026-05-12.*
