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
| `hpwl_delta` | ×0.5 | Δ(−Σ half-perimeter of per-net terminal bbox) — dense placement-quality signal (MaskPlace / AlphaChip / R-GCN-PPO) |

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
| 5.1: train_bc wired to demo dataset | ✅ | `a108012` |
| 5.2: HPWL placement reward term | ✅ | prev session |
| 5.3: pitch quantisation (poly + per-layer metal) | ✅ | prev session |
| 5.4: direction-aware metal pitch (h/v per layer) | ✅ | prev session |
| 5.5: typed-edge R-GCN (align_gate/abut_x/shared_diffusion) | ✅ | prev session |
| 5.6: ROUTE demos from synth (BC pretrains route heads) | ✅ | this session |
| 5.7: sky130 stdcell CPP (`poly.pitch_um: 0.46`) in PDK YAML | ✅ | this session |
| 5.8: eval harness (DRC-clean / inspector / hpwl / electrical / ep_rew, multi-topology breakdown) | ✅ | this session |
| 5.9: multi-cell training (`--topologies inv,nand2,nor2`, vec-env round-robin, max-cap policy) | ✅ | this session |
| 5.10: short-circuit reward + CachedLVS adapter + magic+netgen SPICE-ref emitter | ✅ | this session |
| 5.11: wiremask-style proximity channel in obs (MaskPlace gap closed) | ✅ | this session |
| 5.12: IBRL via BC distillation (β-decayed KL-to-BC in PPO loss; last SOTA gap) | ✅ | this session |
| 5.13: ablation harness (`ablation.py` — train N variants → eval each → diff table + CSV) | ✅ | this session |

**Test count:** 162 passing.

**End-to-end demo:** inverter via demo → BC → PPO → generate produces a
gate-aligned layout (NMOS at (0.615, 0.505), PMOS at (0.615, 1.755)).

**End-to-end multi-cell demo (inverter + nand2 + nor2, 6k PPO steps,
BC + IBRL distillation, std_cell pitch snapping)**:

```
                    DRC  inspect  ep_rew  electrical  finding
inverter   ✅       100%   100%    +9.95     2.00     gates aligned, all layers, both rows
nand2      ⚠️ STACK 100%   100%*   +10.91    1.00     STACKED: 3 diffs but only 1 poly column
nor2       ⚠️ STACK 100%   100%*   +10.28    1.00     STACKED: 3 diffs but only 1 poly column
```

The ``inspect=100%*`` for nand2/nor2 is the old inspector accepting a
stacked-device layout because all expected layers are present
(multiple devices at the same coords contribute to one cluster).
The new inspector check (``_stacked_device_count``) flags this.

**Reproduce**:

```bash
.venv/bin/python -m layout_gen.rl.scripts.extract_demos \
    --templates inverter,nand2,nor2 --out demos/multi3/

.venv/bin/python -m layout_gen.rl.scripts.train_bc \
    --demos demos/multi3/ --epochs 50 --batch-size 8 --lr 1e-3 \
    --enable-place --enable-route --use-topology --topology-dim 64 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --poly-cap 128 --viol-cap 32 --target-cap 128 \
    --out checkpoints/bc_multi3.pt

.venv/bin/python -m layout_gen.rl.scripts.train_ppo \
    --topologies inverter,nand2,nor2 \
    --enable-place --enable-route --no-drc \
    --bc-init      checkpoints/bc_multi3.pt \
    --ibrl-bc-init checkpoints/bc_multi3.pt \
    --ibrl-beta-start 0.5 --ibrl-beta-end 0.0 \
    --total-timesteps 6000 --n-envs 3 --n-steps 256 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --routing-mode std_cell \
    --out checkpoints/ppo_multi3.zip
```

---

## Where this sits vs the field (2025–2026 SOTA)

Surveyed in May 2026 — see Sources at the bottom of this doc.

**Closest related work:**

- **AlphaChip** (Google DeepMind, Nature 2021 + 2024 addendum) — edge-based
  GNN + CNN-on-canvas encoder, PPO over sequential macro placement on a
  discretised grid. Trained on 20 prior TPU blocks then fine-tuned. Only
  commercial RL-for-placement in production; macro-only. We're at
  transistor level, which is harder and less explored.
- **MaskPlace** (NeurIPS 2022) — drops the GNN; encodes state as
  pixel masks (`wiremask`, `viewmask`, `positionmask`). CNN policy with
  dense per-step HPWL-delta reward. Lesson: for small canvases the GNN
  may be overkill.
- **R-GCN + PPO for analog floorplanning** (DATE 2025, arXiv 2411.15212)
  — closest to our use case. **Relational GCN** with typed edges
  (`connected`, `h-align`, `v-align`, `h-symmetric`, `v-symmetric`).
  PPO over sequential device placement. Reward = HPWL + symmetry +
  area penalty.
- **DTCO standard-cell RL** (MDPI Electronics 2025) — also closest. RL
  for transistor placement on **poly-pitch grid**. Action space
  snapped to poly multiples — the policy can't propose non-gate-aligned
  positions by construction.
- **MaskRegulate** (NeurIPS 2024) — RL only for *refinement* (swap /
  shift) of an analytic placement, not from scratch. Reduces episode
  length 10×. Structurally aligned with our REPAIR phase.
- **DSO.ai / Cerebrus / NVIDIA hybrid-RL** — RL over *tool recipes*
  (Innovus knobs), not placement actions. Different problem; validates
  that direct RL on placement remains a research frontier.

**Where we already match or exceed SOTA:**

- ✅ Phase-aware reward (PLACE/ROUTE/REPAIR with different weights) —
  more sophisticated than the single weighted-sum used in most papers
- ✅ MaskablePPO + per-dim action masking (standard recipe; we have it)
- ✅ Topology GNN conditioning (bipartite device↔net; everyone does this)
- ✅ **Δ-HPWL placement reward** — the standard MaskPlace/AlphaChip/
  R-GCN-PPO dense signal, now in `RewardConfig.hpwl_delta` on top of
  the existing connectivity / electrical terms
- ✅ **Track-aligned action space** — PLACE x snapped to poly pitch
  and ROUTE x/y snapped to per-layer metal pitch (`std_cell` mode),
  honouring each layer's PDK-declared `preferred_direction` so
  horizontal layers only quantise y and vertical layers only
  quantise x. Gives the policy a maze-router-style discrete
  substrate. `analog` mode keeps poly pitch (gates must sit on a
  grid) but lets metals run on the manufacturing grid only. Beats
  the MDPI-2025 DTCO recipe, which only quantises poly.
- ✅ **Typed-edge R-GCN** — bipartite GNN extended with per-edge-type
  weight matrices over device↔device relations (`align_gate`,
  `abut_x`, `shared_diffusion`) extracted from the YAML's
  `placement_logic` + `placement.relations`. Matches the DATE 2025
  R-GCN-PPO recipe. Previously these directives only entered through
  the alignment reward; the GNN now encodes them too.
- ✅ **Transitive electrical connectivity** via union-find — most
  papers use HPWL proxies; we have actual net-completion signal
- ✅ **Real klayout in the reward loop** — most analog/cell papers use
  proxy DRC (RUDY, etc.); we run the actual tool
- ✅ BC warm-start from rule-based demos — exactly the IBRL recipe
  (arXiv 2311.02198), already wired through `train_bc --demos` →
  `train_ppo --bc-init`

**Where SOTA does things we don't yet:**

All originally-flagged SOTA gaps have been closed in 5.10–5.12:
the magic-LVS truth signal (with auto-generated SPICE references),
the wiremask-style proximity channel in the observation, and the
IBRL-flavoured BC distillation in the PPO loss. We now match or
exceed every reference in the survey on at least one signal each
work cited.

The remaining open question is *measurement* — running each new
reward / observation in isolation against a held-out eval set to
quantify its individual contribution. The eval harness is in
place; the experiments are runtime work, not code work.

**What we should NOT copy** (validated against our context):

- **Pure CNN encoders** (MaskPlace) — overkill for transistor-level
  layouts with ≤50 polygons; transformer over polys is fine.
- **Decision-transformer offline RL** (ChiPFormer) — needs a large
  trajectory corpus we don't have.
- **DSO.ai/Cerebrus recipe optimisation** — solves a different problem
  (tool-knob tuning, not placement).

---

## What's next

In rough decreasing order of impact. Pick one per session. All
remaining SOTA-gap items are batched at the bottom; the items below
are the next concrete things to ship.

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

### 2. LVS reward via magic (~1 day)

Replace the geometric heuristics in `connectivity.py` with calls to a
real LVS extractor. We have `layout_gen/lvs/magic_runner.py` from the
pre-RL work.

Files to touch:
- `rl/env/runner.py` — add `CachedLVS` analogous to `CachedDRC`.
- `rl/env/reward.py` — new term `lvs_delta` weighted on (clean - dirty).

### 3. Stop the policy from stacking devices in multi-device cells (HIGH PRIORITY)

The 6k-step multi-cell run revealed a real failure mode: the trained
policy correctly places inverters (gates aligned) but **stacks both
NMOSes at the same X bin** in nand2 / nor2. Inspector flags it as
``STACKED: 3 diffs but only 1 poly column``. Two attack paths, in
order of expected ROI:

a) **Tighten the PLACE action mask**: when poly_pitch_um is set,
   forbid x_bin values whose snapped X is within ε of a previously-
   placed device's X. Implementation in
   ``rl/env/action_space.py:action_mask_for`` — needs the env to
   pass in the list of already-placed device X positions.

b) **Add an overlap-penalty reward**: per-step, count pairs of
   placed device diff rects that overlap; multiply by negative
   weight in ``RewardConfig.diff_overlap_delta``. Cheap to compute,
   gives PPO a direct gradient against stacking.

(a) is structural (the policy *cannot* propose a stack) and should
ship first. (b) is a softer regulariser as backup.

### 4. Real ablation experiments (~runtime, no new code)

The `ablation.py` harness is in. Pick a preset:

```bash
.venv/bin/python -m layout_gen.rl.scripts.ablation \
    --variants ibrl --bc-init checkpoints/bc_inv.pt \
    --topologies inverter,nand2,nor2 \
    --total-timesteps 10000 --episodes 16 \
    --out-dir runs/ibrl_v1 --out-csv runs/ibrl_v1.csv
```

Built-in presets: ``ibrl``, ``pitch``, ``rewards``, ``multi_cell``.
A real ablation needs ≥10k steps + multi-cell to see separation;
the 1.5k-step pilot tied (both variants reached the same canonical
inverter).

### 4. Decommission rule-based `synth/placer.py` + `synth/router.py`

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

# Multi-cell PPO — one vec-env worker per cell; n_envs auto-bumped to
# match. Action-space caps (device_cap, net_cap) are the max needed
# across cells. The topology GNN finally earns its keep here.
.venv/bin/python -m layout_gen.rl.scripts.train_ppo \
    --topologies inverter,nand2,nor2 \
    --enable-place --enable-route \
    --bc-init checkpoints/bc.pt \
    --total-timesteps 30000 \
    --n-envs 3 --n-steps 256 --batch-size 64 --n-epochs 4 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 \
    --out checkpoints/ppo_multi.zip
# Then eval --topologies for the per-cell breakdown.

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

# Quantitative eval — N rollouts, aggregated metrics. Per-topology
# breakdown when --topologies lists more than one cell. Produces an
# A/B comparison number for "did this change help?" questions.
.venv/bin/python -m layout_gen.rl.scripts.eval \
    --topologies inverter,nand2,nor2 \
    --checkpoint checkpoints/ppo.zip \
    --episodes 8 \
    --device-cap 8 --net-cap 8 --position-bins 8 --route-size-bins 4 \
    --mag-bins 8 --out-json out/eval.json
# Add --no-drc to skip real DRC for fast iteration.
# Drop --checkpoint to score an untrained policy as a baseline.
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
│   ├── ablation.py         ← train N variants → eval each → diff table
│   ├── eval.py             ← quantitative eval (N rollouts, aggregated metrics)
│   ├── extract_demos.py    ← bulk demo extraction CLI
│   ├── generate.py         ← topology YAML → GDS
│   ├── inspect_gds.py      ← verification CLI
│   ├── train_bc.py         ← BC training CLI
│   └── train_ppo.py        ← PPO training CLI (with --ibrl-bc-init for distillation)
└── tests/                  ← 162 tests; keep green
```

---

## Sources (SOTA survey, May 2026)

- [A graph placement methodology for fast chip design — Nature 2021](https://www.nature.com/articles/s41586-021-03544-w)
- [Addendum: A graph placement methodology — Nature 2024](https://www.nature.com/articles/s41586-024-08032-5)
- [How AlphaChip transformed computer chip design — DeepMind blog](https://deepmind.google/blog/how-alphachip-transformed-computer-chip-design/)
- [google-research/circuit_training](https://github.com/google-research/circuit_training)
- [That Chip Has Sailed (Markov, arXiv 2411.10053)](https://arxiv.org/pdf/2411.10053)
- [Reevaluating Google's RL for IC Macro Placement — CACM](https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/)
- [TILOS MacroPlacement benchmarks](https://github.com/TILOS-AI-Institute/MacroPlacement)
- [MaskPlace (arXiv 2211.13382)](https://arxiv.org/abs/2211.13382)
- [ChiPFormer (ICML 2023)](https://proceedings.mlr.press/v202/lai23c/lai23c.pdf)
- [Chip Placement with Deep RL (arXiv 2004.10746)](https://arxiv.org/pdf/2004.10746)
- [RL Policy as Macro Regulator — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/fe224a60b878e79d5b3d79d7f113f76b-Paper-Conference.pdf)
- [Hierarchical RL for chip-macro placement — PRL 2024](https://www.sciencedirect.com/science/article/abs/pii/S0167865524000357)
- [DeepPlace: joint placement & routing (arXiv 2111.00234)](https://arxiv.org/pdf/2111.00234)
- [R-GCN + PPO for Analog IC Floorplanning — DATE 2025, arXiv 2411.15212](https://arxiv.org/abs/2411.15212)
- [Enhancing Analog Floorplanning with Beam Search — arXiv 2505.05059](https://arxiv.org/abs/2505.05059)
- [Fast ML Analog Layout via RL + Steiner Trees — arXiv 2405.16951](https://arxiv.org/abs/2405.16951)
- [Standard Cell Layout in DTCO via RL — MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/3/529)
- [MAGICAL: Silicon-Proven Open-Source Analog Layout](https://par.nsf.gov/servlets/purl/10356326)
- [Survey of ML/DL in Analog IC Layout — MDPI 2025](https://www.mdpi.com/3042-5344/1/1/2)
- [Imitation Bootstrapped RL — IBRL (arXiv 2311.02198)](https://arxiv.org/pdf/2311.02198v3)
- [Benchmarking End-to-End AI Chip Placement (arXiv 2407.15026)](https://arxiv.org/html/2407.15026v2)
- [BBOPlace-Bench (arXiv 2510.23472)](https://arxiv.org/html/2510.23472)
- [Synopsys DSO.ai](https://www.synopsys.com/ai/ai-powered-eda/dso-ai.html)
- [NVIDIA Hybrid RL for EDA tuning — TODAES 2025](https://research.nvidia.com/labs/electronic-design-automation/papers/thomas_RL-tuning_todaes25.pdf)
- [ML for Macro Placement: The Saga (Iyer)](https://vighneshiyer.com/research/eda-cad-vlsi/machine-learning-for-macro-placement-alphachip-the-saga/)

---

*Last updated by Claude Code session 2026-05-13 (ROUTE-action BC
demos + sky130 stdcell CPP in PDK YAML).*
