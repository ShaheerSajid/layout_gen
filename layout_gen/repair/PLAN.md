# Autonomous DRC Repair Engine — Design Plan

This document is the source of truth for `layout_gen/repair/`. It captures
the goal, constraints, architecture, data strategy, and phasing for the
autonomous DRC repair engine that fabram's synthesizer feeds into.

Branch: `drc-repair-engine` (in both `layout_gen` and `fabram`).

---

## Goal

Given any layout produced by the fabram synthesizer (or any equivalent
source), produce an action sequence that drives the layout to **DRC- and
LVS-clean** without human intervention. The engine should generalise
across CMOS technology nodes (180 nm → 14 nm) and across PDKs (sky130,
gf180, TSMC 65/22/180).

The downstream goal is full memory: the same engine should clean
not just individual cells but the assembled SRAM macro.

---

## Constraints (set the architecture)

1. **Proprietary PDK rules.** Foundry rule values (0.09 µm, 0.27 µm, …)
   cannot appear in training inputs, model weights, or static code. They
   may pass through the engine *at runtime* via the DRC tool's violation
   report, but the trained policy never sees them as fixed constants.

2. **No expert demonstrations.** We can't hire experienced layout engineers
   to label problem→solution pairs. All training data must be
   self-generated.

3. **Memory-domain scope.** Memory uses a finite, well-known set of cell
   primitives (bitcell, sense amp, write driver, decoder, …). We exploit
   this — we do not target arbitrary IC layout.

4. **Closed DRC vocabulary.** Across CMOS techs, every rule maps onto one
   of ~10 universal categories. The engine handles the categories, not
   individual rule numbers.

---

## Mental model: this is a diffusion problem

The training methodology is structurally identical to denoising diffusion
on images:

| Image diffusion | Layout repair |
| --- | --- |
| Image x_0 | Clean layout L_0 |
| Forward: add Gaussian noise k times | Forward: apply k random perturbations |
| Noise level t | Perturbation depth k |
| Network ε_θ(x_t, t) predicts noise | Network π_θ(L_k, k) predicts inverse action |
| Loss ‖ε − ε_θ‖² | Loss ‖a_inv − π_θ(L_k)‖² |
| Reverse: iterate x_t → x_{t-1} | Reverse: iterate apply-fix until DRC clean |
| Conditional (text/class) | Conditional (rule category, primitive) |
| Training data: any image + noise | Training data: any clean layout + perturbation |

That correspondence collapses several earlier architectural questions:

- **No world model** required — DDPM predicts the denoising direction
  directly, without ever simulating future states.
- **No explicit value head** — noise-level conditioning encodes "how far
  from clean" implicitly.
- **No MCTS / beam search at inference** — sampling iterates the network,
  same as DDPM's reverse sampling loop.
- **Multi-noise-level training is mandatory** — training on k=1 only gives
  a model that takes tiny steps; we need every k ∈ [1, K] so one network
  handles all levels.

Architecture (simplified vs. earlier sketch):

```
state encoder ─▶ Denoiser π_θ(state, t) ─▶ action ─▶ apply ─▶ DRC verify ─▶ loop
                       ▲                                         (decreases t)
                       │
                       └── trained on (perturbed, inverse-action) pairs
                            from the inverse-perturbation library
```

Training and inference loops:

```
TRAIN
  loop:
    L_0 ← sample clean seed (open SCL or reference cell)
    k   ← uniform(1, K)
    L_k, [a_1..a_k] ← apply k perturbations
    loss = ‖a_1 − π_θ(L_k, k)‖²
    backprop

INFER
  while DRC violations remain:
    a = π_θ(current_layout, n_violations_remaining)
    apply(a)
```

Why this framing is better than the RL framing it replaces: **free unlimited
data** (the perturbation library is the noise process), **stable training**
(diffusion losses are well-behaved regression, no adversarial dynamics),
and **automatic curriculum** (the model sees every difficulty level at
training; at inference, harder layouts naturally activate higher-t behaviour).


## Why greedy DRC repair fails (worked example)

On `nor2` from our synthesizer we observe 24 `npc.2` and 16 `licon.13`
violations. Single-step greedy fixes oscillate because the rules are
*coupled*:

- Move NPC by +0.005 µm to clear `licon.13`'s 0.085 → 0.09 µm gap.
- That same NPC now overlaps the next polycontact's NPC at < 0.27 µm —
  creates a *new* `npc.2`.
- Fix that `npc.2` by shrinking — re-creates `licon.13` somewhere else.
- Oscillate.

A human engineer's mental shortcut: "all NPC patches in this region
collapse to one merged NPC; both rule classes evaporate at once." A
single high-level action resolves dozens of low-level violations.

The engine needs both **sequence reasoning** (order edits so each
*decreases* total violations) and **action abstraction** (pick the right
granularity — merge cluster vs. move-one-edge).

---

## Universal DRC taxonomy (the abstraction layer)

Every CMOS DRC reduces to one of these:

| Category          | Examples                       | Universal fix primitive             |
| ----------------- | ------------------------------ | ----------------------------------- |
| `width`           | poly.1, m1.1                   | stretch one edge                    |
| `spacing_same`    | poly.2, m1.2                   | push one shape away                 |
| `spacing_cross`   | licon.13 (licon→npc)           | push offending shape from neighbour |
| `enclosure`       | nwell of pdiff, m1 of via      | grow outer or shrink inner          |
| `extension`       | poly endcap over diff          | extend longer-direction edge        |
| `area`            | li1.6, m1.6                    | grow shape until area ≥ required    |
| `merge`           | npc.2 ("merge if less than…")  | union nearby shapes                 |
| `density`         | metal density windows          | fill insertion (global)             |
| `overlap`         | nsdm/psdm exclusivity          | exclude one of two layers           |
| `antenna`         | poly antenna ratio             | tie-down / break route (global)     |

Categories 1–7 are *local* (zone-confined). Categories 8–10 are *global*
(cell- or block-wide) and are scoped out for the first engine.

---

## Layer roles (the abstraction over GDS layers)

Across PDKs, every layer plays one of a small set of *roles*:

- gate poly
- diffusion (n+ / p+, distinguished by implant overlap)
- contact / via (just a position in the metal stack)
- routing metal (m1, m2, …)
- well (n / p)
- implant (n+ / p+)
- well/substrate tap
- merge-mark layer (NPC, RPO, …)

Each PDK YAML (`layout_gen/pdks/<name>.yaml`) maps GDS layer numbers to
roles. The repair engine reads roles, never raw GDS numbers. This is what
makes "fix the licon-to-NPC spacing" work identically across sky130 and
gf180.

---

## Locality property and zone-decomposed repair

The widest CMOS rule horizon is ~1–2 µm (well-spacing); most are < 0.5 µm.
So the **side effects of any geometric edit are bounded** to a *fix zone*
roughly equal to the action's bounding box dilated by the worst rule
distance.

That gives us:

- **Predictable side effects** — before taking an action we can enumerate
  every shape and rule that lives inside the dilated zone.
- **Independent zones** — the conflict graph (violations as nodes; edges
  between violations whose fixes compete for geometric resource) is sparse.
  Connected components are zones. Cross-zone interactions are bounded.

The natural unit of reasoning is therefore neither a single action nor a
single violation but a **zone-level CSP**: given K rules and N polygons in
this region, find the shortest action sequence that satisfies all K.

Empirical zone size in memory cells (estimated, validated in Phase 1):

- Polygons per zone: 5–20
- Rules per zone: 3–8
- Action sequence length per zone: 1–4
- Zones per cell: 5–30
- Cross-zone edges: 0–3 per zone

This factorisation buys orders of magnitude in tractability vs. treating
the whole layout as one big RL problem.

---

## Architecture

```
                  Memory layout
                        │
                        ▼
     ┌──────────────────────────────────────────┐
     │  Primitive recogniser                     │
     │  (mostly deterministic — driven by the    │
     │   connectivity graph; identifies          │
     │   "this region is a 6T bitcell core",     │
     │   "this region is a sense-amp latch")     │
     └──────────────────────────────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────────┐
     │  Zone extractor                           │
     │  Conflict-graph connected components →    │
     │  fix zones, each tagged with the          │
     │  primitive it sits inside                 │
     └──────────────────────────────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────────┐
     │  Per-primitive skill library              │
     │  (~20–30 zone-resolver skills, each       │
     │   trained on its specific zone type;      │
     │   PDK-agnostic by parameterising on       │
     │   deficit values from runtime DRC,        │
     │   never on PDK constants)                 │
     │   • bitcell_cross_couple_skill            │
     │   • bitcell_pass_gate_skill               │
     │   • senseamp_latch_skill                  │
     │   • polycontact_npc_skill                 │
     │   • power_rail_tap_skill                  │
     │   • …                                     │
     └──────────────────────────────────────────┘
                        │
                        ▼
              DRC verify; loop until clean
```

Two classes of model, both small:

- **Recogniser** — small GNN (or hand-coded matcher) over connectivity
  graph; identifies primitive type unambiguously.
- **Skill** — one per primitive zone, narrowly trained, ~few k params.
  Trained from synthetic perturbations of *that exact primitive*.

---

## Are we doing RL?

If primitives and zones are closed-vocabulary, the typical action sequence
per zone is 1–4 steps. We may not need full end-to-end RL. The minimum
viable engine is:

- A **library of deterministic skills**, each implemented as a small
  constraint solver scoped to its zone, parameterised by deficit.
- A **trained dispatcher** that picks the right skill given local context.
- A **trained recogniser** that segments the layout into zones.

Skill-library trade-off vs. end-to-end RL:

|                              | Skill library + classifier | End-to-end RL |
| ---------------------------- | -------------------------- | ------------- |
| Engineering effort           | bounded (~20–30 skills)    | open          |
| Generalisation               | within memory domain       | broader       |
| Sample efficiency            | excellent                  | poor          |
| Debuggability                | high                       | low           |
| Time-to-first-working-system | weeks                      | months        |

**Decision**: start skill-library; design each skill so RL can later layer
on top (skills become sub-policies in a hierarchical RL setup if we want
to graduate beyond memory).

---

## Data strategy (no humans in the loop)

Five sources of training data, none requiring human labelling.

### 1. Inverse perturbation — the killer trick

```
1. Start from a known-DRC-clean layout L.
2. Sample a perturbation a (move edge, shrink polygon, delete enclosure ring …).
3. Apply a → L'.
4. Run DRC on L'. If violations exist, keep the pair.
5. Record (L', a⁻¹) as a labelled (problem, solution) trajectory.
6. k-step variant: apply k perturbations; reverse sequence is the k-step solution.
```

PDK-agnostic, infinite, perfectly labelled, free. The **inverse isn't
always unique** — multiple valid fixes exist for some violations. That's
fine: we train on multiple-fix targets and accept any valid fix at
inference.

### 2. Open standard cell libraries — clean seed corpus

`layout_gen/repair/seeds.py` ingests cells from:

- `${PDK_ROOT}/sky130A/libs.ref/sky130_fd_sc_hd/` (~437 cells)
- additional sky130 SCLs (hs, hdll, lp, ms, ls, hvl)
- `${PDK_ROOT}/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu{7t,9t}5v0/`
- the user-supplied FabRAM reference cells (bitcell, sense_amp, …)

Filtered to memory-relevant primitives via `classify_primitive()`:
inv, buf, nand, nor, and, or, xor, xnor, AOI/OAI, mux, DFFs, latches.

Plus the foundry's own bitcell, sense_amp, write_driver references.

### 3. Rule-based agent as teacher

Once `RuleGeoAgent` covers the six universal local fix categories,
every successful repair it does on real synthesizer output is a labelled
trajectory. Real-distribution, by definition.

### 4. Synthesizer-failure mining

Cells our synthesizer produces with bugs (today: licon.13/npc.2). Hard
cases — no labels, but suitable for RL fine-tuning where the reward
(violation count) doesn't need labels.

### 5. Self-play / iterated improvement

Once a policy is decent, it generates its own training data: successful
trajectories become new supervised examples; failures become harder
curriculum. AlphaGo Zero loop.

---

## Phasing

### Phase 1 — Catalog & taxonomy *(in progress)*

Pure data, no models.

- ✅ `layout_gen/repair/seeds.py` — discover seed layouts.
- ✅ `layout_gen/repair/catalog.py` — empirical rule taxonomy.
- ✅ `layout_gen/repair/build_catalog.py` — runner script.
- ☐ `layout_gen/repair/zones.py` — conflict graph + zone extractor.
- ☐ Coverage matrix: every rule observed → category, with statistics on
  zone size and cross-zone coupling.

**Deliverable**: `layout_gen/repair/data/catalog.yaml`. This data drives
every architecture decision downstream.

### Phase 2 — Strengthen rule-based agent

- Extend `synth/geo/RuleGeoAgent` to cover all six local categories
  (width, spacing-same, spacing-cross, enclosure, extension, area, merge).
- Wire `GeoFixLoop` into the synthesizer's default flow.
- Goal: every existing fabram cell reaches DRC-clean autonomously.
- Side effect: the rule agent becomes the *teacher* for Phase 3.

### Phase 3 — Imitation policy + value head

- Run RuleAgent on hundreds of synthesizer outputs and inverse-perturbed
  seeds across all PDKs. Record full trajectories.
- Train a behaviour-cloning policy on those trajectories. Architecture:
  GNN over polygon graph; nodes = polygons with layer-role + dimension
  features; edges = spatial adjacency. Magnitude inputs normalised by
  deficit.
- Train a value head jointly: `V(s) = remaining steps to clean`.
- Goal: trained policy matches RuleAgent's fix rate on held-out cells.

### Phase 4 — World model + planning fine-tuning

- Learn `f(s, a) → s', Δviol` from the same trajectory data — predicts
  next state without calling the DRC tool. Enables search.
- PPO / actor-critic with reward `−1 per step + bonus(clean)`.
- Inference: lookahead (MCTS / beam search) over depth N using `f` for
  simulation and `V` for leaf evaluation.
- Train on PDK A; eval on PDK B; eval on PDK C. Generalisation claim:
  convergence rate on unseen PDK ≥ 80% of same-PDK.

### Phase 5 — Memory-scale

- Lift to full memory: cell→array→periphery DRCs.
- Hierarchical agent: cell-level vs. inter-cell-boundary fixes.
- Sparse, region-based state representation (flat polygon graph blows
  up at memory scale).

---

## Evaluation metrics

The metrics must match the goal (fast convergence without blowup), not
just "did it eventually clean":

- **Convergence rate** at step budget B (clean fraction vs. B curve).
- **Mean steps-to-clean** on cells that converge.
- **Blow-up incidence** — fraction of steps where Δviolations > 0.
  Target: → 0 in a trained policy.
- **Cross-PDK gap** — convergence rate on held-out PDK / same-PDK rate.
- **Abstraction usage** — fraction of L1/L2 actions vs. L0. A planning
  policy picks the right abstraction; a reactive one drowns in L0.

---

## Open decisions

1. **World-model + MCTS vs. decision transformer.** Both can deliver
   sequence prediction. MCTS+world-model is more interpretable and
   matches our skill-library design; decision transformer is simpler to
   train but less searchable. **Default**: MCTS+world-model in Phase 4;
   keep decision transformer as a fallback.

2. **Zone inflation radius.** Default = worst-rule-distance from active
   PDK (runtime lookup). Per-rule-class sizing is an alternative for
   deep-submicron PDKs where well-spacing dwarfs everything else.
   **Default**: per-rule-class.

3. **Density and antenna rules.** Out of scope for the first engine —
   handled by a separate non-zone agent later, or accepted as out-of-scope.

4. **TSMC SCL ingestion.** Eventually included once the user has the
   libraries unpacked. Same `seeds.py` mechanism, no code changes.

---

## Current status (2026-05-06)

**Phase 1 complete on `drc-repair-engine` branch.**

Built:

- ✅ `repair/__init__.py` — module scaffolding
- ✅ `repair/catalog.py` — `RuleEntry`, `CatalogBuilder`, classifier, parser
- ✅ `repair/seeds.py` — reference + SCL + synthesizer-output seed discovery
- ✅ `repair/build_catalog.py` — multi-tool catalog generator (Magic ∪ KLayout)
- ✅ `repair/zones.py` — conflict graph + zone extractor (with global-rule filter,
  rule-tightened radius, homogeneity stats)
- ✅ `repair/analyze_zones.py` — per-cell + aggregate zone stats CLI
- ✅ `repair/perturb.py` — invertible perturbation library, multi-step
  trajectory generator with round-trip safety
- ✅ `repair/primitives.py` — 19 canonical memory-domain primitives across
  4 families (logic, sequential, memory, periphery), with zone archetypes
- ✅ `repair/data/catalog.yaml` — empirical catalog: 20 distinct rules across
  sky130 + gf180

**Empirical findings from Phase 1**:

- 20 rules total: 7 sky130, 13 gf180.  All map onto the 8-category taxonomy
  except 4 stragglers (`unknown` due to missing description text — fixable
  with one more regex pass).
- Memory-relevant rules concentrated:  ~80% of sky130 synth violations come
  from just 2 rules (`licon.13`, `npc.2`).  Validates the "finite + well-known"
  premise: the engine doesn't need to generalise widely, just to handle this
  small set well.
- Zone-decomposition validation: median 1 zone / cell, 8 violations / zone,
  3.62 µm zone size (after filtering global LU/nwell rules and tightening
  the radius to per-rule horizon).  Locality assumption holds.
- SCL cells produce *homogeneous* zones (1 rule type per zone) — coarse
  abstract action per zone is enough.
- Synth cells produce *heterogeneous* zones (4–8 rule types per zone) —
  these need the planning / sequence reasoning the architecture targets.
- End-to-end inverse-perturbation pipeline works: 20/20 random perturbations
  on the inverter reference cell trigger DRC violations, with the inverse
  trajectory available as the labelled fix.

**Phase 2 (in flight, diffusion-framed)**:

The original Phase 2 plan was "strengthen RuleGeoAgent → wire it in →
generate trajectories."  Diffusion framing keeps the spirit but reorders:

- ✅ `RuleGeoAgent` extended with `merge` and cross-layer `spacing` handlers.
  Reduces inverter violations 21 → 8 in 2 iterations (gets stuck after that;
  expected — the rule agent is the *teacher*, not the final policy).
- ☐ Wire the GeoFixLoop into the synthesizer's default flow with
  conservative iter cap (so the system at least *attempts* repair on
  every cell synth, recording trajectories whether or not it converges).
- ☐ Trajectory recorder format (state + action sequence + final viol count)
  saved to `data/trajectories/`.
- ☐ Trajectory-mining script: for every (clean seed × perturbation depth k),
  run the perturbation library to get a labelled trajectory at noise
  level k.  The teacher is purely the inverse-perturbation pair — the
  rule agent is *secondary* data only used to extend coverage to real
  synthesizer failure modes that aren't simple inverse-perturbations.

**Phase 3 (diffusion training)** — full sweep done, hit data wall:

- ✅ `repair/features.py` — polygon-set + action featurizer (PDK-agnostic).
  + 4 anomaly-aware features per polygon (nearest-neighbour distances,
  density, size z-score within layer cohort).
- ✅ `repair/dataset.py` — torch Dataset / DataLoader with per-step
  trajectory expansion + class weights + D4 symmetry augmentation.
- ✅ `repair/augment.py` — D4 (8-element) layout symmetries.
- ✅ `repair/model.py` — :class:`DRCDenoiser` with kind / target_pointer /
  **target_xy_centroid** / edge / magnitude heads + class-balanced loss
  + LR warmup / cosine decay.
- ✅ `repair/train.py` — CPU-polite (TORCH_NUM_THREADS=2 default) +
  configurable head dim / depth / heads + snap_acc metric.
- ✅ `repair/infer.py` — iterative denoising loop for deployment.
- ✅ `repair/diagnose.py` — kind confusion + top-K target + magnitude L2.

### Training experiments

Two corpora used:
* **small**:   161 trajectories → 291 expanded samples, val=29 (small, noisy)
* **bigger**: 1435 trajectories → 6033 expanded samples, val=603 (reliable)

| | corpus | val | kind% | tgt-top1 | tgt-top10 | mag-L2 | params |
|--|--|--|--|--|--|--|--|
| v0 (orig)            | small  | 6.44 | 37 | n/a   | n/a  | n/a   | 847k |
| v2 (no centroid)     | small  | 2.38 | 43 | 4%    | 18%  | n/a   | 115k |
| v3 (+centroid head)  | small  | 2.39 | 47 | 12%   | 24%  | 0.10  | 119k |
| v4 (50 ep)           | small  | 2.40 | 40 | 11%   | 21%  | 0.066 | 119k |
| v5 (small, 30 ep)    | small  | 2.45 | 32 | 17%   | 26%  | 0.047 | 23k  |
| v6 (small, 50 ep)    | small  | 2.45 | 23 | 15%   | 27%  | 0.058 | 23k  |
| **v7 (small arch, big data)** | bigger | **2.33** | **52** | **23%** | **40%** | 0.052 | 23k |

**Findings**:
1. **Centroid-XY regression head was the biggest single win** — turned
   target top-1 from 4% → 12% (3×).  A smooth 2D regression target is
   far easier to learn from limited data than per-polygon classification.
2. **Smaller model (23k params) generalises better** than 119k on the
   hard heads (target +5%, magnitude 2× better) despite higher val loss.
   Capacity excess on tiny data → memorisation, not generalisation.
3. **Longer training overfits** — v4 (50 ep, 119k) and v6 (50 ep, 23k)
   both regressed vs their 30-ep counterparts.
4. **Kind-head class collapse persists** — even with inverse-frequency
   weights, the model picks 1–2 dominant classes most of the time.
   Class weighting alone isn't strong enough; data scale would help more.

**The data wall** (now confirmed by experiment):  291 samples can't
sustain better than ~17% target top-1 with this architecture.  Bumping
to 1435 trajectories / 6033 samples (v7) lifted target top-1 to 23%
(+35%), target top-10 to 40% (+54%), and kind to 52% (+64%) — clean
data-scale wins on the same architecture.  Magnitude was already
saturated and didn't move.

This validates the diffusion framing: more (clean → perturbed → fix)
trajectories directly improve the model's denoising, no architecture
change required.  Mining further (10k+ trajectories, multi-PDK) is the
clear next lever.

### Knobs that *did* work (locked in for next iteration)
- Per-step trajectory expansion (~2× samples for free)
- Centroid-XY regression auxiliary loss (`lambda_target_xy=1.0`)
- LR warmup + cosine decay
- 4 anomaly-aware per-polygon features
- D4 symmetry augmentation (8× effective data)
- Smaller architecture (hidden=32, n_layers=1 ≈ 23k params)

### Next ideas (when data grows)
- ☐ Mine ~2,000–5,000 trajectories (politely, 1-thread Magic).
- ☐ Focal loss for kind to fight class collapse harder.
- ☐ Per-step k-uniform sampling at mine time (current ~85% k≤2).
- ☐ Self-supervised polygon-mask pretraining for richer geometry priors.

**Phase 4 (RL fine-tuning, optional)**:

- ☐ Use the trained diffusion model as policy initialization.
- ☐ PPO / DPO with reward = -viol_count, against real synthesizer failures.
- ☐ Hard-case mining for cases where the diffusion baseline plateaus.

Already shipped (orthogonal but on the same branch):

- LVS plumbing: `layout_gen/lvs/` — Magic+Netgen runner, reference netlist
  via spice_gen, wired into Synthesizer.  7/7 LVS tests pass on the
  inverter / nand / nor / row_driver family.

---

## File map

```
layout_gen/repair/
  __init__.py            (module docs, design principles)
  PLAN.md                (this file)
  catalog.py             ✅  rule taxonomy, builder, parser
  seeds.py               ✅  reference + SCL + synth seed discovery
  build_catalog.py       ✅  CLI: emit catalog.yaml
  zones.py               ✅  conflict graph + zone extractor
  analyze_zones.py       ✅  zone stats CLI
  perturb.py             ✅  perturbation library + trajectories
  primitives.py          ✅  primitive registry (19 primitives)
  mine_trajectories.py   ✅  produce labelled diffusion training data
  features.py            ✅  layout / action featurization for torch
  dataset.py             ✅  torch Dataset over mined JSONs
  model.py               ✅  DRCDenoiser transformer + denoiser_loss
  train.py               ✅  training CLI (saves to data/denoiser.pt)
  infer.py               ✅  iterative denoising loop for deployment
  data/
    catalog.yaml         ✅  output of build_catalog
    synth_cache/         ✅  per-PDK synth GDS cache
    trajectories/        ✅  mined diffusion training data
    denoiser.pt          ✅  v0 checkpoint (undertrained, baseline)
```
