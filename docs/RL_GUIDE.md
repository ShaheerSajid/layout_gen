# layout_gen RL — Concepts & Code Walkthrough

A reading guide for our RL-based layout generator. Each concept is
paired with the file / function that implements it under
`layout_gen/rl/`. The first sections build just enough theory; later
sections decode the code we actually run.

Skim it top to bottom on a first pass. After that, jump to whichever
section you need — every component links back to its source.

---

## 0. The problem in one paragraph

Given a topology YAML (devices + nets + placement intent) under
`layout_gen/templates/cells/`, produce a DRC-clean, LVS-clean GDS.
The RL policy plays a 3-phase game per episode:
**PLACE** transistors, **ROUTE** metal, optionally **REPAIR** any
remaining DRC violations. Every step it emits one action; every
step we score the layout (DRC count, LVS clean-ness, wire length,
alignment to YAML intent, ...) and feed Δ-score as reward. The
policy learns to produce layouts that humans would call
"reasonable standard cells."

Why RL at all? Because layout is a *sequential* construction problem
with sparse correctness signals (DRC, LVS) and lots of soft quality
signals (wire length, area). RL is the natural fit when you want a
single agent to learn the joint policy across all those signals.

---

## 1. RL in five minutes

If you've used RL before, skip to §2.

### State, action, reward, episode

A reinforcement-learning problem is a Markov Decision Process:

* **State** `s_t` — everything the agent sees at time `t`. For us:
  the current set of polygons, their types and positions, the
  current DRC violations, and which devices have already been
  placed.
* **Action** `a_t` — what the agent picks. For us: which device to
  place next, what (x, y) bin, what orientation; or which net to
  route, on what layer, what rectangle. See §2.3.
* **Reward** `r_t` — a scalar feedback after the action. We compute
  it from before/after deltas of several metrics (§7).
* **Episode** — one full PLACE→ROUTE→REPAIR run. Either ends
  successfully (DRC clean) or truncates at `max_steps`.

### Policy and return

* The **policy** `πθ(a | s)` is a neural network that maps state
  to action distribution. The parameters `θ` are what we train.
* The agent's goal is to maximise the **return** — the discounted
  sum of future rewards `Σ_t γ^t r_t`. We use `γ ≈ 0.99` by default.
* **PPO** (Proximal Policy Optimisation) is the training algorithm.
  It alternates between (a) rolling out the current policy to
  collect (state, action, reward) tuples, and (b) gradient updates
  on those tuples with a clipped surrogate loss that prevents the
  policy from changing too much per update.

### Why "MaskablePPO"?

A vanilla PPO policy would happily propose actions that don't
even make sense in the current state — e.g., placing a device
that's already been placed, or routing on a layer that doesn't
exist. **MaskablePPO** (from `sb3-contrib`) lets us mask invalid
actions before the categorical distribution is sampled, so the
policy never wastes a step on something structurally impossible.

This is huge for our problem: our `MultiDiscrete` action space is
~16 dims, each with up to 32 categories. Without masking, the
policy spends most of its training budget learning that
out-of-range device indices and wrong-phase action kinds are bad.
With masking, those slots are zero-probability by construction.

Reference: [sb3-contrib's MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

---

## 2. The environment

Code: [layout_gen/rl/env/](../layout_gen/rl/env/)

### 2.1 State — `LayoutState`

Defined in [layout_gen/synth/geo/state.py](../layout_gen/synth/geo/state.py).
A `LayoutState` is essentially a list of `Rect` objects, each with:

```python
Rect(layer="li1", x0=0.5, y0=0.0, x1=0.7, y1=0.2,
     net="OUT",                # which net this wire belongs to
     shape_type="wire")        # "wire" | "device" | ...
```

This is the canonical representation we pass to:
- The DRC runner (it serialises to GDS, runs klayout/magic).
- The LVS runner (same, plus a SPICE reference).
- The observation builder (turns it into tensors for the policy).

### 2.2 Phases — `layout_env.LayoutEnv`

Code: [layout_gen/rl/env/layout_env.py](../layout_gen/rl/env/layout_env.py)

Every episode runs through three phases in order:

```
                ┌────────────┐    all devices placed,
                │   PLACE    │    or max_place_steps hit
                │ (add devs) ├────────────────────┐
                └─────┬──────┘                    │
                      │                           │
                      ▼                           │
                ┌────────────┐                    │
                │   ROUTE    │                    │
                │ (add wires)├────────────────────┤
                └─────┬──────┘                    │
                      │ max_route_steps hit       │
                      ▼                           ▼
                ┌──────────────────────────────────┐
                │            REPAIR                │
                │ (perturb rects until DRC clean)  │
                └──────────────────────────────────┘
                            terminate
                            when DRC clean
                            or max_steps hit
```

The env stores `self._phase` and decides per step which action
kinds are valid. PLACE-phase steps consume a `place_device`
action; ROUTE-phase steps consume `route_segment`; REPAIR
consumes one of the six perturb primitives (shift_edge,
shrink_rect, grow_rect, translate, delete_rect, nudge_offgrid).

### 2.3 Action space — `action_space.ActionSpace`

Code: [layout_gen/rl/env/action_space.py](../layout_gen/rl/env/action_space.py)

We use a single `MultiDiscrete` action space that bundles dims for
all three phases. Each step the policy emits all dims; only the
dims relevant to the current phase actually do anything. This
keeps the action shape constant across phases (a hard requirement
for SB3).

Concretely:

| dim       | size   | meaning |
|-----------|--------|---------|
| `kind`    | 6 → 8  | which action kind (one of 6 REPAIR + place_device + route_segment) |
| `target`  | 128    | polygon index (REPAIR only) |
| `edge`    | 4      | left/right/bottom/top (shift_edge) |
| `sign_x`  | 2      | direction (translate/nudge) |
| `sign_y`  | 2      | direction |
| `mag`     | 8      | log-spaced magnitude in µm (perturbs) |
| `device`  | 16     | device index (PLACE) |
| `x_bin`   | 16     | x position bin (PLACE/ROUTE) |
| `y_bin`   | 16     | y position bin |
| `orient`  | 4      | R0 / MX / MY / R180 (PLACE) |
| `net`     | 16     | net index (ROUTE) |
| `route_layer` | 4  | li1/met1/met2/met3 (ROUTE) |
| `route_w_bin` | 4  | log-spaced rectangle width (ROUTE) |
| `route_h_bin` | 4  | log-spaced rectangle height (ROUTE) |

The flat action vector is ~16 ints. `ActionSpace.decode` turns it
into a typed `EnvAction` dataclass that the env's PLACE / ROUTE /
REPAIR handlers consume.

**Important constraint: dims are independent.** The policy emits
a marginal distribution over each dim, not a joint distribution.
So `P(device=PMOS, y_bin=top_row) ≠ P(device=PMOS) · P(y_bin=top_row)`
in general, but our policy *models it that way*. This is the
factored-action-space limitation that bites us when, e.g., the
policy correctly identifies "the next device is a PMOS" and
correctly identifies "the next y is the top row" but routes the
output of `device` and `y_bin` through different heads with no
coupling — see §9.1.

### 2.4 Pitch-aligned action space

When `--routing-mode std_cell` is set, the env snaps coordinates
to PDK-derived track grids:

* PLACE `x_um` snaps to the nearest **poly pitch** line
  (0.46 µm for sky130 stdcell CPP).
* ROUTE `x_um`, `y_um` snap to **per-layer metal pitch**,
  honouring the layer's `preferred_direction` — horizontal
  layers only snap y (track index), vertical layers only snap x.

This effectively gives the policy a maze-router-style discrete
grid. Pitches come from `rules.poly['pitch_um']` and
`rules.<metal>['width_min_um'] + spacing_min_um` via
`derive_poly_pitch_um` / `derive_metal_pitches_um` /
`derive_metal_directions` (all in `action_space.py`).

`analog` mode keeps poly pitch (gates must sit on a grid) but
lets metals run on the manufacturing grid only.

### 2.5 Observation

Code: [layout_gen/rl/env/observation.py](../layout_gen/rl/env/observation.py)

The observation passed to the policy is a dict of tensors:

| key                | shape         | what it is |
|--------------------|---------------|------------|
| `poly_feats`       | (P, 16)       | per-polygon features (layer one-hot, bbox, net, shape_type) |
| `poly_mask`        | (P,)          | 1 for real polys, 0 for padded slots |
| `viol_feats`       | (V, 8)        | per-DRC-violation features |
| `viol_mask`        | (V,)          | same idea for violations |
| `global_feats`     | (G,)          | cell-bbox, phase, step progress |
| `topology_global`  | (T,)          | (optional) GNN's global cell embedding |
| `wiremask`         | (R, R)        | (optional) proximity map per current device |

`P`, `V` are caps (default 128, 32). Slots beyond the real count
are zeroed; the masks tell the policy where the real data ends.

### 2.6 Termination & truncation

* `terminated = True` iff phase = REPAIR AND DRC is clean.
* `truncated = True` iff `step_count >= max_steps`.

PPO treats these differently — a truncated episode's last-step
value is bootstrapped; a terminated episode's isn't. This matters
for correct return estimation.

---

## 3. The policy network

Code: [layout_gen/rl/policy/network.py](../layout_gen/rl/policy/network.py)

```
            poly_feats     viol_feats     global_feats   topology_global
                │              │              │                │
                ▼              ▼              │                │
       ┌─────────────┐  ┌─────────────┐       │                │
       │ Transformer │  │ Transformer │       │                │
       │  over polys │  │  over viols │       │                │
       └──────┬──────┘  └──────┬──────┘       │                │
              │                │              │                │
              │                ▼              │                │
              │           [masked pool]       │                │
              │                │              │                │
              ▼                ▼              ▼                ▼
            ┌────────────────────────────────────────────────────┐
            │            Concat → trunk MLP → ctx                │
            └─┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬┘
              │    │    │    │    │    │    │    │    │    │    │
              ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
            kind  tgt edge sx  sy  mag  dev  xb  yb orient net ...
            head head head head ...
```

* **Per-token Transformer** over the polygon list — handles
  variable-length state. Padding mask comes from `poly_mask`.
* **Per-token Transformer** over the violation list — same idea.
* **Masked mean** over each — turns variable-length tensors into a
  fixed-size summary.
* **Concat + trunk MLP** — fuses the per-token summaries with the
  global features and (optionally) the topology embedding.
* **Per-dim heads** — one `nn.Linear` per action dim. Each emits
  logits of the right cardinality. `ActionLogits` is the
  NamedTuple bundling them.

### 3.1 Topology conditioning — the GNN

Code: [layout_gen/rl/topology/](../layout_gen/rl/topology/)

The topology GNN turns a `CellTemplate` (one YAML) into a
fixed-length embedding the policy can condition on. Without it,
the policy can't tell which *cell* it's supposed to be building —
the observation only shows the current state.

Architecture: a **bipartite GNN** over (devices ↔ nets), with two
rounds of message passing. Plus an **R-GCN** (Relational Graph
Convolutional Network) branch with per-edge-type weight matrices
over typed device↔device edges from the YAML's `placement_logic`
+ `placement.relations`:

* `align_gate` — devices that share an x (gates aligned vertically).
* `abut_x` — devices that sit adjacent on x.
* `shared_diffusion` — devices that abut AND share an S/D net.

The R-GCN gives the policy a way to "see" structural relations
the alignment reward only nudges toward.

Final output: a `(d_token,)` global embedding the policy MLP
concatenates as conditioning. This is the slot that finally
*matters* in multi-cell training — single-cell training has a
constant topology, so the GNN reduces to a constant bias.

---

## 4. PPO + masking, mechanically

Code: [layout_gen/rl/training/ppo_train.py](../layout_gen/rl/training/ppo_train.py)
+ [layout_gen/rl/policy/sb3.py](../layout_gen/rl/policy/sb3.py)

PPO's gradient step looks like:

```
L(θ) = E[ min(rt(θ)·A, clip(rt(θ), 1-ε, 1+ε)·A) ]
        - c_v · (V(s) - return)²
        + c_e · H(π)
```

where `rt(θ) = π_θ(a|s) / π_θ_old(a|s)` and `A` is the advantage.
The `clip` term keeps the policy from changing too much per
update.

In our codebase, sb3-contrib's `MaskablePPO` does all of this; we
provide:

* The **env factory** (or list of factories for multi-cell). Each
  worker in the vec-env builds its own `LayoutEnv`.
* The **features extractor** — `MaskableLayoutPolicy` in
  `policy/sb3.py`. Wraps our `LayoutPolicy` so SB3 can use it.
* The **action masks** — exposed by `LayoutEnv.action_masks()`,
  computed by `action_mask_for` in `action_space.py`. The mask is
  per-dim: it can say "device 7 is already placed → masked" but
  not "if device=PMOS and y_bin<8 → masked" (see §9.1).

### 4.1 IBRL distillation

Code: same, gated by `--ibrl-bc-init`.

A vanilla "BC warm-start" loads a BC-pretrained policy as PPO's
starting point, then PPO updates freely from there. The risk: PPO
quickly forgets the BC behaviour if the early RL reward signal is
noisy.

**IBRL** (Imitation Bootstrapped RL, [arXiv 2311.02198](https://arxiv.org/pdf/2311.02198v3))
keeps the BC policy *frozen alongside* PPO and adds an extra loss
term:

```
L_IBRL(θ) = β(t) · KL(π_θ(·|s) || π_BC(·|s))
```

with `β(t)` decaying linearly over training (we default
`0.5 → 0.0`). Early in training, π_θ is pulled toward π_BC; later,
the KL pull vanishes and the policy is free to surpass BC. This
is implemented as `distill_loss` in `ppo_train.py`.

---

## 5. Behaviour cloning (BC) pretraining

Code: [layout_gen/rl/training/](../layout_gen/rl/training/)

PPO from scratch on a complex task converges slowly. BC pretraining
gives the policy a head start by imitating known-good
trajectories. For us "known-good" comes from two sources:

### 5.1 Synth-derived demos — `demo_extract.py`

The pre-RL rule-based pipeline (`layout_gen.synth.synthesizer.Synthesizer`)
already produces DRC-clean layouts for every cell template. We
run it on each template, then convert its output into an
action sequence:

* For each placed device, emit one `place_device` action with
  `(device_idx, x_um, y_um, orientation)`.
* For each net with at least one placed terminal, emit one
  `route_segment` action whose rect is the bbox of those
  terminal positions on the net's preferred layer (driven by
  net rail position / YAML hints).

This is the *skeleton* of a layout — synth's actual routing is
often more elaborate, but the bbox-rect skeleton is enough for
BC to teach the policy "this net needs a wire spanning roughly
this area on this layer."

### 5.2 SCL miner — `mine_scl_demos.py`

For sky130 we also reverse-engineer real standard cells. The
miner walks `$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/gds/`,
clusters polygons via `inspect_gds._cluster_devices`, sorts by
row (NMOS first, left-to-right), and matches to the topology
device order. Coordinates are rebased so the leftmost NMOS sits
at (0, 0), matching the synth-derived demos' frame.

### 5.3 Dataset + loss

`PlacementDemoDataset` walks each demo through a simulated env
forward (PLACE + ROUTE), building one `(observation, action,
validity)` triple per step. `validity` per dim is True iff
that dim has a meaningful label for this sample (e.g., REPAIR-
phase dims are False for a PLACE sample).

The `BCTrainer` runs cross-entropy per action dim, weighted by
`LayoutPolicyConfig.loss_weights` and masked by `validity`:

```python
for dim_name, logits in policy(obs)._asdict():
    if dim_name not in targets:
        continue
    per_sample = F.cross_entropy(logits, targets[dim_name], reduction="none")
    per_sample *= validity[dim_name]
    loss += weights[dim_name] * per_sample.mean()
```

Final BC checkpoint loads straight into MaskablePPO as the
starting policy.

---

## 6. The composite reward

Code: [layout_gen/rl/env/reward.py](../layout_gen/rl/env/reward.py)

We compute reward as a *sum of Δs* across many metrics:

| term                | weight (default) | what it measures |
|---------------------|------------------|------------------|
| `drc_delta`         | 0.05 / 0.20 / 1.00 (place/route/repair) | Δ in DRC violation count. Phase-aware: full weight only in REPAIR (PLACE/ROUTE intrinsically add violations until layout is complete). |
| `value_delta`       | 0 / 0.05 / 0.05  | Δ in measured µm-deficit summed across violations |
| `step`              | −0.05            | per-step penalty (discourage stalling) |
| `terminal`          | +5.0             | REPAIR phase reaches DRC-clean |
| `invalid`           | −0.5             | action structurally invalid (rejected) |
| `no_change`         | −0.2             | action ran but didn't change geometry |
| `place_success`     | +1.0             | a valid PLACE that changed state |
| `route_success`     | +0.5             | a valid ROUTE that changed state |
| `connectivity_delta`| ×2.0             | Δ(per-net fraction of terminals touched by same-net wire) |
| `alignment_delta`   | ×1.5             | Δ(YAML directive satisfaction: align_gate / abut_x / origin) |
| `electrical_delta`  | ×3.0             | Δ(per-net "all terminals in one connected component" via union-find) |
| `hpwl_delta`        | ×0.5             | Δ(−Σ HPWL_net) — dense placement-quality signal |
| `row_delta`         | ×1.0             | Δ(per-device row-alignment score: NMOS↔bottom, PMOS↔top) |
| `short_delta`       | ×2.0             | −Δ(short-circuit count, cheap geometric heuristic) |
| `lvs_delta`         | ×1.0             | Δ(prev_mismatches − curr_mismatches) — only when --lvs is on |
| `lvs_clean_bonus`   | + bonus          | +X once when mismatches goes >0 → 0 |

### 6.1 Why so many terms?

Each one targets a *specific* failure mode:

* `drc_delta` answers "is the layout legal?" — but it's a sparse
  signal during PLACE (you can't even tell if DRC will clear until
  most of the layout exists).
* `connectivity_delta` and `electrical_delta` answer "is the
  layout *connected*?" — but they're geometric proxies for LVS,
  not LVS itself.
* `hpwl_delta` says "are wires short?" — captures placement
  quality independent of connectivity.
* `alignment_delta` says "did you respect the YAML's intent?" —
  e.g., gate-align the PMOS over the NMOS.
* `row_delta` says "is each device in its structurally-correct
  row?" — closes the loop the factored action heads leave open
  (see §9.1).
* `lvs_delta` is the *truth* signal — magic+netgen verify the
  netlist of the produced GDS matches the topology graph's
  intended netlist.

In aggregate, they cover both correctness (DRC/LVS) and quality
(HPWL/alignment). Tunable weights mean we can dial each up or
down without rewriting the reward.

### 6.2 Phase-aware weights

The DRC term in particular has different weights per phase. During
PLACE, every device you add introduces transient violations
(missing nwell, missing tap, half-resolved spacing). If we
penalised those at full weight, the policy would learn to *not
place* devices — a degenerate policy. So `drc_delta_per_phase
[place] = 0.05` is heavily damped. REPAIR is the only phase where
violation count is unambiguously the right signal.

---

## 7. The LVS truth signal

Code: [layout_gen/rl/env/spice_ref.py](../layout_gen/rl/env/spice_ref.py),
[layout_gen/lvs/](../layout_gen/lvs/),
[layout_gen/rl/env/runner.py:CachedLVS](../layout_gen/rl/env/runner.py)

LVS (Layout vs. Schematic) is the gold-standard correctness
check. It extracts the netlist from the produced GDS via magic
and compares it against a SPICE reference netlist via netgen.

We auto-generate the reference netlist from the topology graph
(no hand-written reference required per cell):

```python
# layout_gen/rl/env/spice_ref.py
.subckt cmos_inverter VDD GND IN OUT
XN OUT IN GND GND sky130_fd_pr__nfet_01v8 w=0.5u l=0.15u
XP OUT IN VDD VDD sky130_fd_pr__pfet_01v8 w=0.5u l=0.15u
.ends
```

Bulks are inferred: NMOS bulk → bottom power net, PMOS bulk → top
power net. Model names default to sky130 nfet/pfet so magic's
extraction matches netgen's comparison without aliasing.

`CachedLVS` is an LRU wrapper around the magic+netgen runner,
keyed by the layout's geometry hash. Same geometry → cache hit;
otherwise it dumps a temp GDS, runs magic, parses the result.

LVS is *slow* (~1s per distinct geometry). For now we keep it
**off** during PPO training by default and only enable it at eval
and post-generate time. The eval scripts (`eval.py`, `generate
--lvs-check`) wire it in. Pass `train_ppo --lvs` if you want it
in the reward loop — expect ~3× slower steps.

---

## 8. PDK-aware tricks

Code mostly in [action_space.py](../layout_gen/rl/env/action_space.py)
and [layout_gen/pdks/](../layout_gen/pdks/).

* **Track-aligned action grid** — see §2.4.
* **Direction-aware metal snapping** — horizontal layers only
  quantise y (cross-axis), vertical layers only quantise x. This
  matches PDK routing conventions and turns the policy into a
  maze-router on a coarse grid.
* **No-stacking guard** — `_apply_place` rejects placements
  within ε of an existing device origin. Stops the multi-cell
  "two NMOSes at the same (x, y)" failure mode.
* **Auto-emitted SPICE references** — see §7.
* **PDK-agnostic env** — invariant #1 in the plan. The env never
  hardcodes µm constants; everything PDK-specific lives behind
  the DRC/LVS runner interface or in
  `repair/features.py:LAYER_ROLES`.

---

## 9. Open algorithmic challenges

These are the actual hard parts left. Each has a tractable next
step listed.

### 9.1 Factored action dims and dependencies

The `MultiDiscrete` action space factorises:

```
P(device, x_bin, y_bin, orient) = P(device) · P(x_bin) · P(y_bin) · P(orient)
```

But the *real* distribution has dependencies:

* `device` and `y_bin` are correlated: an NMOS should pick a low
  `y_bin`, a PMOS a high one.
* `device` and `x_bin` are correlated through the per-row gate
  order (which device sits where in the row).

A factored policy can't model these directly. Workarounds:

* **(A) Auxiliary reward** (what we do now) — `row_delta` rewards
  type-consistent row choices. Soft signal; the policy converges
  to the right marginals if trained long enough.
* **(B) Hard action mask** — would mask `(device, y_bin)` pairs
  that disagree on row. Blocked: the SB3 action-mask interface is
  per-dim, not joint.
* **(C) Auto-regressive heads** — sample `device` first, then
  condition `y_bin`, `x_bin`, `orient` on the sampled device.
  This is the cleanest fix and is what AlphaChip / many recent
  RL-for-EDA papers use. Requires a custom action distribution.
* **(D) Reject + re-sample in the env** — `_apply_place` rejects
  type-row mismatches. Wastes steps; the invalid penalty teaches
  the policy. Easy to add.

Next step (low risk, high signal): try (D) — extend the
no-stacking guard with a type-row check. If `row_delta` plus
hard rejection still isn't enough on nand2/nor2, escalate to (C).

### 9.2 Sparse vs dense rewards

LVS is the only *truth* signal we have, but it's expensive and
sparse (most layouts have many mismatches; transitions to clean
are rare). The dense rewards (HPWL, connectivity, alignment) are
proxies that *correlate* with correctness but don't guarantee it.

This is the standard imitation/exploration trade-off. Three
levers we have:

* **β scheduling** for IBRL — keep BC distillation strong early
  (when reward is noisy) and decay it as the truth signal kicks
  in.
* **Phase-specific weights** — REPAIR weights `drc_delta` at
  1.0 because by then it IS the truth.
* **Bonuses on clean transitions** — `lvs_clean_bonus`,
  `terminal` reward. Sparse but high-magnitude when they fire.

Open question: do we get enough signal from per-step Δ-LVS, or
should LVS only fire as an end-of-episode bonus? Per-step has
better credit assignment but is N× slower. Episode-end is cheap
but back-propagates poorly.

### 9.3 Action masking limits

Our `action_mask_for` does per-dim masking — disallow placed
devices, disallow out-of-range indices, mask kind dims by phase.
It doesn't do *conditional* masking ("if device=X then y_bin
must be in [a, b]"). That's a structural limit of SB3's masking
interface. The fix is option (C) above.

### 9.4 Curriculum

Right now we train on `--topologies inverter,nand2,nor2`
simultaneously. Round-robin across cells. Inverter is *much*
easier (2 devices); the policy may spend its early training
budget mastering nand2/nor2 and losing inverter performance.

A curriculum would start on inverter only, add nand2 once
inverter is solved, then nor2, then the 3-device variants, etc.
Code-wise: `ablation.py` or a small new script that drives
multiple `train_ppo` calls with widening `--topologies`.

### 9.5 Action-space x_bin resolution vs poly pitch

We bumped `--position-bins` to 16 over a 4 µm cell = 0.25 µm/bin,
finer than sky130 poly pitch 0.46 µm. Good — the policy can now
distinguish adjacent gate columns. But this is still cell-w
dependent. A 1 µm cell at 16 bins = 0.0625 µm/bin (overkill);
an 8 µm cell at 16 bins = 0.5 µm/bin (back to colliding).

Right fix: derive `x_bins` from `cell_width / poly_pitch` so the
grid resolution is constant in tracks-per-bin terms. Open task.

---

## 10. How to read the code

Suggested reading order if you're new:

1. **One YAML** — `templates/cells/inverter.yaml`. Get a feel for
   what the policy is supposed to build.
2. **TopologyGraph** — `rl/topology/parser.py`. How the YAML
   becomes a netlist graph.
3. **LayoutState** — `synth/geo/state.py`. The state
   representation.
4. **The env** — `rl/env/layout_env.py`. Top-down read; the
   `step` method is the main loop.
5. **Action space** — `rl/env/action_space.py`. How the policy's
   raw ints become typed actions.
6. **Reward** — `rl/env/reward.py`. All the terms in one place.
7. **The policy** — `rl/policy/network.py`. The transformer +
   heads.
8. **The topology GNN** — `rl/topology/encoder.py`. The bipartite
   + R-GCN encoder.
9. **The trainer** — `rl/training/ppo_train.py`. PPO + IBRL.
10. **The CLIs** — `rl/scripts/`. Where it all gets wired
    together.

After step 6 you should be able to follow any of the
`rl/scripts/*.py` end-to-end.

### 10.1 Useful test entry points

Every concept above has a unit test that exercises it in isolation:

* `test_env.py` — basic env reset/step contract.
* `test_phased_reward.py` — phase-aware reward weights.
* `test_connectivity.py` — connectivity + electrical scores.
* `test_placement_intent.py` — alignment + row scores.
* `test_pitch_snap.py` — pitch quantisation, direction-aware.
* `test_rgcn.py` — typed-edge R-GCN.
* `test_demo_extract.py` — BC demo extraction + dataset.
* `test_ppo.py` — PPO trainer end-to-end.

Reading a test is usually faster than reading the full module
the first time.

---

## 11. End-to-end pipeline (one command per stage)

```bash
# 1. Extract demos from the rule-based pipeline.
.venv/bin/python -m layout_gen.rl.scripts.extract_demos \
    --templates inverter,nand2,nor2,nand3,nor3,aoi21,oai21,buffer,bit_cell_6t,row_driver \
    --out demos/full/

# 2. BC pretrain.
.venv/bin/python -m layout_gen.rl.scripts.train_bc \
    --demos demos/full/ --epochs 60 --batch-size 8 --lr 1e-3 \
    --enable-place --enable-route --use-topology --topology-dim 64 \
    --device-cap 8 --net-cap 8 --position-bins 16 --route-size-bins 4 \
    --out checkpoints/bc_full16.pt

# 3. PPO train (real DRC, multi-cell, BC + IBRL).
.venv/bin/python -m layout_gen.rl.scripts.train_ppo \
    --topologies inverter,nand2,nor2 \
    --enable-place --enable-route \
    --bc-init      checkpoints/bc_full16.pt \
    --ibrl-bc-init checkpoints/bc_full16.pt \
    --ibrl-beta-start 0.5 --ibrl-beta-end 0.0 \
    --total-timesteps 10000 --n-envs 3 --n-steps 128 \
    --device-cap 8 --net-cap 8 --position-bins 16 --route-size-bins 4 \
    --routing-mode std_cell --device cuda \
    --out checkpoints/ppo_multi3.zip

# 4. Generate + verify.
.venv/bin/python -m layout_gen.rl.scripts.generate \
    --topology inverter --checkpoint checkpoints/ppo_multi3.zip \
    --cell-name inv --out out/inv.gds \
    --device-cap 8 --net-cap 8 --position-bins 16 --route-size-bins 4 \
    --routing-mode std_cell \
    --lvs-check

.venv/bin/python -m layout_gen.rl.scripts.inspect_gds out/inv.gds --strict
```

---

## 12. Glossary

* **PPO** — Proximal Policy Optimisation. Our RL algorithm.
* **MaskablePPO** — sb3-contrib's PPO variant that respects per-
  dim action masks.
* **MultiDiscrete** — gymnasium's action space for "tuple of
  categoricals". Each dim is independent.
* **GNN** — Graph Neural Network. For us: bipartite over (devices,
  nets) plus an R-GCN for typed edges.
* **R-GCN** — Relational GCN. One weight matrix per edge type.
* **BC** — Behaviour Cloning. Supervised pretraining from
  demonstrations.
* **IBRL** — Imitation Bootstrapped RL. Distil from a frozen BC
  policy throughout PPO with a decaying weight.
* **HPWL** — Half-Perimeter Wirelength. Sum of bbox half-perimeters
  across nets. A standard placement-quality metric.
* **DRC** — Design Rule Check. Geometric correctness.
* **LVS** — Layout vs Schematic. Netlist correctness via
  extraction + comparison.
* **CPP** — Contacted Poly Pitch. sky130 stdcell value: 0.46 µm.
* **Track** — A unit of routing space the size of one metal pitch.

---

*This document is hand-written and lags the code. The single
source of truth is the code; if you find a mismatch, prefer the
code and update this doc.*
