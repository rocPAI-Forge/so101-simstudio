# Proposal: staged success diagnostics + staged resets

**Status:** proposal only (no code in this change set).  
**Tracking:** [issue #25](https://github.com/rocPAI-Forge/so101-simstudio/issues/25).  
**Origin:** review of [emoPointer/supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation) (SuperMimic + OmniReset-style reset banks on ARX5 / mjlab) against so101-simstudio Lab 01 and `labs/rax_drawer` (2026-11).  
**Goal:** record *why* we would adopt pieces of that idea, what must stay frozen, and what future work it unlocks — so a later PR can implement without rewriting published scores.

## Cause (why this exists)

SuperMimic-manipulation is **not** a VLA/IL stack we should port wholesale:

| Their stack | Ours |
|-------------|------|
| State-only RL through a learned world model | LeRobot IL / VLA + teleop; RAX tool programs |
| mjlab / MuJoCo Warp, ~4k parallel worlds, NVIDIA | Standard MuJoCo, ROCm |
| Privileged poses in the policy obs | Lab 01 cameras; RAX RGB-D with drawer/cube poses **evaluator-only** |
| Drawer task: drawer already open → place pod → close | RAX: open → place cube → close from a closed drawer |

What *is* transferable is how they deal with **sparse terminal success** and **horizons longer than a short imagined / scripted window**:

1. **Staged start states (OmniReset-style banks)** — train (or collect) from physically validated mid-task states so credit assignment is shorter; evaluate the full task separately.
2. **Per-stage success reporting** — overall success mixes “easy” mid-task starts with full-task starts; they publish per-category rates (e.g. Drawer overall ~90% vs category-0 full task ~81%).
3. **Pre-grasp bridge** — without an “open hand above object” category, full-task policies never learn to open for grasp.
4. **Process milestones on long drawer episodes** — grasp / lift / release-in-tray / close as explicit flags; failures rewind a stage instead of only a bool at the end.

We already have soft analogues (Home20 frozen poses; rax explore vs locked; PROCESS.md failure notes; occasional reach/close logs). This proposal makes those ideas **first-class and non-confusable with ranking metrics**.

## Two levers (do not merge them)

| Lever | Changes | Safe default for Lab 01 / rax |
|-------|---------|-------------------------------|
| **A. Staged success rates** | Logging / reporting only | Yes — add beside existing scores |
| **B. Staged resets** | Training / collection (or a *separate* eval table) | Only as a new recipe; never replaces Home20 / locked full-horizon scores |

**Invariant:** published ranking numbers stay on the **full-horizon, category-0-style** protocols we already shipped.

## Impact on what already shipped

### Lab 01 / Home20 / 193-episode line (v0.1.3–v0.1.4)

**Keep as-is (authoritative):**

- Protocol: 20 frozen poses × seeds 1000/1001/1002, `reset_arm: home`, gripper 0.85, success = `final_success` (cube in container at episode end). Ledger: [`labs/lab01_pnp_agent/home20.md`](../../labs/lab01_pnp_agent/home20.md).
- Published cells (examples): ACT-follow lineage, xVLA **47/60**, DM0.5 **37/60**, LingBot-VLA 2.0 **25/60**, negatives (JEPA / FastWAM / GR00T as documented). Hub ids and recipes stay valid citations.

**Lever A (diagnostics) — additive:**

- Optional columns such as approach / close / lift / in-box, derived from evaluator or existing proximity logs.
- Mark explicitly as **diagnostic, not ranking**. Narratives in `home20.md` already discuss lift vs close; formal columns reduce hand-wavy comparisons (e.g. VA fair-EEF “~19/60 reach” vs V2 empty-grasp).
- Does **not** require re-training or re-uploading checkpoints.

**Lever B (staged resets) — new recipe only:**

- Physically validated mid-task banks (grasped-on-table, near-container, optional pre-grasp) would change the **train** distribution.
- Models trained that way may still be scored on Home20 for comparison, but the **training story** is a new ledger row — do not rewrite the 193-episode / Home20 table as if the data contract never changed.
- Using mid-task starts as the *eval* start would be a **different exam**; do not substitute it for Home20 `/60`.

### `labs/rax_drawer` (v0.1.5)

**Keep as-is (authoritative):**

- Open/close locked: **16/20** (`open_close.py` + `manifests/test.json`).
- Place locked freeze 3: **15/20** (`rax_place/`, tool-only). Method/process: `RAX_METHOD.md`, `PROCESS.md`, `PROTOCOL.md`.
- Non-privileged contract: controller sees joints + cameras (+ overhead depth); drawer joint / handle / cube poses stay in the evaluator.

**Lever A — additive:**

- On locked (and explore) runs, record stage flags: drawer opened / cube grasped / cube in tray / drawer closed-to-start. Failures become countable modes (matches PROCESS notes such as “go_home sweeps tray shut”).
- Locked **15/20** and **16/20** remain the headline; stage rates are annex diagnostics.
- Prefer evaluator-side flags so the RAX tool API stays `reset` / `observe` / `step` / `check_success` without leaking privileged poses into the controller.

**Lever B — low value for the current scripted RAX proof; high value later:**

- The RAX claim is full-horizon tool-only behavior from a closed drawer. Mid-task resets do not strengthen that claim and can **inflate** success if mistaken for the locked score.
- Staged resets become relevant if/when we add a **learned** drawer policy lab (IL or RL), analogous to their “drawer already open → place → close” curriculum — then train banks and full-horizon locked eval must stay separate ledgers.

## Impact on future work

| Direction | How this proposal helps | What to avoid |
|-----------|-------------------------|---------------|
| Lab 01 data improvement / new collectors | Pre-grasp and mid-task banks target known failure modes (lift, empty grasp) without throwing away Home20 | Mixing bank-trained rows into the Home20 ranking column without a recipe label |
| Lab 02+ long-horizon scenes | Bake stage success + optional OmniReset banks into the lab template early | Only shipping a final bool after months of opaque failures |
| Learned drawer / multi-stage IL | Train on staged resets; always report full-horizon locked (or Home20-like) separately | Quoting overall success that averages mid-task starts with full task |
| SuperMimic / world-model experiments | If pursued, do it as a **side lab** (likely CUDA / mjlab), not by replacing ROCm LeRobot main path | Importing Warp + privileged state obs into rax non-privileged contract |
| Hub / docs hygiene | Checkpoint contracts (obs dims, controller timing) and “refuse mismatched reset bank” patterns from their workflow | Silent overwrite of eval logs or Hub cards |

## Non-goals

- Porting SuperMimic, mjlab, or 4096-way Warp training into `.venv-rocm`.
- Replacing RAX RGB-D with privileged state policies for the published drawer scores.
- Invalidating or “updating” Home20 `/60` or rax locked **15/20** / **16/20** by changing start-state distributions under the same name.
- Treating diagnostic stage rates as a new model-family ranking.

## Suggested implementation phases (later PRs)

1. **Diagnostics only (Lab 01 + rax)** — define stage predicates; emit them in eval JSON/logs; document columns in `home20.md` / rax PROCESS or a small `STAGES.md`; no training changes.
2. **Lab 01 optional mid-task / pre-grasp collector** — physically validate holds; store under lab-bound paths; train as an explicitly named recipe; still Home20-eval for ranking.
3. **Only if a learned drawer lab starts** — OmniReset-style banks + process milestones for training; locked full-horizon remains the ship gate.

Phase 1 is enough to capture most of the *analysis* value with minimal risk.

## References

- Upstream inspiration: [supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation) — `docs/tasks.md` (Drawer categories, pre-grasp), OmniReset-style banks, per-category held-out reporting.
- Local authorities: [`labs/lab01_pnp_agent/home20.md`](../../labs/lab01_pnp_agent/home20.md), [`labs/rax_drawer/README.md`](../../labs/rax_drawer/README.md), [`ROADMAP.md`](../../ROADMAP.md) v0.1.4 / v0.1.5.
