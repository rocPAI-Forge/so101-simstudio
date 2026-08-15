# Labs — development principles

**Canonical template:** [lab01_pnp/](lab01_pnp/) (pick-and-place record → train → eval).  
**Agent rule:** [`.cursor/rules/lab-development.mdc`](../.cursor/rules/lab-development.mdc).

## How to read these principles

They are **shared defaults for future labs**, generalized from Lab 01 — **not** a requirement that every lab look identical.

1. **Start from the Lab 01 skeleton** when building `lab02_*` and later.
2. **Keep the strong defaults** below unless there is a clear reason not to.
3. **When a lab’s purpose diverges** (different pipeline, no teleop, no Hub publish, etc.), **adapt deliberately** and write a short “Deviations from Lab 01 template” note in that lab’s runbook: what changed and why.
4. Do not invent a second conflicting “house style” without documenting it; prefer extending this guide.

## Strong defaults (usually apply to any lab)

### Lab-bound vs foundational

| Lives under `labs/<lab_id>/` | Stays at repo root |
|------------------------------|--------------------|
| Runbook (`*.md`) | Foundational record/replay/teleop YAML in `configs/` |
| `*.cmd`, `_env.sh` | Scene MJCF / `configs/scenes/` |
| Hub cards, curves, lab notes | Generic keyboard / Joy-Con / leader templates |
| **Lab-only** measurement / demo YAMLs in `labs/<lab_id>/configs/` | Product features the lab merely *wraps* |

Lab 01 example: `configs/so101_mujoco_pick_leader.yaml` is foundational; demo spawn and policy rename maps live under `labs/lab01_pnp/configs/`.

### Centralized `_env.sh`

- Sectioned knobs (Shared / Record / Train / Eval / Hub — omit sections the lab does not need).
- Scripts: `source` `_env.sh` → run; users override with `LAB##_…`.
- Prefer env overrides over host profile enums (`igpu` / `mi300x`).
- Defaults = safe short path; long reference runs documented as overrides.

### Validation and docs

- Script/wiring changes → **smoke** (not full 50-ep / 50K) unless publishing new reference numbers.
- Temp datasets/outputs via distinct `LAB##_` names; scratch under `.tmp/`.
- Update runbook, `_env`, YAML comments, and Hub cards together when behavior/paths change.

## Lab 01–shaped defaults (use when the lab is similar)

These fit a **teleop → dataset → train → sim eval → Hub** lab. Other lab types may drop or replace them:

| Pattern | Lab 01 practice | When you might diverge |
|---------|-----------------|------------------------|
| One entrypoint per role | Single `eval.cmd` + `POLICY_PATH` / `EVAL_CONFIG` | Lab has no eval, or one fixed policy only |
| Policy-specific eval YAMLs | SmolVLA `rename_map` vs ACT without it | Different observation spaces / no VLA |
| `reset_arm` home vs follow | Leader vs keyboard; optional eval contrast | No arm reset, or real-follower-only labs |
| Hub push scripts + cards | Example `alexhegit/…` + override vars | Internal-only lab, no publish |

## Suggested skeleton (adapt as needed)

```text
labs/labNN_topic/
  _env.sh              # LABNN_* defaults (only sections you need)
  labNN_topic.md       # runbook; include “Deviations” if not Lab-01-shaped
  *.cmd                # only the stages this lab actually runs
  configs/             # lab-only YAMLs (if any)
  hf_*.md              # Hub cards (if publishing)
```

Wrap foundational configs via `_env` instead of forking them into the lab unless the lab truly owns a new product default.
