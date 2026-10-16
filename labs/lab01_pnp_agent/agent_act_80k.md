# Lab 01 — scripted-agent ACT 80K (after-action)

Date: 2026-09-22  
Related: [data_improvement.md](data_improvement.md) (Phase 1 collector), Hub dataset [alexhegit/so101-simstudio-lab01-pnp-agent](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent), teleop ACT 15-D **32/50 (64%)**. Coverage figures + three-dataset vs success: [data_vs_success.md](data_vs_success.md).

This note freezes **what we trained, how we evaluated, why it failed, and what we will do next**. It does not replace the living §6.4 table in [lab01_pnp.md](../lab01_pnp/lab01_pnp.md).

---

## 1. Dataset

| | |
|---|---|
| Hub / local | `alexhegit/so101-simstudio-lab01-pnp-agent` · `datasets/so101-simstudio-lab01-pnp-agent` |
| Format | LeRobot **v3.0**, 20 Hz, 6-D joint **position** actions, 15-D `observation.state` |
| Size | **100** successful episodes, **50126** frames (mean ~501 frames ≈ 25 s) |
| Collector | [`auto_collect_poc.py`](../lab01_pnp/auto_collect_poc.py): privileged cube XYZ + joint lerp toward `Q_READY` + damped Jacobian IK on `gripperframe` |
| Spawn box | Same as leader/eval: `x∈[0.26,0.34]`, `y∈[0.165,0.235]`, `z=0.0125`, **yaw ±45°** |
| Arm reset at record | `reset_arm=home`, then scripted approach (not leader `follow`) |
| Side car | `spawns.jsonl`: `episode / success / frames / spawn` **XYZ only** (no yaw) |

Collection yield (all attempts in `spawns.jsonl`): **109 success / 233 attempts** (~47%). Only successes were saved to the Hub set (100 after rewind/top-up). Failures were discarded — typically far-Y creep timeout or cube not lifted.

**Not mixed** with teleop `alexhegit/so101-simstudio-lab01-pnp`.

---

## 2. Policy checkpoint

| | |
|---|---|
| Job | `lab01_pnp_act_agent` — do **not** overwrite `lab01_pnp_act` (teleop 50K) |
| Recipe | ACT, 15-D state, `chunk_size=100`, `n_action_steps=100` in the config; eval used **50** |
| Train | DORobot MI300X, **bs=64**, **80K** steps, save every 10K (bs=128 OOM / dual-job stall; restarted) |
| Loss | Last step **0.013** (teleop 50K was ~0.053) |
| Keeper | `/mnt/doscratch/outputs/train/lab01_pnp_act_agent/checkpoints/080000/` |
| Copy on .43 | `outputs/train/lab01_pnp_act_agent/checkpoints/080000/` |
| Halo train | Stopped at ~16K; disk `last` still **10K**. Do not eval that as the 80K policy. |

Low train loss means the policy **cloned this expert**, not that closed-loop pick-and-place is solved.

---

## 3. Eval (visual + counted)

Protocol target for comparison with teleop: `labs/lab01_pnp/configs/rollout_act.yaml`, `reset_arm=follow`, sync, `n_action_steps=50`, full-range random cube (XY + yaw ±45°).

| Run | Host | GL | s/ep (ticks) | n | Success | Log |
|-----|------|----|----------------|---|---------|-----|
| **Invalid** | DORobot | OSMesa | 60 **wall-clock** (~120 ticks @ ~2 Hz) | 50 | **0/50 (0%)** | `/mnt/doscratch/outputs/eval/act_agent_osmesa50.log` |
| Smoke GUI | .43 | GLFW | 60 s / **1200** ticks | 3 | **1/3 (33%)** | `outputs/eval/act_agent_glfw3.log` |
| **Headline** | .43 | GLFW | **30 s / 600 ticks** | 50 | **9/50 (18%)** | `outputs/eval/act_agent_glfw50_30s.log` |

**Do not cite 0/50.** OSMesa on MI300X cannot create an EGL graphics context (`radeonsi: can't create a graphics context on a compute chip`). The eval loop used **wall time** as the episode cap, so ~2 Hz rendering yielded only ~6 s of 20 Hz simulation per “60 s” episode. `eval.py` was later changed to a **tick budget** (`fps × episode_time_s`). The 9/50 GLFW run used that budget (600 ticks at 30 s).

Teleop ACT 15-D 50K, same YAML, 60 s/ep, EGL, n=50: **32/50 (64%)**. The 9/50 run is **30 s/ep**; scripted successes are ~25 s, so duration is in-family, but it is not a second-for-second clone of the 32/50 clock.

Headline funnel (9/50):

`reach_8cm=50/50 → near_4cm=47/50 → close_after_open=47/50 → lift_2cm=14/50 → over_box=10/50 → released_after_lift=9/50 → final_success=9/50`  
`dropped_after_lift=0/50`

GLFW 3-ep funnel was the same shape: 3/3 approach + close, 1/3 lift/place. Human observation: many fails look like **jaw vs cube face misalignment (slip)** or **close without a committed lift**.

---

## 4. Why the failure rate is high

Not “the 100 episodes are noisy.” They are **too consistent in the wrong way**.

1. **Expert ignores cube yaw.** Approach uses `lock_wrist=True` and pins  
   `WRIST_GRASP = (wrist_flex=1.62, wrist_roll=0.17)`. IK is **translation-only** (`mj_jacSite` on XYZ).  
   Across all 100 saved episodes, **at gripper-close**: `wrist_flex=1.620` and `wrist_roll=0.170` with **std 0**. Global `wrist_roll` only spans ~0–0.17 rad (home → that constant), not ±45° face alignment.

2. **Spawn yaw is random; wrist is not.** The cube box includes yaw ±45°, so RGB shows rotated cubes. The action never teaches “rotate the jaws to the faces.” Hard yaws often fail in the collector (~53% discarded); the Hub set is **successes that happened to work with a fixed wrist**. Recovered top-camera poses (see [data_vs_success.md](data_vs_success.md)): saved yaw σ = **15.0°**, mean **−11.7°**, **0/100** episodes in `+30…+45°`. The success filter, not the sampler, is what narrowed the set.

3. **Record vs eval arm start.** Collector: `reset_arm=home` then scripted ready. Eval: `follow` (path-dependent after a fail), matching **leader** teleop, not this expert.

4. **Cloning a privileged XYZ expert** yields low BC loss and reliable *approach* (50/50 reach), then a grasp that slips. Teleop 50 demos include human wrist compensation; that is why 32/50 can survive the same random yaw box.

5. **Do not confuse with eval bugs.** Invalid 0/50 was truncated sim time. After tick-budget + GLFW, the remaining gap (18% vs 64%) is the policy/data story above.

### 4.1 Other causes (ranked against the 9/50 funnel)

The funnel **50 reach → 47 close → 14 lift → 9 place** says the policy *goes there and squeezes*; capture fails. Yaw/wrist is the strongest *single* explanation, not the only one.

| Rank | Mechanism | Why it fits | Why it is not the whole story |
|------|-----------|-------------|-------------------------------|
| 1 | **Jaw vs cube face (yaw)** | Close without lift; RGB yaw varies, actions at close are identical | Eval yaw not logged; unproven until we bin fails by \|yaw\| |
| 2 | **Creep into the cube, then freeze-close** | Expert drives gripperframe to **cube XY center** with wrist locked, then `set_grip` with **arm frozen**. On a diamond yaw that is a corner shove | Same code path as (1); fix by wrapping along face normal, not only `wrist_roll` |
| 3 | **Grasp height `GRASP_Z=0.012`** | Cube z≈0.0125, half-width ~15 mm. Too high → pinch top; too low → table scrape. Both look like “closed, z stays 0.015” | Teleop uses a range of heights; we did not ablate Z |
| 4 | **`reset_arm=home` record vs `follow` eval** | After a fail the next episode starts from a bad pose. 3/50 never even `near_4cm` | Teleop *also* evals `follow` and still 64% because demos were `follow`. First-ep-from-home still failed in the 3-ep GUI |
| 5 | **Success filter** | ~47% collector yield; discarded attempts are exactly hard yaw / far-Y. Hub 100 looks diverse in XY, not in grasp geometry | Mixing those 100 with anything re-teaches the filter |
| 6 | **Visual yaw is noise** | Pixels show rotation; labels do not change wrist. ACT can ignore cube orientation | Follows from (1); new data must make wrist **covary** with visible yaw |
| 7 | **Lift IK `lock_wrist=False`** | Wrist can unwind while lifting | `dropped_after_lift=0`; 9/14 lifts already place. Secondary |
| 8 | **30 s cap / `n_action_steps=50`** | Recovery needs time | 9 full places finished in 30 s; the 33 close-without-lift would not use extra time |
| 9 | **Gripper target / squeeze schedule** | One open→close profile, no re-grasp | Teleop has multi-close; possible but weaker than geometry |
| 10 | **Workspace far-Y / IK singularity** | Collector already timed out on far Y | 50/50 eval still *reached*; not “can’t get there” |

**Place is not the bottleneck** (9/14 lifts succeed). Do not spend the new 50 episodes on extra place variety until grasp yield is fixed.

---

## 5. Dataset defects (keep)

| Defect | Effect |
|--------|--------|
| Fixed wrist at close | No yaw → `wrist_roll` mapping for ACT to imitate |
| Center-drive creep + freeze-close | Corner contacts; cube slip before lift |
| Success-only save without yaw bins | Hard angles under-represented; easy yaws over-represented |
| `spawns.jsonl` missing yaw | Cannot plot success vs angle without re-sim |
| Cube pose not in parquet | Same as teleop; coverage still needs a sidecar |
| `reset_arm=home` vs eval `follow` | Extra start-pose shift vs leader ACT |
| Mixing this 100 with teleop 50 (as-is) | Conflicting grasp teachers (fixed wrist vs human wrist) |

---

## 6. Lessons

- Scripted experts can write clean LeRobot v3 and train ACT to **low loss** without raising closed-loop success.
- **Coverage of cube XY ≠ coverage of grasp geometry.** Yaw must change the **action**, not only the pixels.
- Privileged pose is fine for the generator; if the generator does not use a degree of freedom (wrist roll vs yaw), the policy will not invent it.
- Discarding failures **filters** the set toward “fixed wrist worked,” which looks like 100 successes and still evals poorly on full-range yaw.
- Headless eval on a **compute-only** GPU: OSMesa + wall-clock episode cap **under-simulates**. Count **ticks** (`fps × episode_time_s`). MI300X EGL graphics is not available.
- Never overwrite `lab01_pnp_act`. Quote `n_action_steps`, `reset_arm`, GL backend, and seconds/ticks with every fraction.
- Cursor Agent programs the collector; it is not a 20 Hz servo.

---

## 7. Next dataset: yaw-aware scripted expert (spec)

**Do not** collect 50 more with the current expert. **Do not** mix the current 100 with teleop as the default train set. **Do not** finetune 080000 on that mix.

New Hub/local id (example): `so101-simstudio-lab01-pnp-agent-yaw`. New ACT dir: `lab01_pnp_act_agent_yaw`. Collect on **.43**, train on **DORobot**.

### 7.1 What the expert must do (privileged, still not policy input)

Cube is a square: align jaws to the **nearest face** (yaw wrapped to ±45°), not the raw ±45° heading.

1. **Ready** — same non-overshoot path toward the cube; **do not** snap `WRIST_GRASP` yet.
2. **Pre-grasp wrist** — set `wrist_roll` from cube yaw (and keep `wrist_flex` at a grasp pitch). Hold until roll error is small **before** the last centimeters of XY.
3. **Standoff in the cube frame** — approach along the jaw-closing axis (face normal), stop ~15–25 mm short of center (half-width + margin). Do not drive `gripperframe` into the geometric center while open.
4. **Creep + wrap** — close the remaining XY with wrist **held**; abort/retry if cube XY slips more than ~8 mm (corner hit).
5. **Close** — squeeze; arm mostly frozen, small compliance optional.
6. **Commit lift only if** privileged cube z rose ≥2 cm. Otherwise discard (do not save a push). A +90° alternate-face retry is deferred until one-shot bin yields show it is needed; v1 keeps demonstrations single-attempt and unambiguous.
7. **Place** — existing to-box / lower / open / retreat. Keep wrist while loaded so the grasp does not unwind (`lock_wrist=True` during lift/carry, unlike v1).

IK stays damped-LS on XYZ; orientation is **joint-space wrist**, not 6D Cartesian IK.

### 7.2 Coverage (quota, not “random until 50 luckies”)

Same XY box and yaw ±45° as eval. **Fill bins**, then stop:

| Axis | Bins (example) | Quota |
|------|----------------|-------|
| Yaw (wrapped to ±45°) | 6 equal bins | ≥8 successes each → **48**, plus 2 extra hard-bin top-ups → **~50–56** |
| XY | 2×2 cells in the 8 cm × 7 cm box | No cell empty; far-Y (`y>0.22`) at least ~10 successes (old collector died here) |

If a bin’s attempt yield is &lt;30%, **fix the expert** (standoff / roll sign / Z), do not starve that bin.

**Starts:** default `reset_arm=home` (matches this generator). Add **~10** extra successes from a **jittered** joint start (uniform small offset from home or a recorded fail pose) so eval `follow` is not a complete OOD. Do not require a full leader-style follow corpus in v1.

**Do not train on** the old 100. Optional later ablation only.

### 7.3 Quality gate (success is not enough)

Save only if all hold:

- Cube in box (existing `check_pick_success`)
- Lift z ≥ spawn z + 2 cm before to-box
- Pre-close cube slip &lt; 8 mm
- At close, `|wrist_roll - target_roll|` &lt; ~0.1 rad
- No table-plow (cube z not driven below spawn − 3 mm)

Sidecar `spawns.jsonl` per attempt: `xyz, yaw_deg, yaw_bin, xy_cell, wrist_roll_target, wrist_roll_at_close, cube_slip_mm, lifted, success, frames`.

### 7.4 Train / eval (unchanged comparison)

- Retrain ACT 15-D **from scratch** on the new set only. Document batch/steps when launched (same family as teleop; 50 ep is smaller than 100 — do not blindly copy 80K if loss has already flattened).
- Eval: `rollout_act.yaml`, `follow`, sync, `n_action_steps=50`, n=50, full-range. Prefer GLFW 30–60 s with **tick budget**. Log spawn yaw in the eval printout so we can bin the next funnel.
- Compare to teleop **32/50** and this campaign **9/50**. If lift_2cm still ≪ close_after_open, iterate expert (Z / standoff / retry), not mix-back of v1.

### 7.5 Implementation order

1. Expert: yaw→`wrist_roll` + standoff in cube frame + lift gate + sidecar yaw.  
2. GLFW smoke 5 successes, inspect wrist vs cube.  
3. Stratified collect ~50 on .43.  
4. New ACT on DORobot.  
5. n=50 eval; yaw-binned funnel.

### 7.6 Collection result (2026-09-22)

Implemented as [`auto_collect_yaw.py`](../lab01_pnp/auto_collect_yaw.py) /
[`auto_collect_yaw.cmd`](../lab01_pnp/auto_collect_yaw.cmd), preserving the fixed-wrist
collector as the v1 baseline.

| | Result |
|---|---|
| Dataset | `so101-simstudio-lab01-pnp-agent-yaw` |
| Attempts → saved | **68 → 50** (73.5% yield) |
| Frames / disk | **22933** / ~1.1 GB |
| Yaw success bins | **[9, 8, 8, 8, 8, 9]** |
| XY cells | near/high 18, far/high 15, near/low 9, far/low 8 |
| Yaw range | −44.93° … +44.66° |
| Wrist target range | −0.650 … +1.049 rad (v1 close was always +0.170) |
| Alignment error | median 0.0036 rad, max 0.0115 rad |
| Pre-close slip | median 3.76 mm, max 4.15 mm; all <8 mm |
| Quality | 50/50 lifted and placed; LeRobot v3 validation **PASSED** |

The wrist target is not expected to correlate linearly with raw yaw: cube
faces repeat every 90°, and the solver chooses the nearest equivalent joint
solution after accounting for shoulder pan and the site frame.

Reference commands:

```bash
# .43 one-click: create 50, or resume and fill to 100
./labs/lab01_pnp/collect_yaw.cmd
./labs/lab01_pnp/collect_yaw.cmd expand        # same as auto when 50 already exist
./labs/lab01_pnp/collect_yaw.cmd expand 150    # grow further

# Hub (run where HF auth is available)
LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw \
LAB01_DATASET_CARD=./labs/lab01_pnp_agent/hf_dataset_card_agent_yaw.md \
  ./labs/lab01_pnp/push_dataset.cmd

# DORobot: 100-ep ACT from scratch (2026-09-22: wiped the 50-ep ~32K job)
LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw \
LAB01_DATASET_ROOT=/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw \
LAB01_ACT_OUTPUT=/mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw \
LAB01_ACT_LOG=/mnt/doscratch/outputs/train_act_agent_yaw.log \
LAB01_ACT_JOB_NAME=lab01_pnp_act_agent_yaw \
LAB01_ACT_BATCH_SIZE=64 LAB01_ACT_STEPS=80000 LAB01_ACT_SAVE_FREQ=10000 \
  ./labs/lab01_pnp/train_act.cmd
```

The first yaw run attempted the exact teleop `bs=128` scale, but ACT's
attention allocation filled the 192 GB MI300X at step 0. The active recipe is
`bs=64 × 80K` (5.12M sample updates vs teleop 6.40M), saving every 10K.
