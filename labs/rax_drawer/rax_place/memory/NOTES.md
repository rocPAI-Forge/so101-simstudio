# Exploration notes

Started: `20261030T121858Z`

## Working methods

- Overhead RGB-D finds the dark lower handle and green cube.
- Open: slide out, pitch wrist, XY approach with clearance, squeeze, pull +Y.
- Pick: joint-position Jacobian IK with wrist locked to `[1.62, 0.17]`, accumulate `q`.
- Place: drop `0.035 m` behind the perceived handle at `z=0.10`.
- Close: re-perceive handle, descend, grasp, push to recorded `grasp_xy`.

## Pitfalls

- Do not 3D-servo onto the handle for open; XY-only after wrist pitch.
- Do not re-read joints every IK tick without accumulating commands.
- Switching to velocity mode right after the IK lift can drop the cube.
- Carrying across the open tray can shove the drawer shut; place then fails.

## Explore self-test peaks

- Round 1, home after the pull: **9 / 12**. Locked **4 / 20**.
- Round 2, fold-then-face before the pick: **4 / 12**. Locked **3 / 20**.
- Round 3, deeper drop plus wrist-locked handle close: explore 0–23 was 18/24, held-out 24–43 was 13/20. Locked **15 / 20**.
