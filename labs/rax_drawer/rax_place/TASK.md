# Task: Open, place the cube, and close the smart drawer

Use the SO-101 arm in this MuJoCo scene. Open the lower drawer by the horizontal handle, put the green cube into the open tray, then close the drawer again.

## Start and success

- Start every attempt with `drawer_reset`. The default arm pose and a nearly closed drawer are the intended start. You may vary `drawer_position_m` in `[0, 0.02]`.
- The harness places the cabinet differently on each explore episode. You do not receive that layout.
- Success requires all of: handle contact while pulling, at least 6 cm of opening from this episode's start, handle contact while returning, return to within 5 mm of that start, and the green cube resting in the drawer tray (not held). Confirm only with `drawer_check_success`.
- The green button does nothing.

## Control

- `drawer_observe` returns joints, calibration for `top` / `side` / `wrist`, their color frames, and overhead depth in millimeters.
- The drawer joint, handle pose, cube pose, and contact flags are not sensor readings.
- `drawer_step` takes short velocity actions. Positive world Y pulls toward the robot and opens the drawer. Negative `gripper_delta` closes the gripper.
- One call lasts at most 2 seconds.

Explore with short steps, write what worked into `memory/`, then compile `code/open_place_close/controller.py`. Do not use the locked test set while exploring or compiling.
