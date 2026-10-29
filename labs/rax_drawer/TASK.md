# Task: Open the smart drawer with the SO-101

Open the lower drawer by grasping its horizontal handle and pulling the drawer toward the robot.

## Start and success

- Start a fresh episode with `drawer_reset`; the default arm pose and closed drawer are the intended start. The scene is deterministic; vary `drawer_position_m` from 0 to 0.02 m for alternate valid starts. The seed is recorded as metadata.
- Success requires three things together: the gripper contacts the lower handle while pulling, the drawer opens at least 6 cm from this episode's start, and it is then returned to within 5 mm of that start. Confirm with `drawer_check_success`. Pushing the drawer face does not count.
- The green button is decorative. There is no unlock or lock behavior in this scene.

## Control

- `drawer_observe` returns joint positions and velocities, calibration for `top`, `side`, and `wrist`, their color frames, and a 16-bit overhead depth image in millimeters. `top` is the fixed Gemini 336L depth camera. `side` is the fixed color camera across the workspace. `wrist` is on the gripper. `front` is not a control input.
- The drawer joint and the handle position are not sensor readings. `drawer_check_success` only reports whether the episode succeeded.
- Use `drawer_step` with short actions and observe after each action.
- `vx/vy/vz` are world-frame velocities in m/s. In this scene **positive Y pulls toward the robot and opens the drawer**.
- Negative `gripper_delta` closes the gripper; positive opens it. Wrist rates are in rad/s.
- Actions are clamped by the backend, and one call cannot exceed 2 seconds.

Approach the lower handle, close the gripper around it, and pull along positive Y. If the drawer does not move, adjust the contact or wrist pose and try again. Do not report success until the success tool confirms it.
