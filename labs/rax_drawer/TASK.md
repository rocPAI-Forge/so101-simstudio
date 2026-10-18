# Task: Open the smart drawer with the SO-101

Open the lower drawer by grasping its horizontal handle and pulling the drawer toward the robot.

## Start and success

- Start a fresh episode with `drawer_reset`; the default arm pose and closed drawer are the intended start. The scene is deterministic; vary `drawer_position_m` from 0 to 0.02 m for alternate valid starts. The seed is recorded as metadata.
- Success is at least 6 cm of additional drawer travel from that episode's starting position. Confirm with `drawer_check_success`.
- The green button is decorative. There is no unlock or lock behavior in this scene.

## Control

- Inspect the front, top, and wrist camera frames with `drawer_observe`.
- Use `drawer_step` with short actions and observe after each action.
- `vx/vy/vz` are world-frame velocities in m/s. In this scene **positive Y pulls toward the robot and opens the drawer**.
- Negative `gripper_delta` closes the gripper; positive opens it. Wrist rates are in rad/s.
- Actions are clamped by the backend, and one call cannot exceed 2 seconds.

Approach the lower handle, close the gripper around it, and pull along positive Y. If the drawer does not move, adjust the contact or wrist pose and try again. Do not report success until the success tool confirms it.
