"""Nintendo Switch Joy-Con teleoperator configuration."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_joycon")
@dataclass
class SO101JoyConTeleopConfig(TeleoperatorConfig):
    """Configuration for the SO-101 Joy-Con teleoperator."""

    side: str = "right"  # left | right
    device: str = "auto"  # auto | left | right

    # Velocity scaling.
    # Translation is axis-aligned: the analog stick is read directly and mapped to a
    # constant end-effector velocity while deflected (no orientation coupling).
    # Effective speed = lin_speed * translation_scale (x/y) or lin_speed * z_scale (z).
    lin_speed: float = 0.04  # Base per-axis linear speed (m/s), matches keyboard teleop
    translation_scale: float = 1.5  # Scale for x/y stick translation
    z_scale: float = 1.0  # Scale for z (up/down) translation
    rotation_scale: float = 0.5  # Scale for gyroscope (tilt) rotation rates

    # Tilt->rotation rate conditioning. The wrist/base rotation is derived by
    # differentiating the IMU orientation, which drifts and is noisy at rest.
    # rotation_deadzone zeroes tiny rates so the arm does not creep when the
    # controller is held still; rotation_max clamps spikes (e.g. IMU settling).
    rotation_deadzone: float = 0.06
    rotation_max: float = 3.0
    # Ignore tilt rotation for this long after connect while the IMU settles.
    rotation_settle_s: float = 0.7

    # Analog-stick decoding (raw 12-bit value, center ~2048).
    # A stick axis only produces motion once it passes the deadzone, which keeps the
    # arm still at rest even if the stick center drifts slightly.
    stick_center: int = 2047
    stick_deadzone: int = 600

    # Per-axis sign inversion. Flip if a direction feels reversed on your unit.
    invert_x: bool = False  # forward/back stick -> vx (reach in/out)
    invert_y: bool = False  # left/right stick -> vy (base swing)
    invert_z: bool = False  # up/down buttons -> vz

    # Gripper control style:
    #   True  -> toggle: press ZR/ZL once to close, press again to open (no holding).
    #   False -> hold:   hold ZR/ZL to close, release to open.
    gripper_toggle: bool = True

    # Button mapping (right Joy-Con defaults)
    gripper_button: str = "zr"  # ZR (right) or ZL (left) - auto-detected based on side

    # One-handed recording controls driven by Joy-Con buttons (coexist with the
    # keyboard evdev controls). Set enable_button_recording: false to disable.
    # Button names: a/b/x/y, plus, minus, home, capture, up/down/left/right.
    # Right Joy-Con has A/Y/Plus; left Joy-Con uses the d-pad + Minus instead.
    enable_button_recording: bool = True
    next_episode_button: str = "a"  # save current episode & go to next (keyboard N/Right)
    restart_episode_button: str = "y"  # cancel current episode & re-record (keyboard R/Left)
    stop_button: str = "plus"  # stop the whole recording session (keyboard Q/Esc)
