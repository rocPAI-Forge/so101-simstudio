"""Run the RAX drawer positive-control demo in a live three-camera window."""
from __future__ import annotations

import time

import glfw

from labs.rax_drawer.environment import DrawerTaskConfig, RaxDrawerEnv


def main() -> None:
    task = DrawerTaskConfig(render_window=True)
    env = RaxDrawerEnv(task)
    try:
        env.reset(seed=12, include_cameras=False)
        window = env.robot._glfw_window
        print("GUI ready. Running scripted drawer pull.", flush=True)
        time.sleep(1.5)

        def apply(action: dict[str, float], duration_s: float) -> dict:
            result = {}
            elapsed = 0.0
            while elapsed < duration_s and not glfw.window_should_close(window):
                chunk = min(0.25, duration_s - elapsed)
                result = env.step(action, duration_s=chunk, include_cameras=False)
                time.sleep(chunk)
                elapsed += chunk
            print(
                f"sim={env.data.time - env.start_time:.2f}s "
                f"drawer={env.drawer_position():.4f}m action={action}",
                flush=True,
            )
            return result

        apply({"wrist_flex_rate": -1.2}, 0.55)
        apply({"vy": -0.04}, 0.9)
        apply({"gripper_delta": -0.7}, 1.0)
        result = {}
        for _ in range(12):
            result = apply({"vy": 0.04}, 0.25)
            if result.get("success") or glfw.window_should_close(window):
                break
        print(
            f"success={env.check_success()} "
            f"opened={env.drawer_position() - env.start_drawer_position:.4f}m",
            flush=True,
        )
        # Keep the completed state visible until the user closes the window.
        while not glfw.window_should_close(window):
            env.robot._render_glfw()
            time.sleep(0.03)
    finally:
        env.close()


if __name__ == "__main__":
    main()