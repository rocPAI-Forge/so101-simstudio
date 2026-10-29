"""Headless smoke for the drawer environment and RAX backend contract."""

import argparse
import base64
import json

import numpy as np

from labs.rax_drawer.backend import RaxDrawerBackend, _pack
from labs.rax_drawer.environment import DrawerTaskConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-cameras",
        action="store_true",
        help="render top/side/wrist frames and top depth (set MUJOCO_GL=egl on headless hosts)",
    )
    parser.add_argument(
        "--pull-demo",
        action="store_true",
        help="run a deterministic positive-control pull for three fresh episodes",
    )
    args = parser.parse_args()
    camera_names = DrawerTaskConfig().camera_names if args.with_cameras else ()
    backend = RaxDrawerBackend(DrawerTaskConfig(camera_names=camera_names))
    try:
        names = {tool["name"] for tool in backend.tools()}
        assert names == {"drawer_observe", "drawer_reset", "drawer_step", "drawer_check_success"}

        first, images = backend.call("drawer_reset", {"seed": 7, "include_cameras": args.with_cameras})
        env = backend.env
        if args.with_cameras:
            assert len(images) == len(camera_names) + 1, len(images)
            assert all(image["mimeType"] == "image/png" for image in images)
            assert first["camera_names"] == list(camera_names)
            assert first["depth_image_names"] == ["top"]
            assert set(first["calibration"]) == set(camera_names)
        else:
            assert not images
        assert abs(env.drawer_position()) < 1e-6
        assert not env.check_success()

        # Simulate the previous episode ending with a displaced, moving drawer.
        env.data.qpos[env.qadr] = 0.04
        env.data.qvel[env.dadr] = 0.2
        backend.call("drawer_observe", {"include_cameras": False})
        assert abs(env.drawer_position() - 0.04) < 1e-6

        reset, _ = backend.call(
            "drawer_reset",
            {"seed": 8, "drawer_position_m": 0.01, "include_cameras": False},
        )
        assert abs(env.drawer_position() - 0.01) < 1e-6
        assert not env.check_success()
        assert abs(env.data.qvel[env.dadr]) < 1e-6

        # The evaluator scores travel since reset, not absolute slide position.
        env.data.qpos[env.qadr] = 0.07
        assert "drawer_position_m" not in backend.call("drawer_observe", {"include_cameras": False})[0]
        assert abs(env.drawer_position() - env.start_drawer_position - 0.06) < 1e-6
        assert not backend.call("drawer_check_success", {})[0]["success"]
        env.reached_open = True
        env.pull_contact = True
        env.close_contact = True
        env.data.qpos[env.qadr] = env.start_drawer_position
        backend.call("drawer_observe", {"include_cameras": False})
        assert backend.call("drawer_check_success", {})[0]["success"]

        backend.call("drawer_reset", {"include_cameras": False})
        stepped, _ = backend.call("drawer_step", {"action": {}, "duration_s": 0.1, "include_cameras": False})
        assert not stepped["terminated"]
        assert not stepped["success"]

        wrist_before = env.data.qpos[env.robot.dof_ids["wrist_flex"]]
        backend.call(
            "drawer_step",
            {"action": {"wrist_flex_rate": -0.2}, "duration_s": 0.1, "include_cameras": False},
        )
        wrist_after = env.data.qpos[env.robot.dof_ids["wrist_flex"]]
        assert abs(wrist_after - wrist_before) > 1e-6

        before = env.data.site_xpos[env.robot.ee_site_id].copy()
        moved, _ = backend.call(
            "drawer_step",
            {"action": {"vx": 0.01}, "duration_s": 0.1, "include_cameras": False},
        )
        after = env.data.site_xpos[env.robot.ee_site_id]
        displacement = float(np.linalg.norm(after - before))
        assert displacement > 1e-5, displacement
        assert "end_effector_position_m" not in moved["observation"]
        assert not moved["success"]

        bounded = env._bound_action({"vy": 1.0})
        assert bounded["vy"] == env.task.max_linear_speed_mps

        # Exercise PNG/MCP image packing without requiring a GL display.
        value, blocks = _pack({"cameras": {"front": np.zeros((2, 3, 3), dtype=np.uint8)}})
        assert value["camera_names"] == ["front"]
        assert len(blocks) == 1
        assert base64.b64decode(blocks[0]["data"]).startswith(b"\x89PNG\r\n\x1a\n")

        pull_results = []
        if args.pull_demo:
            approach = (
                ({"wrist_flex_rate": -1.2}, 0.55),
                ({"vy": -0.04}, 0.9),
                ({"gripper_delta": -0.7}, 1.0),
            )
            for seed, start_m in enumerate((0.0, 0.01, 0.02)):
                backend.call(
                    "drawer_reset",
                    {
                        "seed": seed,
                        "drawer_position_m": start_m,
                        "include_cameras": False,
                    },
                )
                for action, duration in approach:
                    backend.call(
                        "drawer_step",
                        {
                            "action": action,
                            "duration_s": duration,
                            "include_cameras": False,
                        },
                    )
                result = None
                pull_chunks = 0
                while pull_chunks < 12:
                    result, _ = backend.call(
                        "drawer_step",
                        {
                            "action": {"vy": 0.04},
                            "duration_s": 0.25,
                            "include_cameras": False,
                        },
                    )
                    pull_chunks += 1
                    if result["success"]:
                        break
                assert result is not None and result["success"], result
                pull_results.append(
                    {
                        "start_m": start_m,
                        "opened_m": env.drawer_position() - env.start_drawer_position,
                        "pull_chunks": pull_chunks,
                    }
                )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "tool_count": len(names),
                    "rendered_cameras": len(images),
                    "drawer_position_m_after_reset": env.drawer_position(),
                    "end_effector_displacement_m": displacement,
                    "success_threshold_m": env.task.success_position_m,
                    "png_mcp_image": "ok",
                    "pull_demo_results": pull_results,
                },
                indent=2,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
