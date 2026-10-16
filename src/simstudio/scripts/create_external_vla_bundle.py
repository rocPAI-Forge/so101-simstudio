"""Create a local LeRobot checkpoint wrapper around a VLA deployment manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.configs import FeatureType
from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from simstudio.policies.external_vla import DeploymentManifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Source deployment manifest JSON")
    parser.add_argument("--output", required=True, help="Output LeRobot policy directory")
    parser.add_argument("--backend", choices=("inprocess", "sidecar"), default="sidecar")
    parser.add_argument("--endpoint", default="http://127.0.0.1:10086")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--request-timeout-s", type=float, default=10.0)
    parser.add_argument("--stale-after-s", type=float, default=2.0)
    args = parser.parse_args()

    try:
        from lerobot_policy_external_vla import (
            ExternalVLAConfig,
            make_external_vla_pre_post_processors,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install the bridge first: "
            "uv pip install --python .venv-rocm/bin/python --no-deps "
            "-e plugins/lerobot_policy_external_vla"
        ) from exc

    source = Path(args.manifest).resolve()
    manifest = DeploymentManifest.load(source)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_manifest = output / "deployment_manifest.json"
    if source != target_manifest:
        shutil.copy2(source, target_manifest)

    state_dim = 6
    if "joint_velocities" in manifest.required_state_fields:
        state_dim = 12
    if "end_effector_xyz" in manifest.required_state_fields:
        state_dim = 15
    input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    input_features.update(
        {
            f"observation.images.{camera}": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, args.camera_height, args.camera_width),
            )
            for camera in manifest.required_cameras
        }
    )
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    config = ExternalVLAConfig(
        manifest_path=target_manifest.name,
        backend=args.backend,
        endpoint=args.endpoint,
        request_timeout_s=args.request_timeout_s,
        stale_after_s=args.stale_after_s,
        chunk_size=manifest.action_horizon,
        n_action_steps=manifest.execution_horizon,
        input_features=input_features,
        output_features=output_features,
        push_to_hub=False,
    )
    config._save_pretrained(output)
    preprocessor, postprocessor = make_external_vla_pre_post_processors(config)
    preprocessor.save_pretrained(
        output,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        output,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )
    print(f"Created external VLA policy bundle: {output}")


if __name__ == "__main__":
    main()
