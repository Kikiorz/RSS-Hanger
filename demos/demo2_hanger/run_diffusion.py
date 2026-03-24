#!/usr/bin/env python3
"""
Diffusion Policy deployment for the hanger whole-body task.

This script keeps the existing ROS deployment skeleton but switches inference to
LeRobot Diffusion Policy. It uses the checkpoint's saved preprocessor and
postprocessor so deployment normalization stays aligned with training.

Inputs are derived from the loaded checkpoint:
- observation.images.* from the checkpoint visual features
- observation.state (14D)
- observation.base_velocity (3D) when enabled by the checkpoint
- observation.effort (14D) when enabled by the checkpoint or runtime flag

Outputs:
- action (17D) = [base 3D, left arm 7D, right arm 7D]

Notes:
- Velocity is intentionally not used.
- Base publishing keeps only x active; y and omega are forced to 0.
- If a torque-enabled checkpoint is loaded, torque can be disabled at runtime
  and the model will receive zeros for observation.effort.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, JointState

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.control_utils import predict_action

DEFAULT_IMAGE_SIZE = (224, 224)
NODE_NAME = "piper_diffusion_hanger"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "data" / "ACTros2" / "run_diffusion_debug"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CKPT = (
    PROJECT_ROOT / "models" / "diffusion_policy_official_base" / "checkpoints" / "200000" / "pretrained_model"
)
VISUAL_FEATURE_TO_CACHE_KEY = {
    "observation.images.main": "main",
    "observation.images.secondary_0": "secondary_0",
    "observation.images.secondary_1": "secondary_1",
    "observation.images.secondary_2": "secondary_2",
}

latest_imgs = {
    "main": None,
    "secondary_0": None,
    "secondary_1": None,
    "secondary_2": None,
}

latest_q = {
    "left": None,
    "right": None,
}

latest_effort = {
    "left": None,
    "right": None,
}

latest_base_velocity = None

smoothed_action = {
    "left": None,
    "right": None,
    "base": None,
}


def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode compressed image")
    return img_bgr


def preprocess_image_for_policy(img_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != image_size:
        img_rgb = cv2.resize(img_rgb, image_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img_rgb)


def cb_main(msg: CompressedImage):
    latest_imgs["main"] = decode_compressed_image(msg)


def cb_secondary_0(msg: CompressedImage):
    latest_imgs["secondary_0"] = decode_compressed_image(msg)


def cb_secondary_1(msg: CompressedImage):
    latest_imgs["secondary_1"] = decode_compressed_image(msg)


def cb_secondary_2(msg: CompressedImage):
    latest_imgs["secondary_2"] = decode_compressed_image(msg)


def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    if msg.effort:
        latest_effort["left"] = np.array(msg.effort, dtype=np.float32)


def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    if msg.effort:
        latest_effort["right"] = np.array(msg.effort, dtype=np.float32)


def cb_odom(msg: Odometry):
    global latest_base_velocity
    latest_base_velocity = np.array(
        [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.angular.z,
        ],
        dtype=np.float32,
    )


def find_latest_pretrained_dir(search_root: Path) -> Path | None:
    candidates = sorted(
        search_root.glob("**/checkpoints/*/pretrained_model/config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parent


def resolve_pretrained_path(ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        candidate = Path(ckpt_arg).expanduser().resolve()
        if candidate.is_file() and candidate.name == "config.json":
            return candidate.parent
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate
        if candidate.is_dir() and (candidate / "pretrained_model" / "config.json").exists():
            return candidate / "pretrained_model"
        if candidate.is_dir():
            nested = find_latest_pretrained_dir(candidate)
            if nested is not None:
                return nested
        raise FileNotFoundError(f"Invalid checkpoint path: {candidate}")

    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT

    latest = find_latest_pretrained_dir(MODELS_DIR)
    if latest is None:
        raise FileNotFoundError("No DP checkpoint found under models/. Please pass --ckpt.")
    return latest


def summarize_array(arr: np.ndarray | None) -> dict | None:
    if arr is None:
        return None
    arr_np = np.asarray(arr)
    return {
        "shape": list(arr_np.shape),
        "min": float(arr_np.min()),
        "max": float(arr_np.max()),
        "mean": float(arr_np.mean()),
        "std": float(arr_np.std()),
    }


def save_debug_snapshot(
    debug_dir: Path | None,
    *,
    step_count: int,
    reason: str,
    visual_features: list[str],
    obs: dict[str, np.ndarray],
    latest_imgs_cache: dict[str, np.ndarray | None],
    state_raw: np.ndarray,
    base_vel_raw: np.ndarray | None,
    effort_raw: np.ndarray | None,
    action_raw: np.ndarray | None,
    infer_ms: float | None,
    loop_dt_ms: float | None,
    chunk_refill: bool,
    queue_len_before: int,
    queue_len_after: int | None,
    exc_text: str | None,
    save_images: bool,
) -> Path | None:
    if debug_dir is None:
        return None

    step_dir = debug_dir / f"step_{step_count:06d}_{reason}"
    step_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step_count,
        "reason": reason,
        "chunk_refill": chunk_refill,
        "queue_len_before": queue_len_before,
        "queue_len_after": queue_len_after,
        "infer_ms": infer_ms,
        "loop_dt_ms": loop_dt_ms,
        "visual_features": visual_features,
        "state": summarize_array(state_raw),
        "base_velocity": summarize_array(base_vel_raw),
        "effort": summarize_array(effort_raw),
        "action_raw": summarize_array(action_raw),
        "obs": {key: summarize_array(value) for key, value in obs.items()},
        "cache": {
            key: None if value is None else {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in latest_imgs_cache.items()
        },
        "exception": exc_text,
        "wall_time": time.time(),
    }
    (step_dir / "snapshot.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if save_images:
        for feature_name in visual_features:
            cache_key = VISUAL_FEATURE_TO_CACHE_KEY[feature_name]
            raw_img = latest_imgs_cache.get(cache_key)
            if raw_img is not None:
                cv2.imwrite(str(step_dir / f"{cache_key}_raw_bgr.jpg"), raw_img)
            policy_img = obs.get(feature_name)
            if policy_img is not None:
                cv2.imwrite(str(step_dir / f"{cache_key}_policy_rgb.jpg"), cv2.cvtColor(policy_img, cv2.COLOR_RGB2BGR))

    return step_dir


def load_policy(pretrained_dir: Path, device: str):
    config = PreTrainedConfig.from_pretrained(pretrained_dir)
    if not isinstance(config, DiffusionConfig):
        raise TypeError(f"Checkpoint at {pretrained_dir} is not a diffusion policy: {type(config).__name__}")
    config.device = device

    policy = DiffusionPolicy.from_pretrained(pretrained_name_or_path=str(pretrained_dir), config=config)
    policy = policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    visual_features = [key for key in policy.config.input_features if key.startswith("observation.images.")]
    missing_visuals = [key for key in visual_features if key not in VISUAL_FEATURE_TO_CACHE_KEY]
    if missing_visuals:
        raise ValueError(f"Unsupported visual features in checkpoint: {missing_visuals}")

    image_size = DEFAULT_IMAGE_SIZE
    if visual_features:
        first_visual = policy.config.input_features[visual_features[0]]
        image_size = tuple(first_visual.shape[-2:])

    rospy.loginfo("=" * 70)
    rospy.loginfo("[INFO] Diffusion policy loaded successfully")
    rospy.loginfo(f"  checkpoint: {pretrained_dir}")
    rospy.loginfo(f"  device: {device}")
    rospy.loginfo(f"  use_base: {policy.config.use_base}")
    rospy.loginfo(f"  use_torque: {policy.config.use_torque}")
    rospy.loginfo(f"  n_obs_steps: {policy.config.n_obs_steps}")
    rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
    rospy.loginfo(f"  horizon: {policy.config.horizon}")
    rospy.loginfo(f"  use_amp: {policy.config.use_amp}")
    rospy.loginfo(f"  visual_features: {visual_features}")
    rospy.loginfo(f"  image_size: {image_size}")
    rospy.loginfo("=" * 70)

    return policy, preprocessor, postprocessor, visual_features, image_size


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy hanger deployment")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=f"Path to pretrained_model directory or run directory (default: {DEFAULT_CKPT})",
    )
    parser.add_argument("--rate", type=float, default=10.0, help="Control frequency in Hz")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override diffusion sampling steps at inference time (default: checkpoint / LeRobot default)",
    )
    parser.add_argument("--use-torque", action="store_true", help="Feed real joint effort into observation.effort")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable EMA smoothing")
    parser.add_argument("--debug-dir", type=str, default=str(DEFAULT_DEBUG_DIR), help="Directory to write debug snapshots")
    parser.add_argument("--disable-debug-snapshots", action="store_true", help="Disable on-disk debug snapshot capture")
    parser.add_argument("--debug-save-images", action="store_true", help="Save raw and preprocessed images in each debug snapshot")
    parser.add_argument("--debug-log-slow-ms", type=float, default=300.0, help="Capture a snapshot when inference exceeds this latency")
    args, _ = parser.parse_known_args()

    rospy.init_node(NODE_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    pretrained_dir = resolve_pretrained_path(args.ckpt)
    policy, preprocessor, postprocessor, visual_features, image_size = load_policy(pretrained_dir, device)

    if args.num_inference_steps is not None:
        if args.num_inference_steps <= 0:
            raise ValueError(f"--num-inference-steps must be positive, got {args.num_inference_steps}")
        policy.diffusion.num_inference_steps = args.num_inference_steps

    actual_num_inference_steps = policy.diffusion.num_inference_steps
    use_base = bool(policy.config.use_base)
    use_torque = args.use_torque
    required_cache_keys = [VISUAL_FEATURE_TO_CACHE_KEY[key] for key in visual_features]
    debug_dir = None if args.disable_debug_snapshots else Path(args.debug_dir).expanduser().resolve()
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if use_torque and not policy.config.use_torque:
        raise ValueError("--use-torque requires a torque-enabled checkpoint (policy.config.use_torque=True)")
    if policy.config.use_torque and not use_torque:
        rospy.logwarn("Checkpoint was trained with torque but --use-torque not set; observation.effort will be zeros.")

    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_secondary_0, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_secondary_1, queue_size=1)
    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)
    rospy.Subscriber("/ranger_base_node/odom", Odometry, cb_odom, queue_size=1)

    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    rate = rospy.Rate(args.rate)
    enable_smoothing = not args.no_smoothing
    smoothing_alpha = args.smoothing
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    rospy.loginfo("=" * 70)
    rospy.loginfo("[CONFIG] Deployment settings:")
    rospy.loginfo(f"  checkpoint: {pretrained_dir}")
    rospy.loginfo(f"  control rate: {args.rate} Hz")
    rospy.loginfo(f"  use_base: {use_base}")
    rospy.loginfo(f"  use_torque: {use_torque}")
    rospy.loginfo(f"  num_inference_steps: {actual_num_inference_steps}")
    rospy.loginfo(f"  smoothing: {enable_smoothing}, alpha={smoothing_alpha}")
    rospy.loginfo(f"  debug_dir: {debug_dir}")
    rospy.loginfo(f"  debug_log_slow_ms: {args.debug_log_slow_ms}")
    rospy.loginfo("=" * 70)
    rospy.loginfo("Waiting for sensor data...")

    data_ready_logged = False
    step_count = 0
    prev_step_wall_time = None

    global smoothed_action

    while not rospy.is_shutdown():
        if any(latest_imgs[cache_key] is None for cache_key in required_cache_keys):
            rate.sleep()
            continue

        if latest_q["left"] is None or latest_q["right"] is None:
            rate.sleep()
            continue

        if use_torque and (latest_effort["left"] is None or latest_effort["right"] is None):
            rate.sleep()
            continue

        if use_base and latest_base_velocity is None:
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo("All required sensors ready, starting inference...")
            data_ready_logged = True

        state_raw = np.concatenate([latest_q["left"], latest_q["right"]], axis=0).astype(np.float32)
        latest_imgs_snapshot = {key: value for key, value in latest_imgs.items()}

        obs = {
            feature_name: preprocess_image_for_policy(latest_imgs_snapshot[VISUAL_FEATURE_TO_CACHE_KEY[feature_name]], image_size)
            for feature_name in visual_features
        }
        obs["observation.state"] = state_raw

        base_vel_raw = None
        if use_base:
            base_vel_raw = latest_base_velocity.astype(np.float32).copy()
            obs["observation.base_velocity"] = base_vel_raw

        effort_raw = None
        if policy.config.use_torque:
            if use_torque:
                effort_raw = np.concatenate([latest_effort["left"], latest_effort["right"]], axis=0).astype(np.float32)
            else:
                effort_raw = np.zeros(14, dtype=np.float32)
            obs["observation.effort"] = effort_raw

        step_wall_start = time.perf_counter()
        queue_len_before = len(policy._queues[ACTION])
        chunk_refill = queue_len_before == 0
        infer_start = time.perf_counter()
        action = None
        try:
            action_tensor = predict_action(
                observation=obs,
                policy=policy,
                device=device_t,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(policy.config.use_amp),
            )
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            queue_len_after = len(policy._queues[ACTION])
        except Exception:
            exc_text = traceback.format_exc()
            snapshot_dir = save_debug_snapshot(
                debug_dir,
                step_count=step_count + 1,
                reason="exception",
                visual_features=visual_features,
                obs=obs,
                latest_imgs_cache=latest_imgs_snapshot,
                state_raw=state_raw,
                base_vel_raw=base_vel_raw,
                effort_raw=effort_raw,
                action_raw=action,
                infer_ms=None,
                loop_dt_ms=None if prev_step_wall_time is None else (step_wall_start - prev_step_wall_time) * 1000.0,
                chunk_refill=chunk_refill,
                queue_len_before=queue_len_before,
                queue_len_after=None,
                exc_text=exc_text,
                save_images=args.debug_save_images,
            )
            rospy.logerr(f"Inference crashed at step {step_count + 1}. Debug snapshot: {snapshot_dir}")
            rospy.logerr(exc_text)
            raise

        if infer_ms >= args.debug_log_slow_ms:
            snapshot_dir = save_debug_snapshot(
                debug_dir,
                step_count=step_count + 1,
                reason="slow",
                visual_features=visual_features,
                obs=obs,
                latest_imgs_cache=latest_imgs_snapshot,
                state_raw=state_raw,
                base_vel_raw=base_vel_raw,
                effort_raw=effort_raw,
                action_raw=action,
                infer_ms=infer_ms,
                loop_dt_ms=None if prev_step_wall_time is None else (step_wall_start - prev_step_wall_time) * 1000.0,
                chunk_refill=chunk_refill,
                queue_len_before=queue_len_before,
                queue_len_after=queue_len_after,
                exc_text=None,
                save_images=args.debug_save_images,
            )
            rospy.logwarn(f"Slow inference snapshot saved to {snapshot_dir}")
        if action.shape[0] != 17:
            rospy.logwarn(f"Invalid action dim: {action.shape[0]}, expected 17")
            rate.sleep()
            continue

        action_base = action[0:3].copy()
        action_left = action[3:10].copy()
        action_right = action[10:17].copy()

        loop_dt_ms = None if prev_step_wall_time is None else (step_wall_start - prev_step_wall_time) * 1000.0
        step_count += 1
        if step_count <= 5 or step_count % 50 == 0:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[DIAG] Step {step_count}")
            rospy.loginfo(f"  Chunk refill:        {chunk_refill} (queue_before={queue_len_before}, queue_after={queue_len_after})")
            rospy.loginfo(f"  Inference time:      {infer_ms:.1f} ms")
            if loop_dt_ms is not None and loop_dt_ms > 0:
                rospy.loginfo(f"  Loop dt / rate:      {loop_dt_ms:.1f} ms / {1000.0 / loop_dt_ms:.2f} Hz")
            rospy.loginfo(f"  Raw state LEFT:      {np.array2string(latest_q['left'], precision=3)}")
            rospy.loginfo(f"  Raw state RIGHT:     {np.array2string(latest_q['right'], precision=3)}")
            if policy.config.use_torque:
                if use_torque:
                    rospy.loginfo(f"  Raw effort LEFT:     {np.array2string(latest_effort['left'], precision=3)}")
                    rospy.loginfo(f"  Raw effort RIGHT:    {np.array2string(latest_effort['right'], precision=3)}")
                else:
                    rospy.loginfo("  Effort:              zeros (--use-torque not set)")
            if use_base and base_vel_raw is not None:
                rospy.loginfo(
                    f"  Raw base velocity:   vx={base_vel_raw[0]:.4f}, vy={base_vel_raw[1]:.4f}, omega={base_vel_raw[2]:.4f}"
                )
            rospy.loginfo(
                f"  Action BASE:         vx={action_base[0]:.4f}, vy={action_base[1]:.4f}, omega={action_base[2]:.4f}"
            )
            rospy.loginfo(f"  Action LEFT:         {np.array2string(action_left, precision=3)}")
            rospy.loginfo(f"  Action RIGHT:        {np.array2string(action_right, precision=3)}")
            rospy.loginfo(f"  Delta LEFT:          {np.array2string(action_left - latest_q['left'][:7], precision=3)}")
            rospy.loginfo(f"  Delta RIGHT:         {np.array2string(action_right - latest_q['right'][:7], precision=3)}")
            rospy.loginfo("=" * 70)

        if enable_smoothing:
            if smoothed_action["left"] is None:
                smoothed_action["left"] = action_left
                smoothed_action["right"] = action_right
                smoothed_action["base"] = action_base
            else:
                smoothed_action["left"] = smoothing_alpha * action_left + (1.0 - smoothing_alpha) * smoothed_action["left"]
                smoothed_action["right"] = smoothing_alpha * action_right + (1.0 - smoothing_alpha) * smoothed_action["right"]
                smoothed_action["base"] = smoothing_alpha * action_base + (1.0 - smoothing_alpha) * smoothed_action["base"]
            action_left = smoothed_action["left"]
            action_right = smoothed_action["right"]
            action_base = smoothed_action["base"]

        cmd_vel = Twist()
        if use_base:
            cmd_vel.linear.x = float(action_base[0])
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
        pub_cmd_vel.publish(cmd_vel)

        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.name = joint_names
        msg_left.position = action_left.tolist()
        pub_left.publish(msg_left)

        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.name = joint_names
        msg_right.position = action_right.tolist()
        pub_right.publish(msg_right)

        prev_step_wall_time = step_wall_start
        rospy.loginfo_throttle(
            2.0,
            f"Actions sent (base={use_base}, torque={use_torque}, base_vx={action_base[0]:.4f}, infer_ms={infer_ms:.1f})",
        )
        rate.sleep()


if __name__ == "__main__":
    main()
