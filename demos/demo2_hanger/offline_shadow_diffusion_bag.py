#!/usr/bin/env python3
"""Offline shadow deployment for hanger diffusion policy on a ROS1 bag.

This tool mirrors the deployment semantics in `demos/demo2_hanger/run_diffusion.py`
without starting ROS nodes. It:
1. reads a ROS1 bag with `rosbags.highlevel.AnyReader`,
2. replays sensor topics on a fixed timeline,
3. runs the same diffusion-policy inference path offline,
4. writes a ROS2 Humble bag containing all original topics plus inferred commands.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

import cv2
import numpy as np
import torch
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action

IMAGE_SIZE = (224, 224)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

CAM_MAIN = "/realsense_top/color/image_raw/compressed"
CAM_SECONDARY_0 = "/realsense_left/color/image_raw/compressed"
CAM_SECONDARY_1 = "/realsense_right/color/image_raw/compressed"
STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ODOM_TOPIC = "/ranger_base_node/odom"

CMD_VEL_TOPIC = "/cmd_vel"
LEFT_CMD_TOPIC = "/robot/arm_left/vla_joint_cmd"
RIGHT_CMD_TOPIC = "/robot/arm_right/vla_joint_cmd"
DEBUG_ACTION_TOPIC = "/shadow/debug/action_raw"
DEBUG_STATE_TOPIC = "/shadow/debug/observation_state"
DEBUG_BASE_VEL_TOPIC = "/shadow/debug/observation_base_velocity"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "ACTros2"

ROS2_STORE = get_typestore(Stores.ROS2_HUMBLE)


@dataclass
class SensorCache:
    imgs: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {
            "main": None,
            "secondary_0": None,
            "secondary_1": None,
            "secondary_2": None,
        }
    )
    q: dict[str, np.ndarray | None] = field(default_factory=lambda: {"left": None, "right": None})
    effort: dict[str, np.ndarray | None] = field(default_factory=lambda: {"left": None, "right": None})
    base_velocity: np.ndarray | None = None


@dataclass
class SmoothedAction:
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    base: np.ndarray | None = None


def decode_compressed_image(msg) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode compressed image")
    return img_bgr


def preprocess_image_for_policy(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != IMAGE_SIZE:
        img_rgb = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img_rgb)


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

    latest = find_latest_pretrained_dir(MODELS_DIR)
    if latest is None:
        raise FileNotFoundError("No DP checkpoint found under models/. Please pass --ckpt.")
    return latest


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
    return policy, preprocessor, postprocessor


def ns_to_sec(value: int) -> float:
    return value / 1e9


def format_ns_delta(value: int) -> str:
    return f"{ns_to_sec(value):.3f}s"


def ensure_output_path(output_path: Path, force: bool) -> None:
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Output path already exists: {output_path}. Pass --force to overwrite.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()


def make_time_msg(timestamp_ns: int):
    time_cls = ROS2_STORE.types["builtin_interfaces/msg/Time"]
    sec = int(timestamp_ns // 1_000_000_000)
    nanosec = int(timestamp_ns % 1_000_000_000)
    return time_cls(sec=sec, nanosec=nanosec)


def make_header(timestamp_ns: int, frame_id: str = ""):
    header_cls = ROS2_STORE.types["std_msgs/msg/Header"]
    return header_cls(stamp=make_time_msg(timestamp_ns), frame_id=frame_id)


def make_float32_multi_array(values: np.ndarray):
    layout_cls = ROS2_STORE.types["std_msgs/msg/MultiArrayLayout"]
    msg_cls = ROS2_STORE.types["std_msgs/msg/Float32MultiArray"]
    return msg_cls(layout=layout_cls(dim=[], data_offset=0), data=np.asarray(values, dtype=np.float32))


def make_twist_msg(action_base: np.ndarray, use_base: bool):
    vector3_cls = ROS2_STORE.types["geometry_msgs/msg/Vector3"]
    twist_cls = ROS2_STORE.types["geometry_msgs/msg/Twist"]
    linear = vector3_cls(x=0.0, y=0.0, z=0.0)
    angular = vector3_cls(x=0.0, y=0.0, z=0.0)
    if use_base:
        linear.x = float(action_base[0])
        linear.y = 0.0
        angular.z = 0.0
    return twist_cls(linear=linear, angular=angular)


def make_joint_state_msg(timestamp_ns: int, positions: np.ndarray):
    msg_cls = ROS2_STORE.types["sensor_msgs/msg/JointState"]
    return msg_cls(
        header=make_header(timestamp_ns),
        name=list(JOINT_NAMES),
        position=np.asarray(positions, dtype=np.float64),
        velocity=np.asarray([], dtype=np.float64),
        effort=np.asarray([], dtype=np.float64),
    )


def serialize_ros2_message(message, msgtype: str) -> memoryview:
    return ROS2_STORE.serialize_cdr(message, msgtype)


def required_topics(use_base: bool) -> list[str]:
    topics = [CAM_MAIN, CAM_SECONDARY_0, CAM_SECONDARY_1, STATE_LEFT, STATE_RIGHT]
    if use_base:
        topics.append(ODOM_TOPIC)
    return topics


def compute_replay_window(
    bag_path: Path,
    replay_topics: Iterable[str],
    max_duration_s: float | None,
    start_offset_s: float,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    topic_bounds: dict[str, list[int | None]] = {topic: [None, None] for topic in replay_topics}

    with AnyReader([bag_path]) as reader:
        connections = [conn for conn in reader.connections if conn.topic in topic_bounds]
        for conn, timestamp_ns, _ in reader.messages(connections=connections):
            bounds = topic_bounds[conn.topic]
            if bounds[0] is None:
                bounds[0] = timestamp_ns
            bounds[1] = timestamp_ns

    missing = [topic for topic, (start, end) in topic_bounds.items() if start is None or end is None]
    if missing:
        raise RuntimeError(f"Missing required topics in bag: {missing}")

    resolved_bounds = {topic: (int(start), int(end)) for topic, (start, end) in topic_bounds.items() if start is not None and end is not None}
    replay_start_ns = max(start for start, _ in resolved_bounds.values())
    replay_end_ns = min(end for _, end in resolved_bounds.values())

    replay_start_ns += int(start_offset_s * 1e9)
    if max_duration_s is not None:
        replay_end_ns = min(replay_end_ns, replay_start_ns + int(max_duration_s * 1e9))

    if replay_end_ns <= replay_start_ns:
        raise RuntimeError(
            f"Replay window is empty after constraints: start={replay_start_ns}, end={replay_end_ns}"
        )

    return replay_start_ns, replay_end_ns, resolved_bounds


class ShadowReplayRunner:
    def __init__(self, policy, preprocessor, postprocessor, device_t: torch.device, args: argparse.Namespace):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device_t = device_t
        self.args = args
        self.cache = SensorCache()
        self.smoothed_action = SmoothedAction()
        self.inference_steps = 0
        self.published_steps = 0
        self.data_ready_logged = False

    @property
    def use_base(self) -> bool:
        return bool(self.policy.config.use_base)

    @property
    def use_torque_checkpoint(self) -> bool:
        return bool(self.policy.config.use_torque)

    def update_from_message(self, topic: str, msg) -> None:
        if topic == CAM_MAIN:
            self.cache.imgs["main"] = decode_compressed_image(msg)
            return
        if topic == CAM_SECONDARY_0:
            self.cache.imgs["secondary_0"] = decode_compressed_image(msg)
            return
        if topic == CAM_SECONDARY_1:
            self.cache.imgs["secondary_1"] = decode_compressed_image(msg)
            return
        if topic == STATE_LEFT:
            self.cache.q["left"] = np.asarray(msg.position[:7], dtype=np.float32)
            if len(msg.effort) > 0:
                self.cache.effort["left"] = np.asarray(msg.effort[:7], dtype=np.float32)
            return
        if topic == STATE_RIGHT:
            self.cache.q["right"] = np.asarray(msg.position[:7], dtype=np.float32)
            if len(msg.effort) > 0:
                self.cache.effort["right"] = np.asarray(msg.effort[:7], dtype=np.float32)
            return
        if topic == ODOM_TOPIC:
            self.cache.base_velocity = np.array(
                [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.angular.z,
                ],
                dtype=np.float32,
            )

    def sensors_ready(self) -> bool:
        if self.cache.imgs["main"] is None or self.cache.imgs["secondary_0"] is None or self.cache.imgs["secondary_1"] is None:
            return False
        if self.cache.q["left"] is None or self.cache.q["right"] is None:
            return False
        if self.args.use_torque and (self.cache.effort["left"] is None or self.cache.effort["right"] is None):
            return False
        if self.use_base and self.cache.base_velocity is None:
            return False
        return True

    def maybe_predict(self, timestamp_ns: int) -> dict[str, object] | None:
        if not self.sensors_ready():
            return None

        if not self.data_ready_logged:
            print("All required sensors ready, starting offline inference...")
            self.data_ready_logged = True

        main_img = preprocess_image_for_policy(self.cache.imgs["main"])
        secondary_0 = preprocess_image_for_policy(self.cache.imgs["secondary_0"])
        secondary_1 = preprocess_image_for_policy(self.cache.imgs["secondary_1"])
        secondary_2 = (
            preprocess_image_for_policy(self.cache.imgs["secondary_2"])
            if self.cache.imgs["secondary_2"] is not None
            else main_img
        )

        state_raw = np.concatenate([self.cache.q["left"], self.cache.q["right"]], axis=0).astype(np.float32)
        obs = {
            "observation.images.main": main_img,
            "observation.images.secondary_0": secondary_0,
            "observation.images.secondary_1": secondary_1,
            "observation.images.secondary_2": secondary_2,
            "observation.state": state_raw,
        }

        base_vel_raw = None
        if self.use_base:
            base_vel_raw = self.cache.base_velocity.astype(np.float32).copy()
            obs["observation.base_velocity"] = base_vel_raw

        if self.use_torque_checkpoint:
            if self.args.use_torque:
                effort_raw = np.concatenate([self.cache.effort["left"], self.cache.effort["right"]], axis=0).astype(np.float32)
            else:
                effort_raw = np.zeros(14, dtype=np.float32)
            obs["observation.effort"] = effort_raw

        action_tensor = predict_action(
            observation=obs,
            policy=self.policy,
            device=self.device_t,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=bool(self.policy.config.use_amp),
        )

        self.inference_steps += 1
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        if action.shape[0] != 17:
            print(f"Warning: invalid action dim at step {self.inference_steps}: {action.shape[0]} (expected 17)")
            return None

        raw_action = action.copy()
        action_base = action[0:3].copy()
        action_left = action[3:10].copy()
        action_right = action[10:17].copy()

        if not self.args.no_smoothing:
            alpha = self.args.smoothing
            if self.smoothed_action.left is None:
                self.smoothed_action.left = action_left
                self.smoothed_action.right = action_right
                self.smoothed_action.base = action_base
            else:
                self.smoothed_action.left = alpha * action_left + (1.0 - alpha) * self.smoothed_action.left
                self.smoothed_action.right = alpha * action_right + (1.0 - alpha) * self.smoothed_action.right
                self.smoothed_action.base = alpha * action_base + (1.0 - alpha) * self.smoothed_action.base
            action_left = self.smoothed_action.left
            action_right = self.smoothed_action.right
            action_base = self.smoothed_action.base

        self.published_steps += 1
        if self.published_steps <= 5 or self.published_steps % self.args.log_every == 0:
            print(
                f"[shadow] step={self.published_steps} t={ns_to_sec(timestamp_ns):.3f}s "
                f"base_vx={float(action_base[0]):.4f} state_norm={float(np.linalg.norm(state_raw)):.4f}"
            )

        return {
            "cmd_vel": make_twist_msg(action_base, self.use_base),
            "left_cmd": make_joint_state_msg(timestamp_ns, action_left),
            "right_cmd": make_joint_state_msg(timestamp_ns, action_right),
            "debug_action": make_float32_multi_array(raw_action),
            "debug_state": make_float32_multi_array(state_raw),
            "debug_base_velocity": make_float32_multi_array(
                base_vel_raw if base_vel_raw is not None else np.zeros(3, dtype=np.float32)
            ),
        }


def add_connection_once(
    writer: Writer,
    connection_map: dict[tuple[str, str], object],
    topic: str,
    msgtype: str,
    typestore,
):
    key = (topic, msgtype)
    if key not in connection_map:
        connection_map[key] = writer.add_connection(topic=topic, msgtype=msgtype, typestore=typestore)
    return connection_map[key]


def ensure_ros2_msgtype_registered(reader: AnyReader, dst_typestore, src_msgtype: str) -> str:
    msgtype = src_msgtype
    if msgtype not in dst_typestore.fielddefs:
        typs = get_types_from_msg(
            reader.typestore.generate_msgdef(src_msgtype, ros_version=2)[0],
            msgtype,
        )
        _ = typs.pop("std_msgs/msg/Header", None)
        dst_typestore.register(typs)
    return msgtype


def flush_timestamp_group(
    group_timestamp_ns: int,
    group_items: list[tuple[object, bytes]],
    reader: AnyReader,
    writer: Writer,
    writer_connections: dict[tuple[str, str], object],
    replay_runner: ShadowReplayRunner,
    replay_connections: dict[str, object],
    replay_topics_set: set[str],
    next_step_ns: int,
    replay_end_ns: int,
    step_ns: int,
    ros2_copy_typestore,
) -> int:
    while next_step_ns < group_timestamp_ns and next_step_ns <= replay_end_ns:
        result = replay_runner.maybe_predict(next_step_ns)
        if result is not None:
            write_replay_messages(writer, replay_connections, next_step_ns, result, replay_runner.args.write_debug_topics)
        next_step_ns += step_ns

    for conn, raw in group_items:
        msgtype = ensure_ros2_msgtype_registered(reader, ros2_copy_typestore, conn.msgtype)
        writer_conn = add_connection_once(writer, writer_connections, conn.topic, msgtype, ros2_copy_typestore)
        msg = reader.deserialize(raw, conn.msgtype)
        writer.write(writer_conn, group_timestamp_ns, ros2_copy_typestore.ros1_to_cdr(raw, msgtype))
        if conn.topic in replay_topics_set:
            replay_runner.update_from_message(conn.topic, msg)

    while next_step_ns == group_timestamp_ns and next_step_ns <= replay_end_ns:
        result = replay_runner.maybe_predict(next_step_ns)
        if result is not None:
            write_replay_messages(writer, replay_connections, next_step_ns, result, replay_runner.args.write_debug_topics)
        next_step_ns += step_ns

    return next_step_ns


def write_replay_messages(writer: Writer, replay_connections: dict[str, object], timestamp_ns: int, result: dict[str, object], write_debug_topics: bool) -> None:
    writer.write(replay_connections[CMD_VEL_TOPIC], timestamp_ns, serialize_ros2_message(result["cmd_vel"], "geometry_msgs/msg/Twist"))
    writer.write(
        replay_connections[LEFT_CMD_TOPIC],
        timestamp_ns,
        serialize_ros2_message(result["left_cmd"], "sensor_msgs/msg/JointState"),
    )
    writer.write(
        replay_connections[RIGHT_CMD_TOPIC],
        timestamp_ns,
        serialize_ros2_message(result["right_cmd"], "sensor_msgs/msg/JointState"),
    )

    if write_debug_topics:
        writer.write(
            replay_connections[DEBUG_ACTION_TOPIC],
            timestamp_ns,
            serialize_ros2_message(result["debug_action"], "std_msgs/msg/Float32MultiArray"),
        )
        writer.write(
            replay_connections[DEBUG_STATE_TOPIC],
            timestamp_ns,
            serialize_ros2_message(result["debug_state"], "std_msgs/msg/Float32MultiArray"),
        )
        writer.write(
            replay_connections[DEBUG_BASE_VEL_TOPIC],
            timestamp_ns,
            serialize_ros2_message(result["debug_base_velocity"], "std_msgs/msg/Float32MultiArray"),
        )


def convert_bag(args: argparse.Namespace) -> None:
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    if bag_path.suffix != ".bag":
        raise ValueError(f"Expected a ROS1 .bag file, got: {bag_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (DEFAULT_OUTPUT_ROOT / f"{bag_path.stem}_shadow_ros2").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_path(output_path, args.force)

    if args.rate <= 0:
        raise ValueError(f"--rate must be positive, got {args.rate}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    pretrained_dir = resolve_pretrained_path(args.ckpt)
    policy, preprocessor, postprocessor = load_policy(pretrained_dir, device)

    if args.use_torque and not policy.config.use_torque:
        raise ValueError("--use-torque requires a torque-enabled checkpoint (policy.config.use_torque=True)")
    if policy.config.use_torque and not args.use_torque:
        print("Warning: checkpoint uses torque but --use-torque is not set; observation.effort will be zeros.")

    replay_start_ns, replay_end_ns, bounds = compute_replay_window(
        bag_path=bag_path,
        replay_topics=required_topics(bool(policy.config.use_base)),
        max_duration_s=args.max_duration,
        start_offset_s=args.start_offset,
    )
    step_ns = max(1, int(round(1e9 / args.rate)))

    print("=" * 72)
    print("Offline diffusion shadow replay")
    print(f"  bag: {bag_path}")
    print(f"  output: {output_path}")
    print(f"  checkpoint: {pretrained_dir}")
    print(f"  device: {device}")
    print(f"  rate: {args.rate} Hz ({step_ns} ns)")
    print(f"  use_base: {bool(policy.config.use_base)}")
    print(f"  use_torque: runtime={args.use_torque}, checkpoint={bool(policy.config.use_torque)}")
    print(f"  replay window: {format_ns_delta(replay_start_ns)} -> {format_ns_delta(replay_end_ns)}")
    for topic, (topic_start, topic_end) in bounds.items():
        print(f"  topic window {topic}: {format_ns_delta(topic_start)} -> {format_ns_delta(topic_end)}")
    print("=" * 72)

    replay_runner = ShadowReplayRunner(policy, preprocessor, postprocessor, device_t, args)

    with AnyReader([bag_path]) as reader, Writer(output_path, version=8) as writer:
        ros2_copy_typestore = get_typestore(Stores.EMPTY)
        ros2_copy_typestore.register({
            **reader.typestore.fielddefs,
            "std_msgs/msg/Header": get_typestore(Stores.ROS2_HUMBLE).fielddefs["std_msgs/msg/Header"],
        })

        replay_connections = {
            CMD_VEL_TOPIC: writer.add_connection(CMD_VEL_TOPIC, "geometry_msgs/msg/Twist", typestore=ROS2_STORE),
            LEFT_CMD_TOPIC: writer.add_connection(LEFT_CMD_TOPIC, "sensor_msgs/msg/JointState", typestore=ROS2_STORE),
            RIGHT_CMD_TOPIC: writer.add_connection(RIGHT_CMD_TOPIC, "sensor_msgs/msg/JointState", typestore=ROS2_STORE),
        }
        writer_connections: dict[tuple[str, str], object] = {
            (CMD_VEL_TOPIC, "geometry_msgs/msg/Twist"): replay_connections[CMD_VEL_TOPIC],
            (LEFT_CMD_TOPIC, "sensor_msgs/msg/JointState"): replay_connections[LEFT_CMD_TOPIC],
            (RIGHT_CMD_TOPIC, "sensor_msgs/msg/JointState"): replay_connections[RIGHT_CMD_TOPIC],
        }
        if args.write_debug_topics:
            replay_connections[DEBUG_ACTION_TOPIC] = writer.add_connection(
                DEBUG_ACTION_TOPIC, "std_msgs/msg/Float32MultiArray", typestore=ROS2_STORE
            )
            replay_connections[DEBUG_STATE_TOPIC] = writer.add_connection(
                DEBUG_STATE_TOPIC, "std_msgs/msg/Float32MultiArray", typestore=ROS2_STORE
            )
            replay_connections[DEBUG_BASE_VEL_TOPIC] = writer.add_connection(
                DEBUG_BASE_VEL_TOPIC, "std_msgs/msg/Float32MultiArray", typestore=ROS2_STORE
            )
            writer_connections[(DEBUG_ACTION_TOPIC, "std_msgs/msg/Float32MultiArray")] = replay_connections[DEBUG_ACTION_TOPIC]
            writer_connections[(DEBUG_STATE_TOPIC, "std_msgs/msg/Float32MultiArray")] = replay_connections[DEBUG_STATE_TOPIC]
            writer_connections[(DEBUG_BASE_VEL_TOPIC, "std_msgs/msg/Float32MultiArray")] = replay_connections[DEBUG_BASE_VEL_TOPIC]

        replay_topics_set = {CAM_MAIN, CAM_SECONDARY_0, CAM_SECONDARY_1, STATE_LEFT, STATE_RIGHT, ODOM_TOPIC}
        next_step_ns = replay_start_ns
        current_timestamp_ns = None
        group_items: list[tuple[object, bytes]] = []
        raw_message_count = 0

        for conn, timestamp_ns, raw in reader.messages():
            raw_message_count += 1
            if current_timestamp_ns is None:
                current_timestamp_ns = timestamp_ns

            if timestamp_ns != current_timestamp_ns:
                next_step_ns = flush_timestamp_group(
                    group_timestamp_ns=current_timestamp_ns,
                    group_items=group_items,
                    reader=reader,
                    writer=writer,
                    writer_connections=writer_connections,
                    replay_runner=replay_runner,
                    replay_connections=replay_connections,
                    replay_topics_set=replay_topics_set,
                    next_step_ns=next_step_ns,
                    replay_end_ns=replay_end_ns,
                    step_ns=step_ns,
                    ros2_copy_typestore=ros2_copy_typestore,
                )
                current_timestamp_ns = timestamp_ns
                group_items = []

            group_items.append((conn, raw))

        if current_timestamp_ns is not None and group_items:
            next_step_ns = flush_timestamp_group(
                group_timestamp_ns=current_timestamp_ns,
                group_items=group_items,
                reader=reader,
                writer=writer,
                writer_connections=writer_connections,
                replay_runner=replay_runner,
                replay_connections=replay_connections,
                replay_topics_set=replay_topics_set,
                next_step_ns=next_step_ns,
                replay_end_ns=replay_end_ns,
                step_ns=step_ns,
                ros2_copy_typestore=ros2_copy_typestore,
            )

        while next_step_ns <= replay_end_ns:
            result = replay_runner.maybe_predict(next_step_ns)
            if result is not None:
                write_replay_messages(writer, replay_connections, next_step_ns, result, args.write_debug_topics)
            next_step_ns += step_ns

    planned_steps = math.floor((replay_end_ns - replay_start_ns) / step_ns) + 1
    print("=" * 72)
    print("Shadow replay complete")
    print(f"  raw messages copied: {raw_message_count}")
    print(f"  planned replay steps: {planned_steps}")
    print(f"  policy inference calls: {replay_runner.inference_steps}")
    print(f"  published control steps: {replay_runner.published_steps}")
    print(f"  output bag: {output_path}")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline shadow deployment for hanger diffusion policy")
    parser.add_argument("--bag", required=True, help="Path to the source ROS1 .bag file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output ROS2 bag directory. Defaults to <bag_stem>_shadow_ros2 next to the input bag.",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to pretrained_model directory or run directory")
    parser.add_argument("--rate", type=float, default=10.0, help="Replay / control frequency in Hz")
    parser.add_argument("--use-torque", action="store_true", help="Feed real joint effort into observation.effort")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable EMA smoothing")
    parser.add_argument("--start-offset", type=float, default=0.0, help="Skip this many seconds from the replay start")
    parser.add_argument("--max-duration", type=float, default=None, help="Limit replay duration in seconds")
    parser.add_argument("--force", action="store_true", help="Overwrite the output bag directory if it exists")
    parser.add_argument(
        "--write-debug-topics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write extra Float32MultiArray debug topics alongside the deployment-equivalent control topics",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Print one shadow replay log every N published steps")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    convert_bag(args)


if __name__ == "__main__":
    main()
