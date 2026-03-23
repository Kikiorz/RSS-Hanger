from pathlib import Path
import shutil
import numpy as np
import cv2
import time
import os

from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ============================================================
# DATA SOURCE CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data" / "ACT-WHOLE"))
HF_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", PROJECT_ROOT / "data"))
REPO_NAME = os.environ.get("REPO_NAME", "ACT-WHOLE-DP-3CAM-BASE-TORQUE")
TASK_LABEL = os.environ.get(
    "TASK_LABEL",
    "Place the basket on the red marker and pick the yellow pepper into the basket.",
)

# ============================================================
# ROS TOPICS
# ============================================================
CAM_MAIN = "/realsense_top/color/image_raw/compressed"
CAM_SECONDARY_0 = "/realsense_left/color/image_raw/compressed"
CAM_SECONDARY_1 = "/realsense_right/color/image_raw/compressed"

STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ACTION_LEFT = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"

ODOM_TOPIC = "/ranger_base_node/odom"
TELEOP_CMD_VEL = "/teleop/cmd_vel"

# ============================================================
# SETTINGS
# ============================================================
FPS = int(os.environ.get("FPS", 10))
IMG_SIZE = (224, 224)
JOINT_DIM = 7


def decode_compressed_image(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return img_rgb


def nearest_idx(times, t):
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before = times[idx - 1]
    after = times[idx]
    return idx if abs(after - t) < abs(t - before) else idx - 1


def extract_joint_array(msg, field_name, dim=JOINT_DIM):
    values = getattr(msg, field_name, [])
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] >= dim:
        return arr[:dim]
    padded = np.zeros(dim, dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def extract_base_velocity_from_odom(msg):
    return np.array(
        [
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.angular.z),
        ],
        dtype=np.float32,
    )


def extract_base_action_from_cmd_vel(msg):
    return np.array(
        [
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.z),
        ],
        dtype=np.float32,
    )


def collect_bag_files():
    if not DATA_ROOT.exists():
        print(f"Warning: Data directory not found: {DATA_ROOT}")
        return []

    bag_files = sorted(DATA_ROOT.glob("*.bag"))
    if not bag_files:
        print(f"Warning: No .bag files found in {DATA_ROOT}")
    else:
        print(f"Found {len(bag_files)} bag file(s) in {DATA_ROOT}")

    return bag_files


def process_single_bag(args):
    bag_path, task_name, bag_idx, total_bags = args
    print(f"[Bag {bag_idx}/{total_bags}] Processing: {bag_path.name}")

    bag_start_time = time.time()

    try:
        with AnyReader([bag_path]) as reader:
            required_topics = {
                CAM_MAIN,
                CAM_SECONDARY_0,
                CAM_SECONDARY_1,
                STATE_LEFT,
                STATE_RIGHT,
                ACTION_LEFT,
                ACTION_RIGHT,
                ODOM_TOPIC,
            }
            optional_topics = {TELEOP_CMD_VEL}
            interested_topics = required_topics | optional_topics
            topic_to_msgs = {topic: [] for topic in interested_topics}

            connections = [c for c in reader.connections if c.topic in interested_topics]
            if not connections:
                print(f"[Bag {bag_idx}/{total_bags}] Warning: No relevant topics found, skipping.")
                return None

            for conn, t, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)
                topic_to_msgs[conn.topic].append((t, msg))

            cam_main_msgs = topic_to_msgs[CAM_MAIN]
            cam_secondary_0_msgs = topic_to_msgs[CAM_SECONDARY_0]
            cam_secondary_1_msgs = topic_to_msgs[CAM_SECONDARY_1]
            state_left_msgs = topic_to_msgs[STATE_LEFT]
            state_right_msgs = topic_to_msgs[STATE_RIGHT]
            action_left_msgs = topic_to_msgs[ACTION_LEFT]
            action_right_msgs = topic_to_msgs[ACTION_RIGHT]
            odom_msgs = topic_to_msgs[ODOM_TOPIC]
            cmd_vel_msgs = topic_to_msgs[TELEOP_CMD_VEL]

            for topic, msgs in [
                (CAM_MAIN, cam_main_msgs),
                (CAM_SECONDARY_0, cam_secondary_0_msgs),
                (CAM_SECONDARY_1, cam_secondary_1_msgs),
                (STATE_LEFT, state_left_msgs),
                (STATE_RIGHT, state_right_msgs),
                (ACTION_LEFT, action_left_msgs),
                (ACTION_RIGHT, action_right_msgs),
                (ODOM_TOPIC, odom_msgs),
            ]:
                if not msgs:
                    print(f"[Bag {bag_idx}/{total_bags}] Warning: Missing required topic {topic}, skipping")
                    return None

            cam_main_times = np.array([t for t, _ in cam_main_msgs], dtype=np.int64)
            cam_secondary_0_times = np.array([t for t, _ in cam_secondary_0_msgs], dtype=np.int64)
            cam_secondary_1_times = np.array([t for t, _ in cam_secondary_1_msgs], dtype=np.int64)
            state_left_times = np.array([t for t, _ in state_left_msgs], dtype=np.int64)
            state_right_times = np.array([t for t, _ in state_right_msgs], dtype=np.int64)
            action_left_times = np.array([t for t, _ in action_left_msgs], dtype=np.int64)
            action_right_times = np.array([t for t, _ in action_right_msgs], dtype=np.int64)
            odom_times = np.array([t for t, _ in odom_msgs], dtype=np.int64)
            cmd_vel_times = np.array([t for t, _ in cmd_vel_msgs], dtype=np.int64) if cmd_vel_msgs else None

            start_candidates = [
                cam_main_times[0],
                cam_secondary_0_times[0],
                cam_secondary_1_times[0],
                state_left_times[0],
                state_right_times[0],
                action_left_times[0],
                action_right_times[0],
                odom_times[0],
            ]
            end_candidates = [
                cam_main_times[-1],
                cam_secondary_0_times[-1],
                cam_secondary_1_times[-1],
                state_left_times[-1],
                state_right_times[-1],
                action_left_times[-1],
                action_right_times[-1],
                odom_times[-1],
            ]
            if cmd_vel_times is not None and len(cmd_vel_times) > 0:
                start_candidates.append(cmd_vel_times[0])
                end_candidates.append(cmd_vel_times[-1])

            t_start = max(start_candidates)
            t_end = min(end_candidates)

            if t_end <= t_start:
                print(f"[Bag {bag_idx}/{total_bags}] Warning: Non-positive duration, skipping")
                return None

            common_duration = (t_end - t_start) / 1e9
            print(f"[Bag {bag_idx}/{total_bags}] Common time range: {common_duration:.2f} seconds")

            min_dt = int(1e9 / FPS)
            num_frames = int((t_end - t_start) / min_dt)
            if num_frames <= 0:
                print(f"[Bag {bag_idx}/{total_bags}] Warning: num_frames <= 0, skipping")
                return None

            uniform_timestamps = np.linspace(t_start, t_end, num_frames, dtype=np.int64)
            print(f"[Bag {bag_idx}/{total_bags}] Generating {num_frames} frames at {FPS} FPS")

            episode_frames = []

            for t_frame in uniform_timestamps:
                idx_main = nearest_idx(cam_main_times, t_frame)
                idx_s0 = nearest_idx(cam_secondary_0_times, t_frame)
                idx_s1 = nearest_idx(cam_secondary_1_times, t_frame)

                main_image = decode_compressed_image(cam_main_msgs[idx_main][1])
                secondary_0_image = decode_compressed_image(cam_secondary_0_msgs[idx_s0][1])
                secondary_1_image = decode_compressed_image(cam_secondary_1_msgs[idx_s1][1])

                idx_sl = nearest_idx(state_left_times, t_frame)
                idx_sr = nearest_idx(state_right_times, t_frame)
                state_left_msg = state_left_msgs[idx_sl][1]
                state_right_msg = state_right_msgs[idx_sr][1]

                state_left_pos = extract_joint_array(state_left_msg, "position")
                state_right_pos = extract_joint_array(state_right_msg, "position")
                state_left_eff = extract_joint_array(state_left_msg, "effort")
                state_right_eff = extract_joint_array(state_right_msg, "effort")

                state_14d = np.concatenate([state_left_pos, state_right_pos]).astype(np.float32)
                effort_14d = np.concatenate([state_left_eff, state_right_eff]).astype(np.float32)

                idx_odom = nearest_idx(odom_times, t_frame)
                odom_msg = odom_msgs[idx_odom][1]
                base_velocity = extract_base_velocity_from_odom(odom_msg)

                idx_al = nearest_idx(action_left_times, t_frame)
                idx_ar = nearest_idx(action_right_times, t_frame)
                action_left_msg = action_left_msgs[idx_al][1]
                action_right_msg = action_right_msgs[idx_ar][1]
                action_left = extract_joint_array(action_left_msg, "position")
                action_right = extract_joint_array(action_right_msg, "position")

                if cmd_vel_times is not None and len(cmd_vel_times) > 0:
                    idx_cmd = nearest_idx(cmd_vel_times, t_frame)
                    action_base = extract_base_action_from_cmd_vel(cmd_vel_msgs[idx_cmd][1])
                else:
                    action_base = base_velocity.copy()

                action_17d = np.concatenate([action_base, action_left, action_right]).astype(np.float32)

                frame = {
                    "observation.images.main": main_image,
                    "observation.images.secondary_0": secondary_0_image,
                    "observation.images.secondary_1": secondary_1_image,
                    "observation.state": state_14d,
                    "observation.base_velocity": base_velocity,
                    "observation.effort": effort_14d,
                    "action": action_17d,
                    "task": task_name,
                }
                episode_frames.append(frame)

            elapsed = time.time() - bag_start_time
            print(f"[Bag {bag_idx}/{total_bags}] ✓ Completed in {elapsed:.1f}s ({num_frames} frames)")
            return episode_frames

    except Exception as e:
        print(f"[Bag {bag_idx}/{total_bags}] Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    total_start_time = time.time()

    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {
        "action": {
            "dtype": "float32",
            "shape": (17,),
            "names": ["base_vx", "base_vy", "base_omega"]
            + [f"left_joint_{i}" for i in range(7)]
            + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint_{i}" for i in range(7)]
            + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.base_velocity": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["base_vx", "base_vy", "base_omega"],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint_{i}_eff" for i in range(7)]
            + [f"right_joint_{i}_eff" for i in range(7)],
        },
        "observation.images.main": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_0": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_1": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        robot_type="zeno",
        fps=FPS,
        features=features,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=8,
    )

    print(f"\n{'=' * 60}")
    print("Converting ACT-WHOLE to LeRobot format (DP: 3 cameras + 14D state + base_velocity + torque)")
    print(f"Task: {TASK_LABEL}")
    print(f"Data path: {DATA_ROOT}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    bag_files = collect_bag_files()
    total_bags = len(bag_files)

    if total_bags == 0:
        print("No bag files to process.")
    else:
        successful_episodes = 0
        for bag_idx, bag_path in enumerate(bag_files, 1):
            bag_args = (bag_path, TASK_LABEL, bag_idx, total_bags)
            result = process_single_bag(bag_args)

            if result is not None:
                for frame in result:
                    dataset.add_frame(frame)
                dataset.save_episode()
                successful_episodes += 1
                print(f"[Bag {bag_idx}/{total_bags}] Saved to dataset")

            del result

        total_elapsed = time.time() - total_start_time
        print(f"\n{'=' * 60}")
        print("✓ Conversion complete!")
        print(f"  Successfully converted: {successful_episodes}/{total_bags} episodes")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}\n")
