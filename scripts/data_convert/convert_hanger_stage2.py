"""Bag to LeRobot V3.0 Dataset Converter - WholeBody_Hanger Stage2
==================================================================
Parallel version: N_WORKERS bags decoded simultaneously, main process
writes to dataset sequentially.

Task: 抓取衣架移动到衣柜并放置 (pick hanger, move to wardrobe, place)

Usage:
    conda activate base
    python convert_hanger_stage2.py
"""

from pathlib import Path
import shutil
import numpy as np
import cv2
import time
import multiprocessing as mp

from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT   = Path("/home/zeno-rp/2026CoRL/Data/WholeBody_Hanger/stage2")
OUTPUT_DIR  = Path("/home/zeno-rp/2026CoRL/Data/WholeBody_Hanger")
OUTPUT_NAME = "wholebody_hanger_stage2_v30_10HZ"
TASK_LABEL  = "pick up the hanger and move it to the wardrobe to hang it"

N_WORKERS   = 2   # parallel bag decoders (limited by RAM)

# ROS topics
CAM_TOP      = "/realsense_top/color/image_raw/compressed"
CAM_LEFT     = "/realsense_left/color/image_raw/compressed"
CAM_RIGHT    = "/realsense_right/color/image_raw/compressed"
STATE_LEFT   = "/robot/arm_left/joint_states_single"
STATE_RIGHT  = "/robot/arm_right/joint_states_single"
ACTION_LEFT  = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"
ODOM         = "/ranger_base_node/odom"

ALL_TOPICS = {
    CAM_TOP, CAM_LEFT, CAM_RIGHT,
    STATE_LEFT, STATE_RIGHT, ACTION_LEFT, ACTION_RIGHT, ODOM,
}

FPS      = 10
IMG_SIZE = (224, 224)
ARM_DOF  = 7


def decode_compressed_image(msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return img_rgb


def nearest_idx(times, t):
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if abs(times[idx] - t) < abs(t - times[idx - 1]) else idx - 1


def extract_joint_positions(msg, dof=ARM_DOF):
    positions = list(msg.position)
    if len(positions) >= dof:
        return positions[:dof]
    return positions + [0.0] * (dof - len(positions))


def process_single_bag(args):
    """Worker function: decode one bag, return list of frame dicts (images as np arrays)."""
    bag_path, bag_idx, total_bags = args
    print(f"[Bag {bag_idx}/{total_bags}] Start: {bag_path.name}", flush=True)
    t0 = time.time()

    try:
        with AnyReader([bag_path]) as reader:
            topic_to_msgs = {topic: [] for topic in ALL_TOPICS}
            connections = [c for c in reader.connections if c.topic in ALL_TOPICS]
            if not connections:
                print(f"[Bag {bag_idx}] No relevant topics, skipping.", flush=True)
                return None

            for conn, t, raw in reader.messages(connections=connections):
                if conn.topic in topic_to_msgs:
                    msg = reader.deserialize(raw, conn.msgtype)
                    topic_to_msgs[conn.topic].append((t, msg))

            cam_top   = topic_to_msgs[CAM_TOP]
            cam_left  = topic_to_msgs[CAM_LEFT]
            cam_right = topic_to_msgs[CAM_RIGHT]
            st_l      = topic_to_msgs[STATE_LEFT]
            st_r      = topic_to_msgs[STATE_RIGHT]
            ac_l      = topic_to_msgs[ACTION_LEFT]
            ac_r      = topic_to_msgs[ACTION_RIGHT]
            odom_msgs = topic_to_msgs[ODOM]

            for name, msgs in [("cam_top", cam_top), ("cam_left", cam_left),
                                ("cam_right", cam_right), ("st_l", st_l),
                                ("st_r", st_r), ("ac_l", ac_l), ("ac_r", ac_r)]:
                if not msgs:
                    print(f"[Bag {bag_idx}] Missing {name}, skipping.", flush=True)
                    return None

            cam_top_t   = np.array([t for t, _ in cam_top],   dtype=np.int64)
            cam_left_t  = np.array([t for t, _ in cam_left],  dtype=np.int64)
            cam_right_t = np.array([t for t, _ in cam_right], dtype=np.int64)
            st_l_t      = np.array([t for t, _ in st_l],      dtype=np.int64)
            st_r_t      = np.array([t for t, _ in st_r],      dtype=np.int64)
            ac_l_t      = np.array([t for t, _ in ac_l],      dtype=np.int64)
            ac_r_t      = np.array([t for t, _ in ac_r],      dtype=np.int64)
            odom_t = (
                np.array([t for t, _ in odom_msgs], dtype=np.int64)
                if odom_msgs else np.array([], dtype=np.int64)
            )

            t_start = max(cam_top_t[0], cam_left_t[0], cam_right_t[0],
                          st_l_t[0], st_r_t[0], ac_l_t[0], ac_r_t[0])
            t_end   = min(cam_top_t[-1], cam_left_t[-1], cam_right_t[-1],
                          st_l_t[-1], st_r_t[-1], ac_l_t[-1], ac_r_t[-1])

            duration_s = (t_end - t_start) / 1e9
            n_frames   = int(duration_s * FPS)
            if n_frames < 2:
                print(f"[Bag {bag_idx}] Too short ({duration_s:.1f}s), skipping.", flush=True)
                return None

            sample_times = np.linspace(t_start, t_end, n_frames, dtype=np.int64)

            frames = []
            for t in sample_times:
                img_top   = decode_compressed_image(cam_top[nearest_idx(cam_top_t, t)][1])
                img_left  = decode_compressed_image(cam_left[nearest_idx(cam_left_t, t)][1])
                img_right = decode_compressed_image(cam_right[nearest_idx(cam_right_t, t)][1])

                if any(x is None for x in [img_top, img_left, img_right]):
                    continue

                sl = extract_joint_positions(st_l[nearest_idx(st_l_t, t)][1])
                sr = extract_joint_positions(st_r[nearest_idx(st_r_t, t)][1])
                al = extract_joint_positions(ac_l[nearest_idx(ac_l_t, t)][1])
                ar = extract_joint_positions(ac_r[nearest_idx(ac_r_t, t)][1])

                base_vel = np.zeros(3, dtype=np.float32)
                if len(odom_t) > 0:
                    om = odom_msgs[nearest_idx(odom_t, t)][1]
                    tw = om.twist.twist
                    base_vel = np.array([tw.linear.x, tw.linear.y, tw.angular.z], dtype=np.float32)

                state  = np.concatenate([base_vel, np.array(sl + sr, dtype=np.float32)])
                action = np.concatenate([base_vel, np.array(al + ar, dtype=np.float32)])

                frames.append({
                    "observation.images.realsense_top":   img_top,
                    "observation.images.realsense_left":  img_left,
                    "observation.images.realsense_right": img_right,
                    "observation.state": state,
                    "action": action,
                    "task": TASK_LABEL,
                })

            elapsed = time.time() - t0
            print(f"[Bag {bag_idx}] Done: {len(frames)} frames in {elapsed:.1f}s", flush=True)
            return frames if frames else None

    except Exception as e:
        import traceback
        print(f"[Bag {bag_idx}] Error: {e}", flush=True)
        traceback.print_exc()
        return None


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()

    output_path = OUTPUT_DIR / OUTPUT_NAME
    if output_path.exists():
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2 + 3,),
            "names": ["base_vx", "base_vy", "base_omega"]
                     + [f"left_joint_{i}" for i in range(7)]
                     + [f"right_joint_{i}" for i in range(7)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ARM_DOF * 2 + 3,),
            "names": ["base_vx", "base_vy", "base_omega"]
                     + [f"left_joint_{i}" for i in range(7)]
                     + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.images.realsense_top": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.realsense_left": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.realsense_right": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=OUTPUT_NAME,
        root=output_path,
        robot_type="zeno",
        fps=FPS,
        features=features,
        use_videos=True,
        vcodec="h264_nvenc",
        image_writer_threads=8,
        image_writer_processes=4,
    )

    bag_files = sorted(DATA_ROOT.glob("*.bag"))
    total_bags = len(bag_files)
    print(f"\n{'=' * 60}")
    print(f"Converting WholeBody_Hanger Stage2 → LeRobot V3.0")
    print(f"  Task:    {TASK_LABEL}")
    print(f"  Bags:    {total_bags}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Output:  {output_path}")
    print(f"  Codec:   h264_nvenc (GPU)")
    print(f"{'=' * 60}\n")

    args_list = [(p, i + 1, total_bags) for i, p in enumerate(bag_files)]

    successful = 0
    with mp.Pool(processes=N_WORKERS) as pool:
        for bag_idx, result in enumerate(
            pool.imap_unordered(process_single_bag, args_list), start=1
        ):
            if result is not None:
                for frame in result:
                    dataset.add_frame(frame)
                dataset.save_episode()
                successful += 1
                elapsed = time.time() - total_start
                rate = successful / elapsed * 60
                eta = (total_bags - successful) / (successful / elapsed) if successful else 0
                print(f"  [Episode {successful}/{total_bags}] saved | "
                      f"{rate:.1f} ep/min | ETA {eta/60:.1f}min", flush=True)
            del result

    dataset.finalize()

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Episodes: {successful}/{total_bags}")
    print(f"  Time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Output:   {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
