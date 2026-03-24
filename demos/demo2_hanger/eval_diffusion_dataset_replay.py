#!/usr/bin/env python3
"""Offline dataset replay evaluation for hanger diffusion checkpoints.

This script mirrors the inference semantics in `demos/demo2_hanger/run_diffusion.py`
but feeds observations from a local LeRobot dataset instead of ROS topics. It is
useful for a quick offline check of whether a trained checkpoint reproduces the
demonstration actions better than a trivial baseline.

What it does:
1. loads a local LeRobot dataset frame-by-frame,
2. resets the diffusion policy at episode boundaries,
3. runs the same `predict_action(...)` + saved processor path as deployment,
4. compares predicted actions against dataset actions,
5. reports raw-model metrics, deployment-style published metrics, and a simple
   copy-observation baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action

DEFAULT_DATASET_ID = "ACT-WHOLE-DP-3CAM-BASE-TORQUE"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / DEFAULT_DATASET_ID
DEFAULT_RUN_DIR = PROJECT_ROOT / "models" / "diffusion_policy_official_base"
ACTION_DIM = 17

SLICE_MAP = {
    "all": slice(0, 17),
    "base": slice(0, 3),
    "left": slice(3, 10),
    "right": slice(10, 17),
}


@dataclass
class RunningMetrics:
    abs_sum: np.ndarray
    sq_sum: np.ndarray
    pred_abs_sum: np.ndarray
    target_abs_sum: np.ndarray
    count: int = 0

    @classmethod
    def create(cls, dim: int) -> "RunningMetrics":
        zeros = np.zeros(dim, dtype=np.float64)
        return cls(abs_sum=zeros.copy(), sq_sum=zeros.copy(), pred_abs_sum=zeros.copy(), target_abs_sum=zeros.copy(), count=0)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        diff = pred - target
        self.abs_sum += np.abs(diff)
        self.sq_sum += diff * diff
        self.pred_abs_sum += np.abs(pred)
        self.target_abs_sum += np.abs(target)
        self.count += 1

    def block_summary(self) -> dict[str, dict[str, float]]:
        if self.count == 0:
            raise RuntimeError("No samples were accumulated.")

        result: dict[str, dict[str, float]] = {}
        for name, slc in SLICE_MAP.items():
            abs_mean = self.abs_sum[slc] / self.count
            sq_mean = self.sq_sum[slc] / self.count
            pred_abs_mean = self.pred_abs_sum[slc] / self.count
            target_abs_mean = self.target_abs_sum[slc] / self.count
            result[name] = {
                "mae": float(abs_mean.mean()),
                "rmse": float(np.sqrt(sq_mean.mean())),
                "pred_abs_mean": float(pred_abs_mean.mean()),
                "target_abs_mean": float(target_abs_mean.mean()),
            }
        return result


@dataclass
class DeltaMetrics:
    abs_sum: np.ndarray
    pred_abs_sum: np.ndarray
    target_abs_sum: np.ndarray
    count: int = 0

    @classmethod
    def create(cls, dim: int) -> "DeltaMetrics":
        zeros = np.zeros(dim, dtype=np.float64)
        return cls(abs_sum=zeros.copy(), pred_abs_sum=zeros.copy(), target_abs_sum=zeros.copy(), count=0)

    def update(self, pred_delta: np.ndarray, target_delta: np.ndarray) -> None:
        pred_delta = np.asarray(pred_delta, dtype=np.float64)
        target_delta = np.asarray(target_delta, dtype=np.float64)
        self.abs_sum += np.abs(pred_delta - target_delta)
        self.pred_abs_sum += np.abs(pred_delta)
        self.target_abs_sum += np.abs(target_delta)
        self.count += 1

    def block_summary(self) -> dict[str, dict[str, float]]:
        if self.count == 0:
            return {
                name: {"delta_mae": 0.0, "pred_delta_abs_mean": 0.0, "target_delta_abs_mean": 0.0}
                for name in SLICE_MAP
            }
        result: dict[str, dict[str, float]] = {}
        for name, slc in SLICE_MAP.items():
            result[name] = {
                "delta_mae": float((self.abs_sum[slc] / self.count).mean()),
                "pred_delta_abs_mean": float((self.pred_abs_sum[slc] / self.count).mean()),
                "target_delta_abs_mean": float((self.target_abs_sum[slc] / self.count).mean()),
            }
        return result


@dataclass
class SmoothingState:
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    base: np.ndarray | None = None

    def reset(self) -> None:
        self.left = None
        self.right = None
        self.base = None


def find_latest_pretrained_dir(search_root: Path) -> Path | None:
    candidates = sorted(
        search_root.glob("**/checkpoints/*/pretrained_model/config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parent


def resolve_pretrained_paths(ckpt_args: Iterable[str] | None) -> list[Path]:
    if not ckpt_args:
        latest = find_latest_pretrained_dir(DEFAULT_RUN_DIR)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {DEFAULT_RUN_DIR}")
        return [latest]

    resolved: list[Path] = []
    for ckpt_arg in ckpt_args:
        candidate = Path(ckpt_arg).expanduser().resolve()
        if candidate.is_file() and candidate.name == "config.json":
            resolved.append(candidate.parent)
            continue
        if candidate.is_dir() and (candidate / "config.json").exists():
            resolved.append(candidate)
            continue
        if candidate.is_dir() and (candidate / "pretrained_model" / "config.json").exists():
            resolved.append(candidate / "pretrained_model")
            continue
        if candidate.is_dir():
            nested = find_latest_pretrained_dir(candidate)
            if nested is not None:
                resolved.append(nested)
                continue
        raise FileNotFoundError(f"Invalid checkpoint path: {candidate}")
    return resolved


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def image_tensor_to_uint8_hwc(value) -> np.ndarray:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor * 255.0).round().to(torch.uint8).numpy()
    return np.ascontiguousarray(array)


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


def build_observation(sample: dict, policy: DiffusionPolicy, use_torque_runtime: bool) -> dict[str, np.ndarray]:
    obs = {
        "observation.images.main": image_tensor_to_uint8_hwc(sample["observation.images.main"]),
        "observation.images.secondary_0": image_tensor_to_uint8_hwc(sample["observation.images.secondary_0"]),
        "observation.images.secondary_1": image_tensor_to_uint8_hwc(sample["observation.images.secondary_1"]),
        "observation.state": tensor_to_numpy(sample["observation.state"]).astype(np.float32),
    }

    if "observation.images.secondary_2" in policy.config.input_features:
        obs["observation.images.secondary_2"] = obs["observation.images.main"]

    if policy.config.use_base:
        obs["observation.base_velocity"] = tensor_to_numpy(sample["observation.base_velocity"]).astype(np.float32)

    if policy.config.use_torque:
        if use_torque_runtime:
            effort = tensor_to_numpy(sample["observation.effort"]).astype(np.float32)
        else:
            effort = np.zeros(14, dtype=np.float32)
        obs["observation.effort"] = effort

    return obs


def make_copy_observation_baseline(sample: dict, use_base: bool) -> np.ndarray:
    state = tensor_to_numpy(sample["observation.state"]).astype(np.float32)
    if use_base:
        base = tensor_to_numpy(sample["observation.base_velocity"]).astype(np.float32)
    else:
        base = np.zeros(3, dtype=np.float32)
    return np.concatenate([base, state], axis=0).astype(np.float32)


def apply_deployment_postprocess(
    raw_action: np.ndarray,
    use_base: bool,
    smoothing_enabled: bool,
    smoothing_alpha: float,
    smoothing_state: SmoothingState,
) -> np.ndarray:
    action_base = raw_action[0:3].copy()
    action_left = raw_action[3:10].copy()
    action_right = raw_action[10:17].copy()

    if smoothing_enabled:
        if smoothing_state.left is None:
            smoothing_state.left = action_left
            smoothing_state.right = action_right
            smoothing_state.base = action_base
        else:
            smoothing_state.left = smoothing_alpha * action_left + (1.0 - smoothing_alpha) * smoothing_state.left
            smoothing_state.right = smoothing_alpha * action_right + (1.0 - smoothing_alpha) * smoothing_state.right
            smoothing_state.base = smoothing_alpha * action_base + (1.0 - smoothing_alpha) * smoothing_state.base
        action_left = smoothing_state.left
        action_right = smoothing_state.right
        action_base = smoothing_state.base

    published = np.zeros(ACTION_DIM, dtype=np.float32)
    if use_base:
        published[0] = float(action_base[0])
        published[1] = 0.0
        published[2] = 0.0
    published[3:10] = action_left
    published[10:17] = action_right
    return published


def improvement_fraction(baseline: float, model: float) -> float | None:
    if baseline == 0.0:
        return None
    return (baseline - model) / baseline


def evaluate_checkpoint(ckpt_dir: Path, dataset: LeRobotDataset, args: argparse.Namespace) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)
    policy, preprocessor, postprocessor = load_policy(ckpt_dir, device)

    if args.use_torque and not policy.config.use_torque:
        raise ValueError("--use-torque requires a torque-enabled checkpoint")

    raw_metrics = RunningMetrics.create(ACTION_DIM)
    published_metrics = RunningMetrics.create(ACTION_DIM)
    baseline_metrics = RunningMetrics.create(ACTION_DIM)
    raw_delta_metrics = DeltaMetrics.create(ACTION_DIM)
    published_delta_metrics = DeltaMetrics.create(ACTION_DIM)
    baseline_delta_metrics = DeltaMetrics.create(ACTION_DIM)

    smoothing_state = SmoothingState()
    inference_ms: list[float] = []
    episodes_seen: list[int] = []
    previous_episode: int | None = None
    sample_task: str | None = None
    num_samples = 0
    prev_target_action: np.ndarray | None = None
    prev_raw_action: np.ndarray | None = None
    prev_published_action: np.ndarray | None = None
    prev_baseline_action: np.ndarray | None = None

    for idx in range(len(dataset)):
        sample = dataset[idx]
        episode_index = int(tensor_to_numpy(sample["episode_index"]).item())
        if previous_episode != episode_index:
            policy.reset()
            smoothing_state.reset()
            previous_episode = episode_index
            episodes_seen.append(episode_index)
            prev_target_action = None
            prev_raw_action = None
            prev_published_action = None
            prev_baseline_action = None

        if sample_task is None:
            sample_task = sample.get("task")

        target_action = tensor_to_numpy(sample["action"]).astype(np.float32)
        observation = build_observation(sample, policy, args.use_torque)

        infer_start = time.perf_counter()
        action_tensor = predict_action(
            observation=observation,
            policy=policy,
            device=device_t,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=bool(policy.config.use_amp),
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        inference_ms.append(infer_ms)

        raw_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        if raw_action.shape[0] != ACTION_DIM:
            raise RuntimeError(f"Expected action dim {ACTION_DIM}, got {raw_action.shape[0]}")

        published_action = apply_deployment_postprocess(
            raw_action=raw_action,
            use_base=bool(policy.config.use_base),
            smoothing_enabled=not args.no_smoothing,
            smoothing_alpha=args.smoothing,
            smoothing_state=smoothing_state,
        )
        baseline_action = make_copy_observation_baseline(sample, use_base=bool(policy.config.use_base))

        raw_metrics.update(raw_action, target_action)
        published_metrics.update(published_action, target_action)
        baseline_metrics.update(baseline_action, target_action)

        if prev_target_action is not None:
            target_delta = target_action - prev_target_action
            raw_delta_metrics.update(raw_action - prev_raw_action, target_delta)
            published_delta_metrics.update(published_action - prev_published_action, target_delta)
            baseline_delta_metrics.update(baseline_action - prev_baseline_action, target_delta)

        prev_target_action = target_action
        prev_raw_action = raw_action
        prev_published_action = published_action
        prev_baseline_action = baseline_action

        num_samples += 1
        if num_samples <= 3 or (args.log_every > 0 and num_samples % args.log_every == 0):
            print(
                f"[{ckpt_dir.parent.name}] sample={num_samples} episode={episode_index} "
                f"infer_ms={infer_ms:.1f} raw_mae={np.abs(raw_action - target_action).mean():.4f} "
                f"pub_mae={np.abs(published_action - target_action).mean():.4f}"
            )

        if args.max_frames is not None and num_samples >= args.max_frames:
            break

    raw_summary = raw_metrics.block_summary()
    published_summary = published_metrics.block_summary()
    baseline_summary = baseline_metrics.block_summary()
    raw_delta_summary = raw_delta_metrics.block_summary()
    published_delta_summary = published_delta_metrics.block_summary()
    baseline_delta_summary = baseline_delta_metrics.block_summary()

    comparison = {}
    for block in SLICE_MAP:
        comparison[block] = {
            "raw_vs_baseline_improvement": improvement_fraction(baseline_summary[block]["mae"], raw_summary[block]["mae"]),
            "published_vs_baseline_improvement": improvement_fraction(
                baseline_summary[block]["mae"], published_summary[block]["mae"]
            ),
        }

    inference_np = np.asarray(inference_ms, dtype=np.float64)
    return {
        "checkpoint": str(ckpt_dir),
        "checkpoint_step": ckpt_dir.parent.name,
        "device": device,
        "num_samples": num_samples,
        "episodes_seen": episodes_seen,
        "dataset_task_example": sample_task,
        "policy": {
            "use_base": bool(policy.config.use_base),
            "use_torque": bool(policy.config.use_torque),
            "n_obs_steps": int(policy.config.n_obs_steps),
            "n_action_steps": int(policy.config.n_action_steps),
            "horizon": int(policy.config.horizon),
            "use_amp": bool(policy.config.use_amp),
        },
        "inference_ms": {
            "mean": float(inference_np.mean()),
            "p50": float(np.percentile(inference_np, 50)),
            "p95": float(np.percentile(inference_np, 95)),
        },
        "raw": raw_summary,
        "published": published_summary,
        "baseline_copy_observation": baseline_summary,
        "raw_delta": raw_delta_summary,
        "published_delta": published_delta_summary,
        "baseline_delta": baseline_delta_summary,
        "improvement": comparison,
    }


def print_human_summary(summary: dict) -> None:
    print("=" * 80)
    print(f"Checkpoint: {summary['checkpoint']}")
    print(
        f"  samples={summary['num_samples']} episodes={summary['episodes_seen']} "
        f"infer_ms(mean/p50/p95)={summary['inference_ms']['mean']:.1f}/"
        f"{summary['inference_ms']['p50']:.1f}/{summary['inference_ms']['p95']:.1f}"
    )
    print(
        f"  policy: use_base={summary['policy']['use_base']} use_torque={summary['policy']['use_torque']} "
        f"n_obs={summary['policy']['n_obs_steps']} n_action={summary['policy']['n_action_steps']} horizon={summary['policy']['horizon']}"
    )
    if summary.get("dataset_task_example"):
        print(f"  dataset task example: {summary['dataset_task_example']}")
        print("  note: evaluation ignores task text to mirror current deployment semantics")

    for variant_key, label in [
        ("raw", "Raw model action"),
        ("published", "Deployment-style published action"),
        ("baseline_copy_observation", "Copy-observation baseline"),
    ]:
        block = summary[variant_key]
        delta_key = {
            "raw": "raw_delta",
            "published": "published_delta",
            "baseline_copy_observation": "baseline_delta",
        }[variant_key]
        delta_block = summary[delta_key]
        print(f"  {label}:")
        print(
            f"    all  mae={block['all']['mae']:.4f} rmse={block['all']['rmse']:.4f} "
            f"| |pred|={block['all']['pred_abs_mean']:.4f} |target|={block['all']['target_abs_mean']:.4f}"
        )
        print(
            f"    base mae={block['base']['mae']:.4f} left mae={block['left']['mae']:.4f} right mae={block['right']['mae']:.4f}"
        )
        print(
            f"    delta all={delta_block['all']['delta_mae']:.4f} "
            f"predΔ={delta_block['all']['pred_delta_abs_mean']:.4f} targetΔ={delta_block['all']['target_delta_abs_mean']:.4f}"
        )

    for block_name in ["all", "base", "left", "right"]:
        raw_imp = summary["improvement"][block_name]["raw_vs_baseline_improvement"]
        pub_imp = summary["improvement"][block_name]["published_vs_baseline_improvement"]
        raw_text = "n/a" if raw_imp is None else f"{raw_imp * 100.0:+.1f}%"
        pub_text = "n/a" if pub_imp is None else f"{pub_imp * 100.0:+.1f}%"
        print(f"  improvement vs baseline [{block_name}]: raw={raw_text}, published={pub_text}")
    print("=" * 80)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline dataset replay evaluation for hanger diffusion policy")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="LeRobot dataset repo id")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Local root of the dataset directory itself (contains data/, meta/, videos/)",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="Checkpoint path(s): pretrained_model dir, checkpoint dir, or run dir. Repeat this flag to compare multiple checkpoints.",
    )
    parser.add_argument("--episodes", nargs="*", type=int, default=None, help="Optional list of episode indices to evaluate")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum number of frames to evaluate per checkpoint")
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu or cuda)")
    parser.add_argument("--use-torque", action="store_true", help="Feed real observation.effort when checkpoint expects torque")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha used by deployment")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable deployment smoothing for published-action metrics")
    parser.add_argument("--log-every", type=int, default=100, help="Print one progress line every N evaluated samples")
    parser.add_argument("--save-json", default=None, help="Optional path to save the summary JSON")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ckpt_dirs = resolve_pretrained_paths(args.ckpt)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset = LeRobotDataset(
        repo_id=args.dataset_id,
        root=dataset_root,
        episodes=args.episodes,
        video_backend="pyav",
    )

    print("=" * 80)
    print("Offline diffusion dataset replay evaluation")
    print(f"  dataset: {args.dataset_id}")
    print(f"  root: {dataset_root}")
    print(f"  total frames loaded: {len(dataset)}")
    print(f"  total episodes loaded: {dataset.num_episodes}")
    print(f"  checkpoints: {[str(path) for path in ckpt_dirs]}")
    print("=" * 80)

    summaries = [evaluate_checkpoint(ckpt_dir, dataset, args) for ckpt_dir in ckpt_dirs]
    for summary in summaries:
        print_human_summary(summary)

    if args.save_json:
        output_path = Path(args.save_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
        print(f"Saved summary JSON to {output_path}")


if __name__ == "__main__":
    main()
