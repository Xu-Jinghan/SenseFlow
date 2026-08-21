import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_yolo11n_kitti_tracking_sensor as kitti_eval
from models.restoration import TemporalTinyRestorationCNN


REPO_ROOT = kitti_eval.REPO_ROOT
DEFAULT_CASE2_PARAMS_CSV = (
    REPO_ROOT
    / "outputs"
    / "case2_corrected_area_6p2e-5mm2"
    / "case2_native_R_x2_area6p2e-5mm2_prange1to30Wmm2_x1plateauTauRatio_params.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "tiny_restoration_frontend_yolo11n_kitti_case2.pth"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "train_runs"
    / "tiny_restoration_frontend_yolo11n_kitti_case2"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a temporal restoration frontend for KITTI/YOLO sensor detection inputs. "
            "The frontend maps case2 nonideal sensor frames to ideal sensor frames."
        )
    )
    parser.add_argument("--kitti-root", default=kitti_eval._default_kitti_root())
    parser.add_argument("--sequence", default="0007")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=192)
    parser.add_argument("--params-csv", default=str(DEFAULT_CASE2_PARAMS_CSV))

    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--readout", default="tia", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog-readout", default="tia", choices=["tia", "integration"])
    parser.add_argument("--adc-enabled", type=int, default=0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--adc-calibration-low", type=float, default=None)
    parser.add_argument("--adc-calibration-high", type=float, default=None)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--fps-sim", type=float, default=200.0)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--normalization-mode", default="calibration", choices=["physical", "calibration", "per_frame", "none"])
    parser.add_argument("--range-mode", default="minmax", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument("--range-calibration-samples", type=int, default=50)
    parser.add_argument("--range-calibration-max-values-per-frame", type=int, default=20000)
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)

    parser.add_argument("--prange1-density", type=float, default=None)
    parser.add_argument("--prange2-density", type=float, default=None)
    parser.add_argument("--pmin-density", type=float, default=None)
    parser.add_argument("--pmax-density", type=float, default=None)
    parser.add_argument("--device-area-cm2", type=float, default=None)
    parser.add_argument("--force-single-carrier", type=int, default=0)
    parser.add_argument("--single-r", type=float, default=None)
    parser.add_argument("--single-eta", type=float, default=None)
    parser.add_argument("--single-trise", type=float, default=None)
    parser.add_argument("--single-tfall", type=float, default=None)
    parser.add_argument("--trap-saturation-time", type=float, default=None)
    parser.add_argument("--trap-amplitude-pct", type=float, default=None)
    parser.add_argument("--x2-trap-mode", default="power", choices=["binary", "power"])
    parser.add_argument("--x2-trap-output-mode", default="always", choices=["always", "illumination_gated"])
    parser.add_argument("--x2-alpha", type=float, default=None)
    parser.add_argument("--x2-beta", type=float, default=None)
    parser.add_argument("--detection-dark-current-a", type=float, default=0.0)
    parser.add_argument("--noise-1f-density-1hz", type=float, default=None)
    parser.add_argument("--case1-noise-scale-vs-case1", type=float, default=kitti_eval.DEFAULT_CASE1_NOISE_SCALE_VS_CASE1)
    parser.add_argument("--aging-tau-hours", type=float, default=None)
    parser.add_argument("--r-degradation-pct", type=float, default=None)
    parser.add_argument("--spatial-variation-r-pct", type=float, default=0.0)
    parser.add_argument("--tia-gain-ohm", type=float, default=None)
    parser.add_argument("--integration-gain-v-per-c", type=float, default=None)
    parser.add_argument("--shot-noise", type=int, default=0)
    parser.add_argument("--use-noise-fn", type=int, default=1)
    parser.add_argument(
        "--fast-tia-frame-step",
        type=int,
        default=0,
        help=(
            "If 1 and readout resolves to TIA, use the frame-step TIA fast path in "
            "StatefulNonidealVideoSensor. Default keeps the historical per-step path."
        ),
    )
    parser.add_argument(
        "--temporal-noise-mode",
        default="pixel_buffered",
        choices=["pixel_buffered", "pixel_repeated_window", "global_full_sequence", "global_repeated_window"],
        help=(
            "Temporal-noise synthesis mode used to generate restoration pairs. "
            "pixel_repeated_window keeps independent per-pixel reusable traces."
        ),
    )
    parser.add_argument(
        "--temporal-noise-window-frames",
        type=int,
        default=10,
        help="Number of frames in the reusable temporal-noise window for repeated-window modes.",
    )

    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--loss", default="smooth_l1", choices=["l1", "mse", "smooth_l1"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def resolve_torch_device(torch, device_name):
    device_name = str(device_name)
    if device_name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.isdigit():
        return torch.device(f"cuda:{device_name}")
    return torch.device(device_name)


def tensor_from_unit_rgb(unit_frame, torch):
    rgb = kitti_eval.frame_to_rgb_array(unit_frame)
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()


def build_temporal_window(frames, frame_idx, history_frames, torch):
    start_idx = frame_idx - int(history_frames) + 1
    window = []
    for idx in range(start_idx, frame_idx + 1):
        window.append(frames[max(0, idx)])
    return torch.stack(window, dim=0)


class KittiRestorationTensorDataset:
    def __init__(self, nonideal_frames, ideal_frames, indices, history_frames, torch):
        self.nonideal_frames = nonideal_frames
        self.ideal_frames = ideal_frames
        self.indices = list(indices)
        self.history_frames = int(history_frames)
        self.torch = torch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame_idx = self.indices[index]
        nonideal = build_temporal_window(
            self.nonideal_frames,
            frame_idx,
            self.history_frames,
            self.torch,
        )
        ideal = self.ideal_frames[frame_idx]
        return nonideal, ideal


def create_loss(torch, name):
    if name == "l1":
        return torch.nn.L1Loss()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "smooth_l1":
        return torch.nn.SmoothL1Loss(beta=0.05)
    raise ValueError(f"Unsupported loss: {name}")


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sensor_pairs(args, torch):
    requested_noise_config = kitti_eval.resolve_detection_noise_config(args)
    kitti_root = Path(args.kitti_root).expanduser().resolve()
    sequence = kitti_eval.resolve_sequence(kitti_root, args.sequence, args.num_frames, args.start_frame)
    frame_paths = kitti_eval.select_consecutive_frames(
        kitti_root,
        sequence,
        args.start_frame,
        args.num_frames,
    )

    base_pipeline, video_pipeline = kitti_eval.import_sensor_pipeline()
    sensor_args = kitti_eval.make_sensor_args(
        args,
        base_pipeline,
        total_sequence_frames=len(frame_paths) + max(0, int(args.startup_dark_frames)),
    )
    if sensor_args.analog_readout is None:
        sensor_args.analog_readout = video_pipeline.resolve_analog_readout_mode(sensor_args)
    sensor_args.frame_range_mode_override = video_pipeline.effective_frame_range_mode(sensor_args)

    base_params = video_pipeline.resolve_sequence_base_params(sensor_args)
    base_params = kitti_eval.apply_detection_x2_overrides(args, base_params)
    if sensor_args.tia_gain_ohm is None:
        sensor_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if sensor_args.integration_gain_v_per_c is None:
        sensor_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    noise_trace = kitti_eval.summarize_actual_noise_trace(sensor_args, base_params)

    range_cases = ["ideal", "nonideal"]
    case_range_bounds = kitti_eval.compute_dataset_range_bounds(
        args,
        frame_paths,
        sensor_args,
        base_params,
        range_cases,
        base_pipeline,
        video_pipeline,
    )

    sequence_sensor = video_pipeline.StatefulNonidealVideoSensor(args=sensor_args, base_params=base_params)
    zero_power = np.zeros((3, args.output_height, args.output_width), dtype=np.float64)
    for _ in range(max(0, int(args.startup_dark_frames))):
        sequence_sensor.simulate_frame(zero_power)

    ideal_frames = []
    nonideal_frames = []
    started = time.time()
    for idx, image_path in enumerate(frame_paths):
        source = Image.open(image_path).convert("RGB")
        power_maps = kitti_eval.build_power_maps_rect(
            source,
            args.output_width,
            args.output_height,
            3,
            base_params,
        )
        ideal_raw = video_pipeline.simulate_ideal_video_frame(power_maps, sensor_args, base_params)
        ideal_unit = kitti_eval.scale_sensor_frame(
            ideal_raw,
            "ideal",
            sensor_args,
            case_range_bounds,
            base_pipeline,
            video_pipeline,
        )
        nonideal_raw = sequence_sensor.simulate_frame(power_maps)
        nonideal_unit = kitti_eval.scale_sensor_frame(
            nonideal_raw,
            "nonideal",
            sensor_args,
            case_range_bounds,
            base_pipeline,
            video_pipeline,
        )
        ideal_frames.append(tensor_from_unit_rgb(ideal_unit, torch))
        nonideal_frames.append(tensor_from_unit_rgb(nonideal_unit, torch))

        processed = idx + 1
        if processed % 25 == 0 or processed == len(frame_paths):
            print(f"  prepared restoration pairs: {processed}/{len(frame_paths)}", flush=True)

    metadata = {
        "sequence": sequence,
        "frame_paths": [str(path) for path in frame_paths],
        "elapsed_s": time.time() - started,
        "requested_noise_config": requested_noise_config,
        "sensor_args": vars(sensor_args),
        "base_params": {
            key: (None if value is None else float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in base_params.items()
            if key not in {"spatial_variation_cache_dir"}
        },
        "case_range_bounds": case_range_bounds,
        "noise_trace": noise_trace,
    }
    return nonideal_frames, ideal_frames, metadata


def split_indices(num_frames, train_ratio, seed):
    indices = list(range(num_frames))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if num_frames <= 1:
        return indices, indices
    train_count = int(round(float(train_ratio) * num_frames))
    train_count = min(max(train_count, 1), num_frames - 1)
    return sorted(indices[:train_count]), sorted(indices[train_count:])


def evaluate_model(model, loader, loss_fn, torch, device):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    samples = 0
    with torch.no_grad():
        for nonideal, ideal in loader:
            nonideal = nonideal.to(device, non_blocking=True)
            ideal = ideal.to(device, non_blocking=True)
            restored = model(nonideal)
            batch_size = int(ideal.size(0))
            total_loss += float(loss_fn(restored, ideal).item()) * batch_size
            total_l1 += float(torch.mean(torch.abs(restored - ideal)).item()) * batch_size
            total_mse += float(torch.mean(torch.square(restored - ideal)).item()) * batch_size
            samples += batch_size
    denom = max(samples, 1)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "mse": total_mse / denom,
        "samples": samples,
    }


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    seed_everything(torch, int(args.seed))
    device = resolve_torch_device(torch, args.device)
    output_path = Path(args.output_path).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else run_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("KITTI/YOLO case2 learned restoration training", flush=True)
    print("Pipeline: sensor / restoration frontend; YOLO weights are not trained", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Output checkpoint: {output_path}", flush=True)
    print(f"Frames: sequence={args.sequence} start={args.start_frame} count={args.num_frames}", flush=True)

    nonideal_frames, ideal_frames, sensor_metadata = prepare_sensor_pairs(args, torch)
    train_indices, val_indices = split_indices(len(nonideal_frames), args.train_ratio, args.seed)
    train_dataset = KittiRestorationTensorDataset(nonideal_frames, ideal_frames, train_indices, args.history_frames, torch)
    val_dataset = KittiRestorationTensorDataset(nonideal_frames, ideal_frames, val_indices, args.history_frames, torch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = TemporalTinyRestorationCNN(
        in_channels=3,
        hidden_channels=int(args.hidden_channels),
        num_blocks=int(args.num_blocks),
        history_frames=int(args.history_frames),
    ).to(device)
    loss_fn = create_loss(torch, args.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))

    history = []
    best_val_l1 = float("inf")
    best_epoch = 0
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for nonideal, ideal in train_loader:
            nonideal = nonideal.to(device, non_blocking=True)
            ideal = ideal.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            restored = model(nonideal)
            loss = loss_fn(restored, ideal)
            loss.backward()
            optimizer.step()
            batch_size = int(ideal.size(0))
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size

        scheduler.step()
        train_metrics = evaluate_model(model, train_loader, loss_fn, torch, device)
        val_metrics = evaluate_model(model, val_loader, loss_fn, torch, device)
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_batch_loss": epoch_loss / max(epoch_samples, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_path)

        print(
            f"epoch {epoch:03d}/{int(args.epochs):03d} "
            f"train_l1={train_metrics['l1']:.6f} val_l1={val_metrics['l1']:.6f} "
            f"val_mse={val_metrics['mse']:.6f}",
            flush=True,
        )

    if output_path.exists():
        model.load_state_dict(torch.load(output_path, map_location=device, weights_only=False))
    final_train = evaluate_model(model, train_loader, loss_fn, torch, device)
    final_val = evaluate_model(model, val_loader, loss_fn, torch, device)

    summary = {
        "pipeline": "sensor / restoration frontend",
        "task": "object_detection_input_restoration",
        "training_target": "ideal sensor RGB tensor",
        "yolo_training": "disabled",
        "args": vars(args),
        "device": str(device),
        "output_path": str(output_path),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_l1": best_val_l1,
        "final_train": final_train,
        "final_val": final_val,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "sensor": sensor_metadata,
        "history": history,
        "elapsed_s": time.time() - started,
    }
    summary_json.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {output_path}", flush=True)
    print(f"Saved summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
