import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torchvision import transforms

import generate_sensor_verification_images as base_pipeline
import generate_sensor_verification_images_video_sequence as video_sequence_pipeline
from models import resnet


DATASET_CONFIG = {
    "cifar10": {
        "num_classes": 10,
    },
    "cifar100": {
        "num_classes": 100,
    },
}
EVAL_CASES = ("raw", "ideal", "nonideal")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained ResNet18 on CIFAR images passed through the "
            "stateful nonideal video-sequence sensor pipeline to recover accuracy."
        )
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to fine-tune on",
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT.parent / ".datasets"),
        help="Dataset root. Data will be read from <data_path>/<dataset>-data",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        help="Pretrained ResNet18 checkpoint to start from. Defaults to models/resnet18_<dataset>.pth",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save the best fine-tuned model state_dict",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional path to save the training summary JSON",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--train_case",
        default="nonideal",
        choices=EVAL_CASES,
        help="Input domain used for fine-tuning",
    )
    parser.add_argument(
        "--selection_case",
        default="nonideal",
        choices=EVAL_CASES,
        help="Validation case used to select the best checkpoint",
    )
    parser.add_argument(
        "--eval_cases",
        nargs="+",
        default=list(EVAL_CASES),
        choices=EVAL_CASES,
        help="Validation cases to report after each epoch",
    )
    parser.add_argument(
        "--train_augment",
        type=int,
        default=1,
        help="Apply CIFAR-style RandomCrop/RandomHorizontalFlip before sensor simulation during fine-tuning",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor_rng_seed", type=int, default=42)
    parser.add_argument(
        "--epoch_seed_stride",
        type=int,
        default=0,
        help="Add epoch_seed_stride * (epoch - 1) to sensor_rng_seed for training epochs",
    )
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from run_dir/last_checkpoint.pth if it exists")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If > 0, save an extra checkpoint every N epochs",
    )

    parser.add_argument("--target_size", type=int, default=32)
    parser.add_argument("--output_channels", type=int, default=3)
    parser.add_argument(
        "--post_norm",
        default="auto",
        choices=["none", "auto", "cifar10", "cifar100", "imagenet"],
    )
    parser.add_argument("--array_size", type=int, default=32)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--power_max", type=float, default=0.1)
    parser.add_argument(
        "--params_csv",
        default=str(base_pipeline.DEFAULT_PARAMS_CSV),
    )
    parser.add_argument("--video_fps", type=float, default=50.0)
    parser.add_argument("--fps_sim", type=float, default=1000.0)
    parser.add_argument("--adc_bits", type=int, default=8)
    parser.add_argument("--adc_full_scale", type=float, default=None)
    parser.add_argument("--range_mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument(
        "--range_scope",
        default="calibration",
        choices=["per_frame", "calibration"],
        help="Whether range scaling is computed per frame or fixed from a calibration set.",
    )
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument("--range_calibration_split", default="train", choices=["train", "test"])
    parser.add_argument("--range_calibration_samples", type=int, default=1024)
    parser.add_argument("--range_calibration_skip_frames", type=int, default=0)
    parser.add_argument("--use_noise_fn", type=int, default=1)
    parser.add_argument("--shot_noise", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=5000.0)
    parser.add_argument("--i_thermal", type=float, default=0.0)
    parser.add_argument("--startup_dark_frames", type=int, default=0)
    parser.add_argument(
        "--temporal_noise_mode",
        default="pixel_buffered",
        choices=[
            "pixel_buffered",
            "pixel_repeated_window",
            "global_full_sequence",
            "global_repeated_window",
        ],
    )
    parser.add_argument("--temporal_noise_window_frames", type=int, default=10)
    parser.add_argument("--fast_tia_frame_step", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_paths(args):
    dataset_name = args.dataset.lower()
    pretrained_path = (
        Path(args.pretrained_path)
        if args.pretrained_path
        else PROJECT_ROOT / "models" / f"resnet18_{dataset_name}.pth"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "models" / f"resnet18_{dataset_name}_video_sequence_{args.train_case}_finetuned.pth"
    )
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else PROJECT_ROOT / "artifacts" / "train_runs" / f"resnet18_{dataset_name}_video_sequence_{args.train_case}_finetune"
    )
    summary_json = Path(args.summary_json) if args.summary_json else run_dir / "summary.json"
    return pretrained_path, output_path, run_dir, summary_json


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def append_history_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict payload")
    return checkpoint


def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict)
    return checkpoint


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return preds.eq(labels).sum().item(), labels.size(0)


def resolve_sample_limit(dataset_length, max_batches, batch_size):
    if max_batches > 0:
        return min(dataset_length, max_batches * batch_size)
    return dataset_length


def build_sensor_args(args, sensor_rng_seed):
    return SimpleNamespace(
        source_dataset=args.dataset,
        seed=args.seed,
        sensor_rng_seed=sensor_rng_seed,
        spatial_variation_cache_dir=getattr(args, "spatial_variation_cache_dir", None),
        target_size=args.target_size,
        output_channels=args.output_channels,
        post_norm=args.post_norm,
        array_size=args.array_size,
        readout=args.readout,
        analog_readout=getattr(args, "analog_readout", None),
        adc_enabled=getattr(args, "adc_enabled", 0),
        power_max=args.power_max,
        params_csv=args.params_csv,
        normalization_mode=getattr(args, "normalization_mode", "calibration"),
        prange1_density=getattr(args, "prange1_density", None),
        prange2_density=getattr(args, "prange2_density", None),
        pmin_density=getattr(args, "pmin_density", None),
        pmax_density=getattr(args, "pmax_density", None),
        device_area_cm2=getattr(args, "device_area_cm2", None),
        force_single_carrier=getattr(args, "force_single_carrier", 0),
        single_r=getattr(args, "single_r", None),
        single_eta=getattr(args, "single_eta", None),
        single_trise=getattr(args, "single_trise", None),
        single_tfall=getattr(args, "single_tfall", None),
        trap_saturation_time=getattr(args, "trap_saturation_time", None),
        trap_amplitude_pct=getattr(args, "trap_amplitude_pct", None),
        noise_1f_density_1hz=getattr(args, "noise_1f_density_1hz", None),
        aging_tau_hours=getattr(args, "aging_tau_hours", None),
        r_degradation_pct=getattr(args, "r_degradation_pct", None),
        spatial_variation_r_pct=getattr(args, "spatial_variation_r_pct", None),
        tia_gain_ohm=getattr(args, "tia_gain_ohm", None),
        integration_gain_v_per_c=getattr(args, "integration_gain_v_per_c", None),
        video_fps=args.video_fps,
        fps_sim=args.fps_sim,
        adc_bits=args.adc_bits,
        adc_full_scale=args.adc_full_scale,
        range_mode=args.range_mode,
        range_scope=args.range_scope,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        range_calibration_split=args.range_calibration_split,
        range_calibration_samples=args.range_calibration_samples,
        range_calibration_skip_frames=getattr(args, "range_calibration_skip_frames", 0),
        i_thermal=args.i_thermal,
        bandwidth=args.bandwidth,
        shot_noise=args.shot_noise,
        use_noise_fn=args.use_noise_fn,
        startup_dark_frames=args.startup_dark_frames,
        temporal_noise_mode=getattr(args, "temporal_noise_mode", "pixel_buffered"),
        temporal_noise_window_frames=getattr(args, "temporal_noise_window_frames", 10),
        fast_tia_frame_step=getattr(args, "fast_tia_frame_step", 0),
    )


def build_train_augmentation(enabled):
    if not bool(enabled):
        return None
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )


def infer_output_channels(base_dataset, sensor_args):
    if len(base_dataset) == 0:
        raise ValueError("Dataset is empty")
    first_image, _ = base_dataset[0]
    return base_pipeline._resolve_output_channels(first_image, sensor_args.output_channels)


def warmup_sequence_sensor(sequence_sensor, output_channels, sensor_args):
    for _ in range(max(0, sensor_args.startup_dark_frames)):
        zero_power = torch.zeros((output_channels, sensor_args.array_size, sensor_args.array_size), dtype=torch.float64)
        if output_channels == 1:
            zero_power = zero_power[0]
        sequence_sensor.simulate_frame(zero_power.numpy())


def select_calibration_dataset(args, train_dataset, val_dataset):
    if args.range_calibration_split == "train":
        return train_dataset
    return val_dataset


def build_case_tensors(
    image,
    sensor_args,
    base_params,
    output_channels,
    requested_cases,
    sequence_sensor=None,
    case_range_bounds=None,
):
    requested_cases = set(requested_cases)
    tensors = {}

    if "raw" in requested_cases:
        tensors["raw"] = base_pipeline._raw_image_to_model_tensor(image, sensor_args)

    need_ideal = "ideal" in requested_cases
    need_nonideal = "nonideal" in requested_cases
    if not need_ideal and not need_nonideal:
        return tensors

    if hasattr(video_sequence_pipeline, "build_sequence_power_maps"):
        power_maps = video_sequence_pipeline.build_sequence_power_maps(image, sensor_args, base_params)
    else:
        power_maps = base_pipeline.build_power_maps(
            image,
            sensor_args.array_size,
            output_channels,
            sensor_args.power_max,
        )

    if need_ideal:
        ideal_raw = video_sequence_pipeline.simulate_ideal_video_frame(power_maps, sensor_args, base_params)
        ideal_scaled = video_sequence_pipeline.scale_case_frame(
            ideal_raw,
            "ideal",
            sensor_args,
            case_range_bounds,
        )
        tensors["ideal"] = base_pipeline._frame_to_model_tensor(ideal_scaled, sensor_args)

    if need_nonideal:
        if sequence_sensor is None:
            raise ValueError("sequence_sensor is required when requesting the nonideal video-sequence case")
        nonideal_raw = sequence_sensor.simulate_frame(power_maps)
        nonideal_scaled = video_sequence_pipeline.scale_case_frame(
            nonideal_raw,
            "nonideal",
            sensor_args,
            case_range_bounds,
        )
        tensors["nonideal"] = base_pipeline._frame_to_model_tensor(nonideal_scaled, sensor_args)

    return tensors


def train_one_epoch(
    model,
    base_dataset,
    criterion,
    optimizer,
    device,
    epoch,
    args,
    base_params,
    train_transform,
    case_range_bounds,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    sensor_seed = args.sensor_rng_seed + args.epoch_seed_stride * max(epoch - 1, 0)
    sensor_args = build_sensor_args(args, sensor_seed)
    output_channels = infer_output_channels(base_dataset, sensor_args)

    sequence_sensor = None
    if args.train_case == "nonideal":
        sequence_sensor = video_sequence_pipeline.StatefulNonidealVideoSensor(sensor_args, base_params)
        warmup_sequence_sensor(sequence_sensor, output_channels, sensor_args)

    sample_limit = resolve_sample_limit(len(base_dataset), args.max_train_batches, args.batch_size)
    batch_images = []
    batch_labels = []
    num_optimizer_steps = 0
    effective_total_steps = max(1, (sample_limit + args.batch_size - 1) // args.batch_size)

    for dataset_index in range(sample_limit):
        image, label = base_dataset[dataset_index]
        if train_transform is not None:
            image = train_transform(image)

        case_tensors = build_case_tensors(
            image=image,
            sensor_args=sensor_args,
            base_params=base_params,
            output_channels=output_channels,
            requested_cases=[args.train_case],
            sequence_sensor=sequence_sensor,
            case_range_bounds=case_range_bounds,
        )
        batch_images.append(case_tensors[args.train_case])
        batch_labels.append(int(label))

        if len(batch_images) < args.batch_size and dataset_index + 1 < sample_limit:
            continue

        images = torch.stack(batch_images, dim=0).to(device, non_blocking=True)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        correct, batch_size = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        num_optimizer_steps += 1

        if num_optimizer_steps % args.print_every == 0 or dataset_index + 1 == sample_limit:
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = 100.0 * total_correct / max(total_samples, 1)
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}/{args.epochs}] Step [{num_optimizer_steps}/{effective_total_steps}] "
                f"samples={total_samples}/{sample_limit} loss={avg_loss:.4f} acc={avg_acc:.2f}% "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        batch_images.clear()
        batch_labels.clear()

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    epoch_time = time.time() - start_time
    return avg_loss, avg_acc, epoch_time, sample_limit


def evaluate_cases(model, base_dataset, args, base_params, device, cases, case_range_bounds):
    sensor_args = build_sensor_args(args, args.sensor_rng_seed)
    sample_limit = resolve_sample_limit(len(base_dataset), args.max_eval_batches, args.eval_batch_size)
    output_channels = infer_output_channels(base_dataset, sensor_args)
    needs_nonideal = "nonideal" in cases

    sequence_sensor = None
    if needs_nonideal:
        sequence_sensor = video_sequence_pipeline.StatefulNonidealVideoSensor(sensor_args, base_params)
        warmup_sequence_sensor(sequence_sensor, output_channels, sensor_args)

    eval_accumulator = video_sequence_pipeline.EvalAccumulator(model=model, device=device, cases=cases)

    print(
        f"Validation sensor backend: stateful sample-and-hold sequence, fps={sensor_args.video_fps:.4f}, "
        f"steps_per_frame={video_sequence_pipeline._steps_per_frame(sensor_args)}, cases={list(cases)}",
        flush=True,
    )

    for dataset_index in range(sample_limit):
        image, label = base_dataset[dataset_index]
        case_tensors = build_case_tensors(
            image=image,
            sensor_args=sensor_args,
            base_params=base_params,
            output_channels=output_channels,
            requested_cases=cases,
            sequence_sensor=sequence_sensor,
            case_range_bounds=case_range_bounds,
        )
        eval_accumulator.add(case_tensors, label)
        if eval_accumulator.pending_size() >= args.eval_batch_size:
            eval_accumulator.flush()

        processed = dataset_index + 1
        if processed % max(args.eval_batch_size * 10, args.eval_batch_size) == 0 or processed == sample_limit:
            elapsed = time.time() - eval_accumulator.start_time
            print(
                f"  val progress: samples={processed}/{sample_limit} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return eval_accumulator.finalize()


def format_eval_metrics(evaluation):
    parts = []
    for case in evaluation["cases"]:
        case_result = evaluation["cases"][case]
        parts.append(
            f"{case}: acc={case_result['accuracy']:.2f}% loss={case_result['loss']:.4f} samples={case_result['samples']}"
        )
    return " | ".join(parts)


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.selection_case not in args.eval_cases:
        raise ValueError(
            f"selection_case={args.selection_case!r} must be included in eval_cases={args.eval_cases!r}"
        )

    if args.num_workers != 0:
        print(
            "Warning: this fine-tuning pipeline keeps a stateful sensor sequence, so num_workers is ignored and "
            "samples are processed sequentially.",
            flush=True,
        )

    dataset_name = args.dataset.lower()
    num_classes = DATASET_CONFIG[dataset_name]["num_classes"]
    pretrained_path, output_path, run_dir, summary_json = resolve_run_paths(args)
    checkpoint_path = run_dir / "last_checkpoint.pth"
    history_path = run_dir / "history.csv"

    print(f"Using device: {base_pipeline.select_device()}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Training data root: {args.data_path}", flush=True)
    print(f"Pretrained checkpoint: {pretrained_path}", flush=True)
    print(f"Best model path: {output_path}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    train_dataset = base_pipeline.load_base_dataset(dataset_name, args.data_path, split="train")
    val_dataset = base_pipeline.load_base_dataset(dataset_name, args.data_path, split="test")
    base_params = base_pipeline.resolve_base_params(args.params_csv)
    device = base_pipeline.select_device()
    calibration_dataset = select_calibration_dataset(args, train_dataset, val_dataset)
    calibration_sensor_args = build_sensor_args(args, args.sensor_rng_seed)
    calibration_cases = {case for case in args.eval_cases if case in video_sequence_pipeline.RANGE_CASES}
    if args.train_case in video_sequence_pipeline.RANGE_CASES:
        calibration_cases.add(args.train_case)
    case_range_bounds = video_sequence_pipeline.compute_case_range_bounds(
        args=calibration_sensor_args,
        calibration_dataset=calibration_dataset,
        base_params=base_params,
        cases=sorted(calibration_cases),
    )

    model = resnet.resnet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    train_transform = build_train_augmentation(args.train_augment)

    start_epoch = 1
    best_metric = float("-inf")
    best_eval_results = None
    baseline_eval_results = None
    training_start_time = time.time()

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip_module_prefix(_extract_state_dict(checkpoint)))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        best_eval_results = checkpoint.get("best_eval_results")
        baseline_eval_results = checkpoint.get("baseline_eval_results")
        print(
            f"Resumed from epoch {checkpoint['epoch']} with best_{args.selection_case}={best_metric:.2f}%",
            flush=True,
        )
    else:
        if args.resume:
            print(f"Warning: no checkpoint found at {checkpoint_path}, starting from pretrained weights.", flush=True)
        load_weights(model, pretrained_path)
        print(f"Loaded pretrained weights from {pretrained_path}", flush=True)

    if start_epoch == 1:
        print("\n=== Baseline evaluation before fine-tuning ===", flush=True)
        baseline_eval_results = evaluate_cases(
            model=model,
            base_dataset=val_dataset,
            args=args,
            base_params=base_params,
            device=device,
            cases=args.eval_cases,
            case_range_bounds=case_range_bounds,
        )
        print(format_eval_metrics(baseline_eval_results), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} lr={current_lr:.6f} train_case={args.train_case}", flush=True)

        train_loss, train_acc, epoch_time, train_samples = train_one_epoch(
            model=model,
            base_dataset=train_dataset,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            base_params=base_params,
            train_transform=train_transform,
            case_range_bounds=case_range_bounds,
        )
        evaluation = evaluate_cases(
            model=model,
            base_dataset=val_dataset,
            args=args,
            base_params=base_params,
            device=device,
            cases=args.eval_cases,
            case_range_bounds=case_range_bounds,
        )
        scheduler.step()

        print(
            f"Epoch {epoch} finished: train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"time={epoch_time:.1f}s",
            flush=True,
        )
        print(format_eval_metrics(evaluation), flush=True)

        history_row = {
            "epoch": epoch,
            "lr": f"{current_lr:.8f}",
            "train_case": args.train_case,
            "train_samples": train_samples,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.4f}",
            "epoch_time_sec": f"{epoch_time:.2f}",
        }
        for case in args.eval_cases:
            case_result = evaluation["cases"][case]
            history_row[f"val_{case}_loss"] = f"{case_result['loss']:.6f}"
            history_row[f"val_{case}_acc"] = f"{case_result['accuracy']:.4f}"
            history_row[f"val_{case}_samples"] = case_result["samples"]
        append_history_row(history_path, history_row)

        selection_metric = float(evaluation["cases"][args.selection_case]["accuracy"])
        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_eval_results": best_eval_results,
            "baseline_eval_results": baseline_eval_results,
            "case_range_bounds": case_range_bounds,
            "args": vars(args),
            "pretrained_path": str(pretrained_path),
            "output_path": str(output_path),
            "run_dir": str(run_dir),
        }
        save_checkpoint(checkpoint_path, checkpoint_payload)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch}.pth", checkpoint_payload)

        if selection_metric >= best_metric:
            best_metric = selection_metric
            best_eval_results = evaluation
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(
                f"Saved new best model with {args.selection_case} acc={best_metric:.2f}% to {output_path}",
                flush=True,
            )

        checkpoint_payload["best_metric"] = best_metric
        checkpoint_payload["best_eval_results"] = best_eval_results
        save_checkpoint(checkpoint_path, checkpoint_payload)

    final_eval_results = None
    if output_path.exists():
        best_model = resnet.resnet18(num_classes=num_classes).to(device)
        load_weights(best_model, output_path)
        print("\n=== Final evaluation of best fine-tuned checkpoint ===", flush=True)
        final_eval_results = evaluate_cases(
            model=best_model,
            base_dataset=val_dataset,
            args=args,
            base_params=base_params,
            device=device,
            cases=args.eval_cases,
            case_range_bounds=case_range_bounds,
        )
        print(format_eval_metrics(final_eval_results), flush=True)

    total_elapsed = time.time() - training_start_time
    summary_payload = {
        "args": vars(args),
        "dataset": dataset_name,
        "pretrained_path": str(pretrained_path),
        "best_model_path": str(output_path),
        "run_dir": str(run_dir),
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_path),
        "selection_case": args.selection_case,
        "best_selection_accuracy": best_metric if best_metric > float("-inf") else None,
        "case_range_bounds": case_range_bounds,
        "baseline_eval": baseline_eval_results,
        "best_eval": best_eval_results,
        "final_eval": final_eval_results,
        "total_elapsed_sec": total_elapsed,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_json}", flush=True)
    print(
        f"Fine-tuning complete. Best {args.selection_case} accuracy: "
        f"{best_metric if best_metric > float('-inf') else float('nan'):.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
