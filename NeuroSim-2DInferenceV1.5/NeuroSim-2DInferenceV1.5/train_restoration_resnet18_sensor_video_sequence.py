import argparse
from collections import deque
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import generate_sensor_verification_images as base_pipeline
import generate_sensor_verification_images_video_sequence as video_sequence_pipeline
from finetune_resnet18_sensor_video_sequence import (
    DATASET_CONFIG,
    append_history_row,
    build_case_tensors,
    build_sensor_args,
    build_train_augmentation,
    infer_output_channels,
    load_weights,
    resolve_sample_limit,
    save_checkpoint,
    seed_everything,
    select_calibration_dataset,
    warmup_sequence_sensor,
)
from models import resnet
from models.restoration import TemporalTinyRestorationCNN


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CASE = "nonideal"
TARGET_CASE = "ideal"
CLASSIFICATION_CASES = ("raw", "ideal", "nonideal", "restored")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny causal temporal restoration frontend on stateful nonideal "
            "video-sequence sensor outputs before a frozen ResNet18 classifier."
        )
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT.parent / ".datasets"),
        help="Dataset root. Data will be read from <data_path>/<dataset>-data",
    )
    parser.add_argument(
        "--classifier_path",
        default=None,
        help="Frozen pretrained ResNet18 checkpoint. Defaults to models/resnet18_<dataset>.pth",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save the best restoration frontend state_dict",
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--selection_case",
        default="restored",
        choices=CLASSIFICATION_CASES,
        help="Validation case used to select the best checkpoint",
    )
    parser.add_argument(
        "--selection_metric",
        default="accuracy",
        choices=["accuracy", "restore_l1", "restore_mse"],
        help=(
            "Metric used to select the best checkpoint. The default keeps the legacy "
            "classification-accuracy selection; restore_l1/restore_mse select by "
            "minimum restored-vs-ideal reconstruction error."
        ),
    )
    parser.add_argument(
        "--eval_cases",
        nargs="+",
        default=list(CLASSIFICATION_CASES),
        choices=CLASSIFICATION_CASES,
        help="Validation classification cases to report after each epoch",
    )
    parser.add_argument(
        "--train_augment",
        type=int,
        default=1,
        help="Apply CIFAR-style RandomCrop/RandomHorizontalFlip before sensor simulation",
    )
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
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. Defaults to cpu to avoid interfering with active GPU jobs.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="If > 0, call torch.set_num_threads(cpu_threads). Useful for CPU training control.",
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
    parser.add_argument("--analog_readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc_enabled", type=int, default=0)
    parser.add_argument("--power_max", type=float, default=0.1)
    parser.add_argument(
        "--params_csv",
        default=str(base_pipeline.DEFAULT_PARAMS_CSV),
    )
    parser.add_argument(
        "--normalization_mode",
        default="calibration",
        choices=["physical", "calibration", "per_frame", "none"],
        help="Normalization path used by the current video-sequence pipeline; default keeps the calibrated legacy behavior.",
    )
    parser.add_argument("--prange1_density", type=float, default=None)
    parser.add_argument("--prange2_density", type=float, default=None)
    parser.add_argument("--pmin_density", type=float, default=None)
    parser.add_argument("--pmax_density", type=float, default=None)
    parser.add_argument("--device_area_cm2", type=float, default=None)
    parser.add_argument("--force_single_carrier", type=int, default=0)
    parser.add_argument("--single_r", type=float, default=None)
    parser.add_argument("--single_eta", type=float, default=None)
    parser.add_argument("--single_trise", type=float, default=None)
    parser.add_argument("--single_tfall", type=float, default=None)
    parser.add_argument("--trap_saturation_time", type=float, default=None)
    parser.add_argument("--trap_amplitude_pct", type=float, default=None)
    parser.add_argument("--noise_1f_density_1hz", type=float, default=None)
    parser.add_argument("--aging_tau_hours", type=float, default=None)
    parser.add_argument("--r_degradation_pct", type=float, default=None)
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
    parser.add_argument(
        "--spatial_variation_r_pct",
        type=float,
        default=None,
        help="Fixed pixel-wise responsivity variation percentage used by the stateful sensor backend.",
    )
    parser.add_argument(
        "--spatial_variation_cache_dir",
        default=None,
        help="Optional cache directory for fixed responsivity-variation random maps.",
    )
    parser.add_argument("--tia_gain_ohm", type=float, default=None)
    parser.add_argument("--integration_gain_v_per_c", type=float, default=None)

    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument(
        "--history_frames",
        type=int,
        default=4,
        help="Number of causal nonideal frames fed into the temporal restoration frontend",
    )
    parser.add_argument(
        "--reconstruction_loss",
        default="smooth_l1",
        choices=["l1", "mse", "smooth_l1"],
        help="Reconstruction loss between restored and ideal tensors",
    )
    parser.add_argument(
        "--reconstruction_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the restoration loss term",
    )
    parser.add_argument(
        "--classification_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the frozen-classifier cross-entropy term",
    )
    return parser.parse_args()


def resolve_run_paths(args):
    dataset_name = args.dataset.lower()
    classifier_path = (
        Path(args.classifier_path)
        if args.classifier_path
        else PROJECT_ROOT / "models" / f"resnet18_{dataset_name}.pth"
    )
    history_tag = f"h{args.history_frames}"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "models" / f"tiny_restoration_frontend_resnet18_{dataset_name}_video_sequence_{history_tag}.pth"
    )
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else PROJECT_ROOT / "artifacts" / "train_runs" / f"tiny_restoration_frontend_resnet18_{dataset_name}_video_sequence_{history_tag}"
    )
    summary_json = Path(args.summary_json) if args.summary_json else run_dir / "summary.json"
    return classifier_path, output_path, run_dir, summary_json


def resolve_device(device_name):
    if device_name == "auto":
        return base_pipeline.select_device()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if device_name == "mps":
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    raise ValueError(f"Unsupported device: {device_name}")


def create_reconstruction_criterion(name):
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.05)
    raise ValueError(f"Unsupported reconstruction loss: {name}")


def resolve_sequence_base_params(args):
    return video_sequence_pipeline.resolve_sequence_base_params(args)


def freeze_module(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def build_models(args, device, num_classes):
    classifier = resnet.resnet18(num_classes=num_classes).to(device)
    restoration = TemporalTinyRestorationCNN(
        in_channels=args.output_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        history_frames=args.history_frames,
    ).to(device)
    return restoration, classifier


def average_metric_sum(metric_sum, total_samples):
    return metric_sum / max(total_samples, 1)


def build_temporal_window(history_buffer, history_frames):
    frames = list(history_buffer)
    if not frames:
        raise ValueError("history_buffer must contain at least one frame")
    if len(frames) >= history_frames:
        frames = frames[-history_frames:]
    else:
        frames = [frames[0]] * (history_frames - len(frames)) + frames
    return torch.stack(frames, dim=0)


def format_eval_summary(evaluation):
    parts = []
    for case in evaluation["cases"]:
        case_result = evaluation["cases"][case]
        parts.append(
            f"{case}: acc={case_result['accuracy']:.2f}% loss={case_result['loss']:.4f} samples={case_result['samples']}"
        )
    recon = evaluation["restoration"]
    parts.append(
        f"restore: l1={recon['l1']:.6f} mse={recon['mse']:.6f} samples={recon['samples']}"
    )
    return " | ".join(parts)


def train_one_epoch(
    restoration_model,
    classifier_model,
    base_dataset,
    classification_criterion,
    reconstruction_criterion,
    optimizer,
    device,
    epoch,
    args,
    base_params,
    train_transform,
    case_range_bounds,
):
    restoration_model.train()
    classifier_model.eval()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_classification_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    sensor_seed = args.sensor_rng_seed + args.epoch_seed_stride * max(epoch - 1, 0)
    sensor_args = build_sensor_args(args, sensor_seed)
    output_channels = infer_output_channels(base_dataset, sensor_args)
    if output_channels != args.output_channels:
        raise ValueError(
            f"Resolved output_channels={output_channels}, but restoration model was configured for "
            f"{args.output_channels}. Please keep --output_channels compatible with the dataset."
        )

    sequence_sensor = video_sequence_pipeline.StatefulNonidealVideoSensor(sensor_args, base_params)
    warmup_sequence_sensor(sequence_sensor, output_channels, sensor_args)

    sample_limit = resolve_sample_limit(len(base_dataset), args.max_train_batches, args.batch_size)
    batch_temporal_nonideal = []
    batch_ideal = []
    batch_labels = []
    history_buffer = deque(maxlen=max(1, args.history_frames))
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
            requested_cases=[TARGET_CASE, TRAIN_CASE],
            sequence_sensor=sequence_sensor,
            case_range_bounds=case_range_bounds,
        )
        history_buffer.append(case_tensors[TRAIN_CASE])
        batch_temporal_nonideal.append(build_temporal_window(history_buffer, args.history_frames))
        batch_ideal.append(case_tensors[TARGET_CASE])
        batch_labels.append(int(label))

        if len(batch_temporal_nonideal) < args.batch_size and dataset_index + 1 < sample_limit:
            continue

        nonideal = torch.stack(batch_temporal_nonideal, dim=0).to(device, non_blocking=True)
        ideal = torch.stack(batch_ideal, dim=0).to(device, non_blocking=True)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        restored = restoration_model(nonideal)
        logits = classifier_model(restored)
        reconstruction_loss = reconstruction_criterion(restored, ideal)
        classification_loss = classification_criterion(logits, labels)
        total_batch_loss = (
            args.reconstruction_loss_weight * reconstruction_loss
            + args.classification_loss_weight * classification_loss
        )
        total_batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_correct += int(preds.eq(labels).sum().item())
            total_samples += int(batch_size)
            total_loss += float(total_batch_loss.item()) * batch_size
            total_reconstruction_loss += float(reconstruction_loss.item()) * batch_size
            total_classification_loss += float(classification_loss.item()) * batch_size
            num_optimizer_steps += 1

        if num_optimizer_steps % args.print_every == 0 or dataset_index + 1 == sample_limit:
            avg_loss = average_metric_sum(total_loss, total_samples)
            avg_recon_loss = average_metric_sum(total_reconstruction_loss, total_samples)
            avg_cls_loss = average_metric_sum(total_classification_loss, total_samples)
            avg_acc = 100.0 * total_correct / max(total_samples, 1)
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}/{args.epochs}] Step [{num_optimizer_steps}/{effective_total_steps}] "
                f"samples={total_samples}/{sample_limit} loss={avg_loss:.4f} "
                f"recon={avg_recon_loss:.4f} cls={avg_cls_loss:.4f} acc={avg_acc:.2f}% "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        batch_temporal_nonideal.clear()
        batch_ideal.clear()
        batch_labels.clear()

    avg_loss = average_metric_sum(total_loss, total_samples)
    avg_recon_loss = average_metric_sum(total_reconstruction_loss, total_samples)
    avg_cls_loss = average_metric_sum(total_classification_loss, total_samples)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    epoch_time = time.time() - start_time
    return {
        "loss": avg_loss,
        "reconstruction_loss": avg_recon_loss,
        "classification_loss": avg_cls_loss,
        "accuracy": avg_acc,
        "epoch_time_sec": epoch_time,
        "train_samples": sample_limit,
    }


def evaluate_model(
    restoration_model,
    classifier_model,
    base_dataset,
    args,
    base_params,
    device,
    cases,
    case_range_bounds,
):
    restoration_model.eval()
    classifier_model.eval()

    sensor_args = build_sensor_args(args, args.sensor_rng_seed)
    sample_limit = resolve_sample_limit(len(base_dataset), args.max_eval_batches, args.eval_batch_size)
    output_channels = infer_output_channels(base_dataset, sensor_args)
    if output_channels != args.output_channels:
        raise ValueError(
            f"Resolved output_channels={output_channels}, but restoration model was configured for "
            f"{args.output_channels}. Please keep --output_channels compatible with the dataset."
        )

    sequence_sensor = None
    needs_nonideal = TRAIN_CASE in cases or "restored" in cases
    if needs_nonideal:
        sequence_sensor = video_sequence_pipeline.StatefulNonidealVideoSensor(sensor_args, base_params)
        warmup_sequence_sensor(sequence_sensor, output_channels, sensor_args)

    eval_cases = [case for case in cases if case != "restored"]
    eval_accumulator = video_sequence_pipeline.EvalAccumulator(model=classifier_model, device=device, cases=cases)
    reconstruction_l1_sum = 0.0
    reconstruction_mse_sum = 0.0
    reconstruction_samples = 0
    history_buffer = deque(maxlen=max(1, args.history_frames))

    print(
        f"Validation sensor backend: stateful sample-and-hold sequence, fps={sensor_args.video_fps:.4f}, "
        f"steps_per_frame={video_sequence_pipeline._steps_per_frame(sensor_args)}, cases={list(cases)}",
        flush=True,
    )

    with torch.no_grad():
        for dataset_index in range(sample_limit):
            image, label = base_dataset[dataset_index]
            requested_cases = set(eval_cases)
            if "restored" in cases:
                requested_cases.add(TARGET_CASE)
                requested_cases.add(TRAIN_CASE)

            case_tensors = build_case_tensors(
                image=image,
                sensor_args=sensor_args,
                base_params=base_params,
                output_channels=output_channels,
                requested_cases=sorted(requested_cases),
                sequence_sensor=sequence_sensor,
                case_range_bounds=case_range_bounds,
            )

            if "restored" in cases:
                history_buffer.append(case_tensors[TRAIN_CASE])
                temporal_nonideal = build_temporal_window(history_buffer, args.history_frames)
                nonideal = temporal_nonideal.unsqueeze(0).to(device, non_blocking=True)
                ideal = case_tensors[TARGET_CASE].unsqueeze(0).to(device, non_blocking=True)
                restored = restoration_model(nonideal)
                diff = restored - ideal
                reconstruction_l1_sum += float(diff.abs().mean(dim=(1, 2, 3)).sum().item())
                reconstruction_mse_sum += float(diff.pow(2).mean(dim=(1, 2, 3)).sum().item())
                reconstruction_samples += int(restored.size(0))
                case_tensors["restored"] = restored.squeeze(0).cpu()

            eval_accumulator.add({case: case_tensors[case] for case in cases}, label)
            if eval_accumulator.pending_size() >= args.eval_batch_size:
                eval_accumulator.flush()

            processed = dataset_index + 1
            if processed % max(args.eval_batch_size * 10, args.eval_batch_size) == 0 or processed == sample_limit:
                elapsed = time.time() - eval_accumulator.start_time
                print(
                    f"  val progress: samples={processed}/{sample_limit} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    evaluation = eval_accumulator.finalize()
    evaluation["restoration"] = {
        "l1": average_metric_sum(reconstruction_l1_sum, reconstruction_samples),
        "mse": average_metric_sum(reconstruction_mse_sum, reconstruction_samples),
        "samples": reconstruction_samples,
    }
    return evaluation


def selection_value(evaluation, args):
    if args.selection_metric == "accuracy":
        return float(evaluation["cases"][args.selection_case]["accuracy"])
    if args.selection_metric == "restore_l1":
        return float(evaluation["restoration"]["l1"])
    if args.selection_metric == "restore_mse":
        return float(evaluation["restoration"]["mse"])
    raise ValueError(f"Unsupported selection_metric: {args.selection_metric}")


def selection_is_better(candidate, best, args):
    if args.selection_metric == "accuracy":
        return candidate >= best
    return candidate <= best


def selection_description(value, args):
    if args.selection_metric == "accuracy":
        return f"{args.selection_case} acc={value:.2f}%"
    return f"{args.selection_metric}={value:.6f}"


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    if args.selection_metric == "accuracy" and args.selection_case not in args.eval_cases:
        raise ValueError(
            f"selection_case={args.selection_case!r} must be included in eval_cases={args.eval_cases!r}"
        )
    if args.selection_metric in {"restore_l1", "restore_mse"} and "restored" not in args.eval_cases:
        raise ValueError("selection_metric restore_l1/restore_mse requires 'restored' in eval_cases")

    dataset_name = args.dataset.lower()
    num_classes = DATASET_CONFIG[dataset_name]["num_classes"]
    classifier_path, output_path, run_dir, summary_json = resolve_run_paths(args)
    checkpoint_path = run_dir / "last_checkpoint.pth"
    history_path = run_dir / "history.csv"
    device = resolve_device(args.device)

    print(f"Using device: {device}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Temporal history frames: {args.history_frames}", flush=True)
    print(
        f"Spatial responsivity variation: "
        f"{0.0 if args.spatial_variation_r_pct is None else float(args.spatial_variation_r_pct):.2f}%",
        flush=True,
    )
    print(f"Training data root: {args.data_path}", flush=True)
    print(f"Frozen classifier checkpoint: {classifier_path}", flush=True)
    print(f"Restoration model path: {output_path}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    train_dataset = base_pipeline.load_base_dataset(dataset_name, args.data_path, split="train")
    val_dataset = base_pipeline.load_base_dataset(dataset_name, args.data_path, split="test")
    base_params = resolve_sequence_base_params(args)
    if args.tia_gain_ohm is None:
        args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if args.integration_gain_v_per_c is None:
        args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    calibration_dataset = select_calibration_dataset(args, train_dataset, val_dataset)
    calibration_sensor_args = build_sensor_args(args, args.sensor_rng_seed)
    case_range_bounds = video_sequence_pipeline.compute_case_range_bounds(
        args=calibration_sensor_args,
        calibration_dataset=calibration_dataset,
        base_params=base_params,
        cases=[TARGET_CASE, TRAIN_CASE],
    )

    restoration_model, classifier_model = build_models(args, device, num_classes)
    load_weights(classifier_model, classifier_path)
    freeze_module(classifier_model)

    classification_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    reconstruction_criterion = create_reconstruction_criterion(args.reconstruction_loss)
    optimizer = torch.optim.AdamW(
        restoration_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    train_transform = build_train_augmentation(args.train_augment)

    start_epoch = 1
    best_metric = float("-inf") if args.selection_metric == "accuracy" else float("inf")
    best_eval_results = None
    baseline_eval_results = None
    training_start_time = time.time()

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restoration_model.load_state_dict(checkpoint["restoration_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        best_eval_results = checkpoint.get("best_eval_results")
        baseline_eval_results = checkpoint.get("baseline_eval_results")
        print(
            f"Resumed from epoch {checkpoint['epoch']} with best selection metric "
            f"{selection_description(best_metric, args)}",
            flush=True,
        )
    else:
        if args.resume:
            print(f"Warning: no checkpoint found at {checkpoint_path}, starting from scratch.", flush=True)

    print("\n=== Baseline evaluation before restoration training ===", flush=True)
    baseline_eval_results = evaluate_model(
        restoration_model=restoration_model,
        classifier_model=classifier_model,
        base_dataset=val_dataset,
        args=args,
        base_params=base_params,
        device=device,
        cases=args.eval_cases,
        case_range_bounds=case_range_bounds,
    )
    print(format_eval_summary(baseline_eval_results), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} lr={current_lr:.6f} train_case={TRAIN_CASE}->{TARGET_CASE}", flush=True)

        train_metrics = train_one_epoch(
            restoration_model=restoration_model,
            classifier_model=classifier_model,
            base_dataset=train_dataset,
            classification_criterion=classification_criterion,
            reconstruction_criterion=reconstruction_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            base_params=base_params,
            train_transform=train_transform,
            case_range_bounds=case_range_bounds,
        )
        evaluation = evaluate_model(
            restoration_model=restoration_model,
            classifier_model=classifier_model,
            base_dataset=val_dataset,
            args=args,
            base_params=base_params,
            device=device,
            cases=args.eval_cases,
            case_range_bounds=case_range_bounds,
        )
        scheduler.step()

        print(
            f"Epoch {epoch} finished: train_loss={train_metrics['loss']:.4f} "
            f"train_recon={train_metrics['reconstruction_loss']:.4f} "
            f"train_cls={train_metrics['classification_loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.2f}% "
            f"time={train_metrics['epoch_time_sec']:.1f}s",
            flush=True,
        )
        print(format_eval_summary(evaluation), flush=True)

        history_row = {
            "epoch": epoch,
            "lr": f"{current_lr:.8f}",
            "train_case": TRAIN_CASE,
            "target_case": TARGET_CASE,
            "train_samples": train_metrics["train_samples"],
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_reconstruction_loss": f"{train_metrics['reconstruction_loss']:.6f}",
            "train_classification_loss": f"{train_metrics['classification_loss']:.6f}",
            "train_acc": f"{train_metrics['accuracy']:.4f}",
            "epoch_time_sec": f"{train_metrics['epoch_time_sec']:.2f}",
            "val_restore_l1": f"{evaluation['restoration']['l1']:.6f}",
            "val_restore_mse": f"{evaluation['restoration']['mse']:.6f}",
            "val_restore_samples": evaluation["restoration"]["samples"],
        }
        for case in args.eval_cases:
            case_result = evaluation["cases"][case]
            history_row[f"val_{case}_loss"] = f"{case_result['loss']:.6f}"
            history_row[f"val_{case}_acc"] = f"{case_result['accuracy']:.4f}"
            history_row[f"val_{case}_samples"] = case_result["samples"]
        append_history_row(history_path, history_row)

        current_selection_value = selection_value(evaluation, args)
        checkpoint_payload = {
            "epoch": epoch,
            "restoration_model": restoration_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_eval_results": best_eval_results,
            "baseline_eval_results": baseline_eval_results,
            "case_range_bounds": case_range_bounds,
            "args": vars(args),
            "classifier_path": str(classifier_path),
            "output_path": str(output_path),
            "run_dir": str(run_dir),
        }
        save_checkpoint(checkpoint_path, checkpoint_payload)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch}.pth", checkpoint_payload)

        if selection_is_better(current_selection_value, best_metric, args):
            best_metric = current_selection_value
            best_eval_results = evaluation
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(restoration_model.state_dict(), output_path)
            print(
                f"Saved new best restoration model with {selection_description(best_metric, args)} to {output_path}",
                flush=True,
            )

        checkpoint_payload["best_metric"] = best_metric
        checkpoint_payload["best_eval_results"] = best_eval_results
        save_checkpoint(checkpoint_path, checkpoint_payload)

    final_eval_results = None
    if output_path.exists():
        best_restoration_model, _ = build_models(args, device, num_classes)
        best_restoration_model.load_state_dict(torch.load(output_path, map_location="cpu", weights_only=False))
        best_restoration_model = best_restoration_model.to(device)
        print("\n=== Final evaluation of best restoration checkpoint ===", flush=True)
        final_eval_results = evaluate_model(
            restoration_model=best_restoration_model,
            classifier_model=classifier_model,
            base_dataset=val_dataset,
            args=args,
            base_params=base_params,
            device=device,
            cases=args.eval_cases,
            case_range_bounds=case_range_bounds,
        )
        print(format_eval_summary(final_eval_results), flush=True)

    total_elapsed = time.time() - training_start_time
    summary_payload = {
        "args": vars(args),
        "dataset": dataset_name,
        "classifier_path": str(classifier_path),
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
        f"Restoration training complete. Best {args.selection_case} accuracy: "
        f"{best_metric if best_metric > float('-inf') else float('nan'):.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
