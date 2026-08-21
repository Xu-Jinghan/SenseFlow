import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import generate_sensor_verification_images as base_pipeline
import generate_sensor_verification_images_video_sequence as video_sequence_pipeline
from finetune_resnet18_sensor_video_sequence import (
    DATASET_CONFIG,
    build_sensor_args,
    infer_output_channels,
    load_weights,
    seed_everything,
    select_calibration_dataset,
    warmup_sequence_sensor,
)
from train_restoration_resnet18_sensor_video_sequence import (
    build_models,
    build_temporal_window,
    resolve_device,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EXPORT_CASES = ("raw", "ideal", "nonideal", "restored")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export original / ideal / nonideal / restored images for the temporal "
            "restoration frontend on the stateful video-sequence sensor pipeline."
        )
    )
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_CONFIG.keys()))
    parser.add_argument("--data_path", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=24)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor_rng_seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--cpu_threads", type=int, default=0)
    parser.add_argument(
        "--classifier_path",
        default=None,
        help="Frozen pretrained ResNet18 checkpoint. Defaults to models/resnet18_<dataset>.pth",
    )
    parser.add_argument(
        "--restoration_model_path",
        default=None,
        help="Restoration frontend checkpoint. Defaults to models/tiny_restoration_frontend_resnet18_<dataset>_video_sequence_h<history_frames>.pth",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save exported images and comparison panels",
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
    parser.add_argument("--params_csv", default=str(base_pipeline.DEFAULT_PARAMS_CSV))
    parser.add_argument(
        "--normalization_mode",
        default="calibration",
        choices=["physical", "calibration", "per_frame", "none"],
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
    )
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument("--range_calibration_split", default="train", choices=["train", "test"])
    parser.add_argument("--range_calibration_samples", type=int, default=1024)
    parser.add_argument("--use_noise_fn", type=int, default=1)
    parser.add_argument("--shot_noise", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=5000.0)
    parser.add_argument("--i_thermal", type=float, default=0.0)
    parser.add_argument("--startup_dark_frames", type=int, default=0)
    parser.add_argument("--spatial_variation_r_pct", type=float, default=None)
    parser.add_argument("--spatial_variation_cache_dir", default=None)
    parser.add_argument("--tia_gain_ohm", type=float, default=None)
    parser.add_argument("--integration_gain_v_per_c", type=float, default=None)

    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--history_frames", type=int, default=4)
    return parser.parse_args()


def resolve_default_paths(args):
    dataset_name = args.dataset.lower()
    history_tag = f"h{args.history_frames}"
    classifier_path = (
        Path(args.classifier_path)
        if args.classifier_path
        else PROJECT_ROOT / "models" / f"resnet18_{dataset_name}.pth"
    )
    restoration_model_path = (
        Path(args.restoration_model_path)
        if args.restoration_model_path
        else PROJECT_ROOT / "models" / f"tiny_restoration_frontend_resnet18_{dataset_name}_video_sequence_{history_tag}.pth"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "artifacts" / "restoration_visuals" / f"{dataset_name}_{args.split}_{history_tag}"
    )
    return classifier_path, restoration_model_path, output_dir


def prepare_output_dirs(output_dir):
    raw_dir = output_dir / "input"
    ideal_dir = output_dir / "sensor_ideal"
    nonideal_dir = output_dir / "sensor_nonideal"
    restored_dir = output_dir / "sensor_restored"
    compare_dir = output_dir / "comparison"
    for folder in [raw_dir, ideal_dir, nonideal_dir, restored_dir, compare_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    return raw_dir, ideal_dir, nonideal_dir, restored_dir, compare_dir


def _resolve_norm_key(post_norm, source_dataset):
    if post_norm == "none":
        return None
    if post_norm == "auto":
        return source_dataset
    return post_norm


def tensor_to_display_array(tensor, args):
    tensor = tensor.detach().cpu().float()
    norm_key = _resolve_norm_key(args.post_norm, args.dataset)
    if norm_key is not None:
        if norm_key not in base_pipeline._POST_NORM_STATS:
            raise ValueError(f"Unsupported post norm: {args.post_norm}")
        mean, std = base_pipeline._POST_NORM_STATS[norm_key]
        if tensor.shape[0] == 1:
            mean = (mean[0],)
            std = (std[0],)
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
        tensor = tensor * std_tensor + mean_tensor

    tensor = tensor.clamp(0.0, 1.0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got rank {tensor.ndim}")
    if tensor.shape[0] == 1:
        return tensor.squeeze(0).numpy()
    return np.transpose(tensor.numpy(), (1, 2, 0))


def frame_to_display_array(frame):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    return np.clip(frame, 0.0, 1.0)


def compose_quadtych(images, labels, tile_size):
    title_h = 42
    canvas = Image.new("RGB", (tile_size * len(images), tile_size + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (img, text) in enumerate(zip(images, labels)):
        x0 = idx * tile_size
        canvas.paste(img, (x0, title_h))
        draw.text((x0 + 8, 8), text, fill=(17, 24, 39))
    return canvas


def summarize_case_with_psnr(name, psnr_db=None, is_reference=False):
    if is_reference:
        return f"{name}\nreference"
    return f"{name}\nPSNR={base_pipeline.format_psnr(psnr_db)}"


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    classifier_path, restoration_model_path, output_dir = resolve_default_paths(args)
    device = resolve_device(args.device)
    raw_dir, ideal_dir, nonideal_dir, restored_dir, compare_dir = prepare_output_dirs(output_dir)

    print(f"Using device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Split: {args.split}", flush=True)
    print(f"Temporal history frames: {args.history_frames}", flush=True)
    print(f"Classifier checkpoint: {classifier_path}", flush=True)
    print(f"Restoration checkpoint: {restoration_model_path}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    dataset_name = args.dataset.lower()
    if not restoration_model_path.is_file():
        raise FileNotFoundError(
            f"Restoration checkpoint not found: {restoration_model_path}. "
            "Pass --restoration_model_path to use a specific trained model."
        )

    base_dataset = base_pipeline.load_base_dataset(dataset_name, args.data_path, split=args.split)
    calibration_train = base_pipeline.load_base_dataset(dataset_name, args.data_path, split="train")
    calibration_test = base_pipeline.load_base_dataset(dataset_name, args.data_path, split="test")
    calibration_dataset = select_calibration_dataset(args, calibration_train, calibration_test)
    base_params = video_sequence_pipeline.resolve_sequence_base_params(args)
    if args.tia_gain_ohm is None:
        args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if args.integration_gain_v_per_c is None:
        args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))

    sensor_args = build_sensor_args(args, args.sensor_rng_seed)
    case_range_bounds = video_sequence_pipeline.compute_case_range_bounds(
        args=sensor_args,
        calibration_dataset=calibration_dataset,
        base_params=base_params,
        cases=["ideal", "nonideal"],
    )

    num_classes = DATASET_CONFIG[dataset_name]["num_classes"]
    restoration_model, classifier_model = build_models(args, device, num_classes)
    load_weights(classifier_model, classifier_path)
    restoration_model.load_state_dict(torch.load(restoration_model_path, map_location="cpu", weights_only=False))
    restoration_model = restoration_model.to(device).eval()
    classifier_model = classifier_model.to(device).eval()

    stop_index = min(len(base_dataset), args.start_index + max(0, args.num_images))
    if stop_index <= args.start_index:
        raise ValueError(
            f"No images to export: start_index={args.start_index}, num_images={args.num_images}, dataset_size={len(base_dataset)}"
        )

    output_channels = infer_output_channels(base_dataset, sensor_args)
    if output_channels != args.output_channels:
        raise ValueError(
            f"Resolved output_channels={output_channels}, but exporter was configured for "
            f"{args.output_channels}. Please keep --output_channels compatible with the dataset."
        )

    sequence_sensor = video_sequence_pipeline.StatefulNonidealVideoSensor(sensor_args, base_params)
    warmup_sequence_sensor(sequence_sensor, output_channels, sensor_args)
    history_buffer = deque(maxlen=max(1, args.history_frames))
    manifest = []

    print(
        f"Exporting samples in sequence order [0, {stop_index - 1}] and saving the requested window "
        f"[{args.start_index}, {stop_index - 1}]",
        flush=True,
    )

    for dataset_index in range(stop_index):
        image, label = base_dataset[dataset_index]
        label_name = base_dataset.classes[label]
        if hasattr(video_sequence_pipeline, "build_sequence_power_maps"):
            power_maps = video_sequence_pipeline.build_sequence_power_maps(image, sensor_args, base_params)
        else:
            power_maps = base_pipeline.build_power_maps(
                image,
                sensor_args.array_size,
                output_channels,
                sensor_args.power_max,
            )

        ideal_raw = video_sequence_pipeline.simulate_ideal_video_frame(power_maps, sensor_args, base_params)
        nonideal_raw = sequence_sensor.simulate_frame(power_maps)
        ideal_scaled = video_sequence_pipeline.scale_case_frame(ideal_raw, "ideal", sensor_args, case_range_bounds)
        nonideal_scaled = video_sequence_pipeline.scale_case_frame(nonideal_raw, "nonideal", sensor_args, case_range_bounds)

        raw_tensor = base_pipeline._raw_image_to_model_tensor(image, sensor_args)
        ideal_tensor = base_pipeline._frame_to_model_tensor(ideal_scaled, sensor_args)
        nonideal_tensor = base_pipeline._frame_to_model_tensor(nonideal_scaled, sensor_args)
        history_buffer.append(nonideal_tensor)
        temporal_nonideal = build_temporal_window(history_buffer, args.history_frames)

        with torch.no_grad():
            restored_tensor = restoration_model(
                temporal_nonideal.unsqueeze(0).to(device, non_blocking=True)
            ).squeeze(0).cpu()

        if dataset_index < args.start_index:
            continue

        input_img = base_pipeline.to_rgb_image(np.asarray(image).astype(np.float32) / 255.0, args.tile_size)
        ideal_img = base_pipeline.to_rgb_image(
            base_pipeline.scaled_frame_to_unit_interval(ideal_scaled, args.readout, args.range_mode),
            args.tile_size,
        )
        nonideal_img = base_pipeline.to_rgb_image(
            base_pipeline.scaled_frame_to_unit_interval(nonideal_scaled, args.readout, args.range_mode),
            args.tile_size,
        )
        nonideal_display = frame_to_display_array(
            base_pipeline.scaled_frame_to_unit_interval(nonideal_scaled, args.readout, args.range_mode),
        )
        restored_display = tensor_to_display_array(restored_tensor, args)
        restored_img = base_pipeline.to_rgb_image(restored_display, args.tile_size)

        predictions = {
            "raw": base_pipeline.predict_tensor(classifier_model, raw_tensor, device, base_dataset.classes),
            "ideal": base_pipeline.predict_tensor(classifier_model, ideal_tensor, device, base_dataset.classes),
            "nonideal": base_pipeline.predict_tensor(classifier_model, nonideal_tensor, device, base_dataset.classes),
            "restored": base_pipeline.predict_tensor(classifier_model, restored_tensor, device, base_dataset.classes),
        }

        restored_display_np = np.asarray(restored_display, dtype=np.float32)
        ideal_display_np = frame_to_display_array(
            base_pipeline.scaled_frame_to_unit_interval(ideal_scaled, args.readout, args.range_mode),
        )
        nonideal_display_np = np.asarray(nonideal_display, dtype=np.float32)
        nonideal_display_psnr_db = base_pipeline.compute_psnr(ideal_display_np, nonideal_display_np)
        restored_display_psnr_db = base_pipeline.compute_psnr(ideal_display_np, restored_display_np)

        quadtych = compose_quadtych(
            images=[input_img, ideal_img, nonideal_img, restored_img],
            labels=[
                f"Input\nidx={dataset_index} gt={label_name}",
                summarize_case_with_psnr("Ideal", is_reference=True),
                summarize_case_with_psnr("Nonideal", nonideal_display_psnr_db),
                summarize_case_with_psnr("Restored", restored_display_psnr_db),
            ],
            tile_size=args.tile_size,
        )

        stem = f"sample_{dataset_index:04d}_{label_name}"
        input_path = raw_dir / f"{stem}_input.png"
        ideal_path = ideal_dir / f"{stem}_sensor_ideal.png"
        nonideal_path = nonideal_dir / f"{stem}_sensor_nonideal.png"
        restored_path = restored_dir / f"{stem}_sensor_restored.png"
        compare_path = compare_dir / f"{stem}_quadtych.png"

        input_img.save(input_path)
        ideal_img.save(ideal_path)
        nonideal_img.save(nonideal_path)
        restored_img.save(restored_path)
        quadtych.save(compare_path)

        item = {
            "dataset_index": dataset_index,
            "label_index": int(label),
            "label": label_name,
            "input_path": str(input_path),
            "ideal_path": str(ideal_path),
            "nonideal_path": str(nonideal_path),
            "restored_path": str(restored_path),
            "compare_path": str(compare_path),
            "history_frames": args.history_frames,
            "predictions": predictions,
            "correct": {
                case: (predictions[case]["pred_index"] == int(label))
                for case in EXPORT_CASES
            },
            "nonideal_psnr_db": base_pipeline.compute_psnr(ideal_raw, nonideal_raw),
            "nonideal_display_psnr_db": nonideal_display_psnr_db,
            "restored_display_psnr_db": restored_display_psnr_db,
            "restored_display_l1": float(np.mean(np.abs(restored_display_np - ideal_display_np))),
            "restored_display_mse": float(np.mean(np.square(restored_display_np - ideal_display_np))),
        }
        manifest.append(item)

        print(
            f"saved sample idx={dataset_index} class={label_name} "
            f"nonideal={predictions['nonideal']['pred_label']} restored={predictions['restored']['pred_label']} "
            f"compare={compare_path}",
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_payload = {
        "args": vars(args),
        "classifier_path": str(classifier_path),
        "restoration_model_path": str(restoration_model_path),
        "output_dir": str(output_dir),
        "num_images_exported": len(manifest),
        "samples": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
