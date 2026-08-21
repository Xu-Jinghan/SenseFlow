import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from detection_sensor_trace import CenterPixelWaveformRecorder


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
LOCAL_DATA_PATH = REPO_ROOT / ".datasets"
DEFAULT_PARAMS_CSV = REPO_ROOT / "outputs" / "case2_noiseless_x2_eval_cifar10" / "case2_native_eval_params.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "detection_runs" / "yolo11n_coco_val2017_sensor_subset"
EVAL_CASES = ("raw", "ideal", "nonideal")


def _default_data_path():
    if LOCAL_DATA_PATH.exists():
        return str(LOCAL_DATA_PATH)
    return str(PROJECT_ROOT / ".datasets")


def _default_model_path():
    local_model = LOCAL_DATA_PATH / "models" / "yolo11n.pt"
    if local_model.exists():
        return str(local_model)
    return "yolo11n.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sensor non-ideal + software YOLO11n object-detection evaluation on a COCO "
            "val2017 subset. This script does not train and does not invoke CIM/NeuroSim."
        )
    )
    parser.add_argument("--model", default=_default_model_path(), help="YOLO model name or checkpoint path.")
    parser.add_argument("--data-path", default=_default_data_path(), help="Base dataset directory.")
    parser.add_argument(
        "--coco-root",
        default=None,
        help="COCO root directory. Defaults to <data-path>/coco if it exists, otherwise <data-path>.",
    )
    parser.add_argument("--image-dir", default=None, help="Optional explicit COCO val2017 image directory.")
    parser.add_argument(
        "--annotations-json",
        default=None,
        help=(
            "Optional COCO instances_val2017.json. Used to create YOLO-format labels when "
            "labels/val2017 txt files are not present."
        ),
    )
    parser.add_argument("--subset-size", type=int, default=500, help="Number of val2017 images to use.")
    parser.add_argument("--start-index", type=int, default=0, help="First sorted image index used without shuffle.")
    parser.add_argument("--shuffle", type=int, default=0, help="Shuffle val2017 images before selecting the subset.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--eval-cases",
        nargs="+",
        default=["nonideal"],
        choices=EVAL_CASES,
        help="Input cases evaluated by YOLO: raw, ideal, nonideal.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device string: auto, cpu, 0, 0,1, etc. auto uses CUDA:0 when available.",
    )
    parser.add_argument("--run-predict", type=int, default=1, help="Save per-image predictions JSON.")
    parser.add_argument("--run-val", type=int, default=1, help="Run mAP validation on the generated subset.")
    parser.add_argument("--predict-conf", type=float, default=0.25)
    parser.add_argument("--val-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-json", type=int, default=1, help="Ask Ultralytics val to save COCO JSON results.")
    parser.add_argument("--save-txt", type=int, default=0, help="Ask Ultralytics val to save YOLO-format txt labels.")
    parser.add_argument("--plots", type=int, default=1, help="Ask Ultralytics val to save metric plots.")
    parser.add_argument("--save-annotated", type=int, default=1, help="Save annotated prediction images.")
    parser.add_argument("--max-save-images", type=int, default=20, help="Maximum annotated images to save per case.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    # Sensor frontend parameters. Names intentionally mirror the video-sequence
    # verification script so fitted parameter CSVs and sweep settings can be reused.
    parser.add_argument("--sensor-array-size", type=int, default=320)
    parser.add_argument("--sensor-output-channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog-readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc-enabled", type=int, default=0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--adc-calibration-low", type=float, default=None)
    parser.add_argument("--adc-calibration-high", type=float, default=None)
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--params-csv", default=None, help=f"Sensor parameter CSV. Defaults to {DEFAULT_PARAMS_CSV}.")
    parser.add_argument("--normalization-mode", default="physical", choices=["physical", "per_frame", "none"])
    parser.add_argument("--range-mode", default="auto", choices=["auto", "minmax", "signed", "none"])
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
    parser.add_argument("--x2-alpha", type=float, default=None, help="Override x2 alpha; default uses the case2 CSV value.")
    parser.add_argument("--x2-beta", type=float, default=None, help="Override x2 beta; default uses the CSV value.")
    parser.add_argument(
        "--detection-dark-current-a",
        type=float,
        default=0.0,
        help="Dark-current baseline used for detection sensor simulation. Default disables dark current.",
    )
    parser.add_argument("--noise-1f-density-1hz", type=float, default=None)
    parser.add_argument("--aging-tau-hours", type=float, default=None)
    parser.add_argument("--r-degradation-pct", type=float, default=None)
    parser.add_argument("--spatial-variation-r-pct", type=float, default=0.0)
    parser.add_argument("--tia-gain-ohm", type=float, default=None)
    parser.add_argument("--integration-gain-v-per-c", type=float, default=None)
    parser.add_argument("--shot-noise", type=int, default=0)
    parser.add_argument("--use-noise-fn", type=int, default=0)
    parser.add_argument("--save-center-waveform", type=int, default=1)
    parser.add_argument("--center-waveform-channel", type=int, default=1)
    parser.add_argument("--center-waveform-max-frames", type=int, default=120)
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: torch. Use the same environment as the existing inference scripts, "
            "for example: conda run -n xjhenv python ..."
        ) from exc
    return torch


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it in the active environment, "
            "for example: pip install ultralytics"
        ) from exc
    return YOLO


def import_pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: Pillow. Use the same environment as the existing image/sensor "
            "scripts, or install it with: pip install pillow"
        ) from exc
    return Image


def import_sensor_pipeline():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import generate_sensor_verification_images as base_pipeline
    import generate_sensor_verification_images_video_sequence as video_pipeline

    return base_pipeline, video_pipeline


def torch_cuda_available():
    try:
        torch = import_torch()
    except SystemExit:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(device):
    device = str(device).strip()
    if device.lower() == "auto":
        return 0 if torch_cuda_available() else "cpu"
    return device


def resolve_coco_root(args):
    if args.coco_root:
        return Path(args.coco_root).expanduser().resolve()

    data_path = Path(args.data_path).expanduser().resolve()
    coco_child = data_path / "coco"
    if coco_child.exists():
        return coco_child
    return data_path


def resolve_val_image_dir(args, coco_root):
    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {image_dir}")
        return image_dir

    candidates = [
        coco_root / "images" / "val2017",
        coco_root / "val2017",
        coco_root / "images" / "val",
        coco_root / "val",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find COCO val images. Expected one of:\n"
        f"  {tried}\n"
        "Pass --image-dir if your val2017 images live somewhere else."
    )


def expected_label_dir(image_dir):
    parts = list(image_dir.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            label_parts = parts[:idx] + ["labels"] + parts[idx + 1 :]
            return Path(*label_parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def resolve_annotations_json(args, coco_root):
    if args.annotations_json:
        annotation_path = Path(args.annotations_json).expanduser().resolve()
        if not annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotations JSON not found: {annotation_path}")
        return annotation_path

    candidates = [
        coco_root / "annotations" / "instances_val2017.json",
        coco_root / "instances_val2017.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def list_val_images(image_dir):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    return sorted(images)


def select_subset(images, subset_size, start_index, shuffle, seed):
    if subset_size <= 0:
        raise ValueError(f"subset-size must be positive, got {subset_size}")
    if not images:
        raise ValueError("No images found for COCO subset selection.")

    selected = list(images)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
        return selected[: min(subset_size, len(selected))]

    start_index = max(0, int(start_index))
    stop_index = min(start_index + subset_size, len(selected))
    return selected[start_index:stop_index]


def normalize_names(names):
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {idx: str(value) for idx, value in enumerate(names)}


def yaml_quote(value):
    return json.dumps(str(value).replace("\\", "/"))


def write_case_yaml(case_dir, image_paths, names):
    subset_file = case_dir / f"{case_dir.name}_images.txt"
    yaml_file = case_dir / f"{case_dir.name}.yaml"
    subset_file.write_text(
        "\n".join(str(path.resolve()).replace("\\", "/") for path in image_paths) + "\n",
        encoding="utf-8",
    )

    names = normalize_names(names)
    yaml_lines = [
        f"path: {yaml_quote(case_dir.resolve())}",
        "train: images/val2017",
        f"val: {yaml_quote(subset_file.resolve())}",
        "names:",
    ]
    for class_id in sorted(names):
        yaml_lines.append(f"  {class_id}: {yaml_quote(names[class_id])}")
    yaml_file.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return subset_file, yaml_file


def load_coco_labels_from_json(annotation_path, image_paths, model_names):
    if annotation_path is None:
        return {}

    names = normalize_names(model_names)
    class_id_by_name = {name: class_id for class_id, name in names.items()}
    selected_names = {path.name for path in image_paths}

    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    image_meta_by_id = {}
    image_id_by_name = {}
    for image_info in coco.get("images", []):
        file_name = image_info.get("file_name")
        if file_name not in selected_names:
            continue
        image_id = int(image_info["id"])
        image_meta_by_id[image_id] = {
            "file_name": file_name,
            "width": float(image_info["width"]),
            "height": float(image_info["height"]),
        }
        image_id_by_name[file_name] = image_id

    category_name_by_id = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    labels_by_name = {file_name: [] for file_name in selected_names}

    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id not in image_meta_by_id:
            continue
        if int(ann.get("iscrowd", 0)):
            continue

        category_name = category_name_by_id.get(int(ann.get("category_id", -1)))
        if category_name not in class_id_by_name:
            continue

        x, y, w, h = [float(value) for value in ann.get("bbox", [0, 0, 0, 0])]
        if w <= 0.0 or h <= 0.0:
            continue

        meta = image_meta_by_id[image_id]
        img_w = max(float(meta["width"]), 1.0)
        img_h = max(float(meta["height"]), 1.0)
        xc = np.clip((x + 0.5 * w) / img_w, 0.0, 1.0)
        yc = np.clip((y + 0.5 * h) / img_h, 0.0, 1.0)
        bw = np.clip(w / img_w, 0.0, 1.0)
        bh = np.clip(h / img_h, 0.0, 1.0)
        class_id = class_id_by_name[category_name]
        labels_by_name[meta["file_name"]].append(
            f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    return {
        file_name: "\n".join(labels) + ("\n" if labels else "")
        for file_name, labels in labels_by_name.items()
        if file_name in image_id_by_name
    }


def write_label_file(source_label_dir, image_path, target_label_dir, converted_labels):
    target_label_dir.mkdir(parents=True, exist_ok=True)
    source_label = source_label_dir / f"{image_path.stem}.txt"
    target_label = target_label_dir / f"{image_path.stem}.txt"
    if source_label.is_file():
        shutil.copy2(source_label, target_label)
        return "copied"
    if image_path.name in converted_labels:
        target_label.write_text(converted_labels[image_path.name], encoding="utf-8")
        return "converted"
    target_label.write_text("", encoding="utf-8")
    return "missing"


def make_sensor_args(args, base_pipeline):
    return SimpleNamespace(
        params_csv=args.params_csv or str(DEFAULT_PARAMS_CSV),
        array_size=int(args.sensor_array_size),
        output_channels=int(args.sensor_output_channels),
        sensor_rng_seed=int(args.sensor_rng_seed),
        seed=int(args.seed),
        readout=args.readout,
        analog_readout=args.analog_readout,
        adc_enabled=int(args.adc_enabled),
        adc_bits=int(args.adc_bits),
        adc_full_scale=args.adc_full_scale,
        adc_calibration_low=args.adc_calibration_low,
        adc_calibration_high=args.adc_calibration_high,
        video_fps=float(args.video_fps),
        fps_sim=float(args.fps_sim),
        startup_dark_frames=int(args.startup_dark_frames),
        normalization_mode=args.normalization_mode,
        range_mode=args.range_mode,
        percentile_low=float(args.percentile_low),
        percentile_high=float(args.percentile_high),
        prange1_density=args.prange1_density,
        prange2_density=args.prange2_density,
        pmin_density=args.pmin_density,
        pmax_density=args.pmax_density,
        device_area_cm2=args.device_area_cm2,
        force_single_carrier=int(args.force_single_carrier),
        single_r=args.single_r,
        single_eta=args.single_eta,
        single_trise=args.single_trise,
        single_tfall=args.single_tfall,
        trap_saturation_time=args.trap_saturation_time,
        trap_amplitude_pct=args.trap_amplitude_pct,
        noise_1f_density_1hz=args.noise_1f_density_1hz,
        aging_tau_hours=args.aging_tau_hours,
        r_degradation_pct=args.r_degradation_pct,
        spatial_variation_r_pct=args.spatial_variation_r_pct,
        tia_gain_ohm=args.tia_gain_ohm,
        integration_gain_v_per_c=args.integration_gain_v_per_c,
        shot_noise=int(args.shot_noise),
        use_noise_fn=int(args.use_noise_fn),
        i_thermal=0.0,
        bandwidth=5000.0,
        total_sequence_frames=0,
    )


def apply_detection_x2_overrides(args, base_params):
    params = dict(base_params)
    params["trap_mode"] = str(args.x2_trap_mode)
    params["trap_output_mode"] = str(args.x2_trap_output_mode)
    if args.x2_alpha is not None:
        params["alpha"] = float(args.x2_alpha)
    if args.x2_beta is not None:
        params["beta"] = float(args.x2_beta)
    if getattr(args, "detection_dark_current_a", None) is not None:
        params["dark_current_a"] = float(args.detection_dark_current_a)
    return params


def frame_to_pil(unit_frame, image_module):
    frame = np.asarray(unit_frame, dtype=np.float32)
    if frame.ndim == 2:
        frame = frame[:, :, None]
    elif frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    frame = np.clip(frame, 0.0, 1.0)
    return image_module.fromarray((frame * 255.0).round().astype(np.uint8), mode="RGB")


def resize_raw_image(image, size, image_module):
    return image.convert("RGB").resize((size, size), resample=image_module.Resampling.BILINEAR)


def scale_sensor_frame(frame, case_name, sensor_args, case_range_bounds, base_pipeline, video_pipeline):
    bounds = case_range_bounds.get(case_name) if case_range_bounds else None
    scaled = base_pipeline.scale_frame(
        frame,
        sensor_args.readout,
        video_pipeline.effective_frame_range_mode(sensor_args),
        sensor_args.percentile_low,
        sensor_args.percentile_high,
        bounds=bounds,
    )
    unit = base_pipeline.scaled_frame_to_unit_interval(
        scaled,
        sensor_args.readout,
        video_pipeline.effective_frame_range_mode(sensor_args),
    )
    return unit


def prepare_case_images(args, image_paths, source_label_dir, converted_labels, model_names, output_dir):
    Image = import_pil_image()
    base_pipeline, video_pipeline = import_sensor_pipeline()
    sensor_args = make_sensor_args(args, base_pipeline)
    if sensor_args.analog_readout is None:
        sensor_args.analog_readout = video_pipeline.resolve_analog_readout_mode(sensor_args)
    sensor_args.frame_range_mode_override = video_pipeline.effective_frame_range_mode(sensor_args)

    base_params = video_pipeline.resolve_sequence_base_params(sensor_args)
    base_params = apply_detection_x2_overrides(args, base_params)
    if sensor_args.tia_gain_ohm is None:
        sensor_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if sensor_args.integration_gain_v_per_c is None:
        sensor_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))

    case_range_bounds = {}
    if args.normalization_mode == "physical":
        case_range_bounds = video_pipeline.compute_physical_case_range_bounds(
            sensor_args,
            base_params,
            [case for case in args.eval_cases if case in {"ideal", "nonideal"}],
        )

    sequence_sensor = None
    center_trace_recorder = CenterPixelWaveformRecorder(
        enabled=("nonideal" in args.eval_cases and bool(args.save_center_waveform)),
        output_dir=output_dir / "center_pixel_waveform",
        target_channel=args.center_waveform_channel,
        max_plot_frames=args.center_waveform_max_frames,
        title="COCO val2017 center pixel case2 native noiseless nonideal waveform",
    )
    if "nonideal" in args.eval_cases:
        sensor_args.total_sequence_frames = len(image_paths) + max(0, int(args.startup_dark_frames))
        sequence_sensor = video_pipeline.StatefulNonidealVideoSensor(args=sensor_args, base_params=base_params)
        zero_power = np.zeros(
            (sensor_args.output_channels, sensor_args.array_size, sensor_args.array_size),
            dtype=np.float64,
        )
        if sensor_args.output_channels == 1:
            zero_power = zero_power[0]
        for _ in range(max(0, int(args.startup_dark_frames))):
            sequence_sensor.simulate_frame(zero_power)

    case_payloads = {}
    for case_name in args.eval_cases:
        case_dir = output_dir / case_name
        image_dir = case_dir / "images" / "val2017"
        label_dir = case_dir / "labels" / "val2017"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        case_payloads[case_name] = {
            "case_dir": case_dir,
            "image_dir": image_dir,
            "label_dir": label_dir,
            "image_paths": [],
            "copied_labels": 0,
            "converted_labels": 0,
            "missing_labels": 0,
        }

    psnr_values = []
    started = time.time()
    for image_index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        resized_raw = resize_raw_image(image, sensor_args.array_size, Image)
        generated = {}

        if "raw" in args.eval_cases:
            generated["raw"] = resized_raw

        power_maps = None
        ideal_raw = None
        if "ideal" in args.eval_cases or "nonideal" in args.eval_cases:
            power_maps = video_pipeline.build_sequence_power_maps(image, sensor_args, base_params)

        if "ideal" in args.eval_cases:
            ideal_raw = video_pipeline.simulate_ideal_video_frame(power_maps, sensor_args, base_params)
            ideal_unit = scale_sensor_frame(
                ideal_raw,
                "ideal",
                sensor_args,
                case_range_bounds,
                base_pipeline,
                video_pipeline,
            )
            generated["ideal"] = frame_to_pil(ideal_unit, Image)

        if "nonideal" in args.eval_cases:
            if sequence_sensor is None:
                raise RuntimeError("Nonideal case requested without a sensor backend.")
            if center_trace_recorder.enabled:
                nonideal_raw, center_trace = sequence_sensor.simulate_frame(
                    power_maps,
                    record_center_trace=True,
                )
                center_trace_recorder.record(
                    image_index,
                    image_path.name,
                    power_maps,
                    center_trace,
                    dt_s=sequence_sensor.dt,
                    video_fps=sensor_args.video_fps,
                )
            else:
                nonideal_raw = sequence_sensor.simulate_frame(power_maps)
            nonideal_unit = scale_sensor_frame(
                nonideal_raw,
                "nonideal",
                sensor_args,
                case_range_bounds,
                base_pipeline,
                video_pipeline,
            )
            generated["nonideal"] = frame_to_pil(nonideal_unit, Image)
            if ideal_raw is not None:
                psnr_values.append(float(base_pipeline.compute_psnr(ideal_raw, nonideal_raw)))

        for case_name, pil_image in generated.items():
            payload = case_payloads[case_name]
            output_image_path = payload["image_dir"] / image_path.name
            pil_image.save(output_image_path, quality=95)
            payload["image_paths"].append(output_image_path)
            label_status = write_label_file(source_label_dir, image_path, payload["label_dir"], converted_labels)
            if label_status == "copied":
                payload["copied_labels"] += 1
            elif label_status == "converted":
                payload["converted_labels"] += 1
            else:
                payload["missing_labels"] += 1

        processed = image_index + 1
        if processed % 50 == 0 or processed == len(image_paths):
            print(f"  sensor export progress: {processed}/{len(image_paths)}", flush=True)

    for case_name, payload in case_payloads.items():
        subset_file, yaml_file = write_case_yaml(payload["case_dir"], payload["image_paths"], model_names)
        payload["subset_file"] = subset_file
        payload["yaml_file"] = yaml_file

    elapsed_s = time.time() - started
    center_trace_summary = center_trace_recorder.finalize(sensor_args, base_params)
    return {
        "sensor_args": vars(sensor_args),
        "base_params": {
            key: (None if value is None else float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in base_params.items()
            if key not in {"spatial_variation_cache_dir"}
        },
        "case_range_bounds": case_range_bounds,
        "case_payloads": case_payloads,
        "psnr_ideal_nonideal_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "psnr_ideal_nonideal_min": float(np.min(psnr_values)) if psnr_values else None,
        "psnr_ideal_nonideal_max": float(np.max(psnr_values)) if psnr_values else None,
        "center_pixel_waveform": center_trace_summary,
        "elapsed_s": elapsed_s,
    }


def boxes_to_records(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    names = normalize_names(result.names)

    records = []
    for box_xyxy, score, class_id in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = [float(value) for value in box_xyxy]
        records.append(
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(class_id)),
                "confidence": float(score),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return records


def run_predict(model, image_paths, args, device, case_dir):
    predict_dir = case_dir / "predict"
    annotated_dir = predict_dir / "annotated"
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_detections = 0
    started = time.time()
    source = [str(path) for path in image_paths]
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device,
        conf=args.predict_conf,
        iou=args.iou,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )
    for image_index, result in enumerate(results):
        detections = boxes_to_records(result)
        total_detections += len(detections)
        image_path = Path(result.path)
        records.append(
            {
                "subset_index": image_index,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "orig_shape": list(result.orig_shape),
                "detections": detections,
            }
        )
        if args.save_annotated and image_index < max(0, args.max_save_images):
            result.save(filename=str(annotated_dir / image_path.name))

    elapsed_s = time.time() - started
    predictions_path = predict_dir / "predictions.json"
    predict_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {
        "predictions_path": str(predictions_path),
        "annotated_dir": str(annotated_dir) if args.save_annotated else None,
        "num_images": len(records),
        "num_detections": int(total_detections),
        "elapsed_s": elapsed_s,
        "images_per_s": float(len(records) / elapsed_s) if elapsed_s > 0 else None,
    }


def metric_value(obj, path):
    current = obj
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return current


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_val(model, yaml_file, args, device, case_dir):
    started = time.time()
    metrics = model.val(
        data=str(yaml_file),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch_size,
        workers=args.num_workers,
        device=device,
        conf=args.val_conf,
        iou=args.iou,
        max_det=args.max_det,
        save_json=bool(args.save_json),
        save_txt=bool(args.save_txt),
        plots=bool(args.plots),
        project=str(case_dir),
        name="val",
        exist_ok=True,
        verbose=True,
    )
    elapsed_s = time.time() - started
    save_dir = getattr(metrics, "save_dir", None)
    speed = getattr(metrics, "speed", None)
    return {
        "save_dir": str(save_dir) if save_dir is not None else None,
        "elapsed_s": elapsed_s,
        "box_map": metric_value(metrics, ["box", "map"]),
        "box_map50": metric_value(metrics, ["box", "map50"]),
        "box_map75": metric_value(metrics, ["box", "map75"]),
        "box_mean_precision": metric_value(metrics, ["box", "mp"]),
        "box_mean_recall": metric_value(metrics, ["box", "mr"]),
        "speed": json_safe(speed),
    }


def main():
    args = parse_args()
    if not args.run_predict and not args.run_val:
        raise ValueError("At least one of --run-predict or --run-val must be enabled.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    coco_root = resolve_coco_root(args)
    source_image_dir = resolve_val_image_dir(args, coco_root)
    source_label_dir = expected_label_dir(source_image_dir)
    annotations_json = resolve_annotations_json(args, coco_root)
    all_images = list_val_images(source_image_dir)
    subset_images = select_subset(
        all_images,
        subset_size=args.subset_size,
        start_index=args.start_index,
        shuffle=bool(args.shuffle),
        seed=args.seed,
    )
    if not subset_images:
        raise ValueError("Selected subset is empty; check --subset-size and --start-index.")

    YOLO = import_yolo()
    device = resolve_device(args.device)
    model = YOLO(args.model)

    print("=" * 72, flush=True)
    print("Sensor nonideal + software YOLO detection evaluation", flush=True)
    print(f"Pipeline: sensor / model eval only; CIM/NeuroSim disabled", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"COCO root: {coco_root}", flush=True)
    print(f"Source images: {source_image_dir}", flush=True)
    print(f"Source labels: {source_label_dir}", flush=True)
    print(f"Annotations JSON: {annotations_json}", flush=True)
    print(f"Subset images: {len(subset_images)} / {len(all_images)}", flush=True)
    print(f"Cases: {args.eval_cases}", flush=True)
    print(f"Sensor output size: {args.sensor_array_size}x{args.sensor_array_size}", flush=True)
    print(f"Sensor params: {args.params_csv or DEFAULT_PARAMS_CSV}", flush=True)
    print(
        f"Sensor nonideal extras: shot_noise={args.shot_noise} "
        f"use_noise_fn={args.use_noise_fn} spatial_variation_r_pct={args.spatial_variation_r_pct}",
        flush=True,
    )
    print(
        f"Sensor x2 mode: trap_mode={args.x2_trap_mode} "
        f"trap_output_mode={args.x2_trap_output_mode} alpha={args.x2_alpha or 'case2_csv'} beta={args.x2_beta}",
        flush=True,
    )
    print(f"Center waveform export: {bool(args.save_center_waveform)}", flush=True)

    if args.run_val and not source_label_dir.is_dir() and annotations_json is None:
        raise FileNotFoundError(
            "YOLO-format COCO labels are required for --run-val 1, but the expected source label "
            f"directory does not exist: {source_label_dir}\n"
            "No COCO instances_val2017.json was found either. Prepare labels/val2017, pass "
            "--annotations-json, pass --run-val 0 for prediction-only inference, or point "
            "--image-dir to a COCO layout with sibling labels/val2017."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_labels = load_coco_labels_from_json(annotations_json, subset_images, model.names)
    prep = prepare_case_images(
        args,
        subset_images,
        source_label_dir,
        converted_labels,
        model.names,
        output_dir,
    )

    summary = {
        "pipeline": "sensor / model eval",
        "cim_neurosim": "disabled",
        "task": "object_detection",
        "model": args.model,
        "device": device,
        "cuda_available": torch_cuda_available(),
        "coco_root": str(coco_root),
        "source_image_dir": str(source_image_dir),
        "source_label_dir": str(source_label_dir),
        "annotations_json": str(annotations_json) if annotations_json is not None else None,
        "subset_size": len(subset_images),
        "num_available_images": len(all_images),
        "args": vars(args),
        "sensor": {
            "elapsed_s": prep["elapsed_s"],
            "args": prep["sensor_args"],
            "base_params": prep["base_params"],
            "case_range_bounds": prep["case_range_bounds"],
            "psnr_ideal_nonideal_mean": prep["psnr_ideal_nonideal_mean"],
            "psnr_ideal_nonideal_min": prep["psnr_ideal_nonideal_min"],
            "psnr_ideal_nonideal_max": prep["psnr_ideal_nonideal_max"],
            "center_pixel_waveform": prep["center_pixel_waveform"],
        },
        "cases": {},
    }

    for case_name in args.eval_cases:
        payload = prep["case_payloads"][case_name]
        case_summary = {
            "case_dir": str(payload["case_dir"]),
            "image_dir": str(payload["image_dir"]),
            "label_dir": str(payload["label_dir"]),
            "subset_file": str(payload["subset_file"]),
            "yaml_file": str(payload["yaml_file"]),
            "num_images": len(payload["image_paths"]),
            "copied_labels": int(payload["copied_labels"]),
            "converted_labels": int(payload["converted_labels"]),
            "missing_labels": int(payload["missing_labels"]),
            "predict": None,
            "val": None,
        }

        if args.run_predict:
            case_summary["predict"] = run_predict(
                model,
                payload["image_paths"],
                args,
                device,
                payload["case_dir"],
            )
            print(
                f"{case_name} predict: images={case_summary['predict']['num_images']} "
                f"detections={case_summary['predict']['num_detections']} "
                f"elapsed={case_summary['predict']['elapsed_s']:.2f}s",
                flush=True,
            )

        if args.run_val:
            case_summary["val"] = run_val(
                model,
                payload["yaml_file"],
                args,
                device,
                payload["case_dir"],
            )
            print(
                f"{case_name} val: mAP50-95={case_summary['val']['box_map']} "
                f"mAP50={case_summary['val']['box_map50']}",
                flush=True,
            )

        summary["cases"][case_name] = case_summary

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
