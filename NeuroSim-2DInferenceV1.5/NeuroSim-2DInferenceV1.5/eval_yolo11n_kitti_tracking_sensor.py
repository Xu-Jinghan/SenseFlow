import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detection_sensor_trace import CenterPixelWaveformRecorder


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
LOCAL_DATA_PATH = REPO_ROOT / ".datasets"
DEFAULT_PARAMS_CSV = REPO_ROOT / "data" / "case2" / "case2_fit_parameters.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "detection_runs" / "yolo11n_kitti_tracking_sensor_subset"
DEFAULT_CASE1_NOISE_SCALE_VS_CASE1 = 5.0
EVAL_CASES = ("clean", "ideal", "nonideal", "restored")
TARGET_CLASSES = ("car", "person")
YOLO_CLASS_IDS = {
    "person": {0},
    "car": {2},
}
KITTI_GT_TYPES = {
    "car": {"Car"},
    "person": {"Pedestrian", "Person_sitting"},
}
KITTI_IGNORE_TYPES = {
    "car": {"Van", "Truck", "Tram", "Misc", "DontCare"},
    "person": {"DontCare"},
}


def _default_data_path():
    return str(LOCAL_DATA_PATH if LOCAL_DATA_PATH.exists() else PROJECT_ROOT / ".datasets")


def _default_kitti_root():
    return str((LOCAL_DATA_PATH / "kitti_tracking") if LOCAL_DATA_PATH.exists() else PROJECT_ROOT / ".datasets" / "kitti_tracking")


def _default_model_path():
    local_model = LOCAL_DATA_PATH / "models" / "yolo11n.pt"
    if local_model.exists():
        return str(local_model)
    return "yolo11n.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "YOLO11n detection AP on consecutive KITTI Tracking training frames with "
            "clean / sensor-ideal / sensor-nonideal / restored inputs. "
            "This is software YOLO eval only."
        )
    )
    parser.add_argument("--kitti-root", default=_default_kitti_root())
    parser.add_argument("--model", default=_default_model_path())
    parser.add_argument("--sequence", default="auto", help="KITTI tracking training sequence id, or auto.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1000, help="Use 500 or 1000 consecutive frames.")
    parser.add_argument("--eval-cases", nargs="+", default=list(EVAL_CASES), choices=EVAL_CASES)
    parser.add_argument("--target-classes", nargs="+", default=list(TARGET_CLASSES), choices=TARGET_CLASSES)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=192)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predict-conf", type=float, default=0.001, help="Low threshold keeps detections for AP curves.")
    parser.add_argument(
        "--annotated-conf",
        type=float,
        default=0.25,
        help="Confidence threshold used only for saved annotated preview images.",
    )
    parser.add_argument("--iou-nms", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-annotated", type=int, default=1)
    parser.add_argument("--max-save-images", type=int, default=20)
    parser.add_argument("--restoration-mode", default="clahe", choices=["none", "percentile", "clahe", "learned"])
    parser.add_argument(
        "--restoration-model-path",
        default=None,
        help="Temporal restoration frontend checkpoint used when --restoration-mode learned.",
    )
    parser.add_argument("--restoration-history-frames", type=int, default=4)
    parser.add_argument("--restoration-hidden-channels", type=int, default=16)
    parser.add_argument("--restoration-num-blocks", type=int, default=3)
    parser.add_argument("--restoration-device", default="auto", help="Device for learned restoration: auto/cpu/cuda/cuda:0.")
    parser.add_argument("--ap-iou", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    # Sensor parameters mirror the video-sequence sensor pipeline.
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--readout", default="tia", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog-readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc-enabled", type=int, default=0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--adc-calibration-low", type=float, default=None)
    parser.add_argument("--adc-calibration-high", type=float, default=None)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--params-csv", default=None, help=f"Sensor parameter CSV. Defaults to {DEFAULT_PARAMS_CSV}.")
    parser.add_argument("--normalization-mode", default="physical", choices=["physical", "calibration", "per_frame", "none"])
    parser.add_argument("--range-mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument(
        "--range-calibration-samples",
        type=int,
        default=1024,
        help="Number of selected KITTI frames used to estimate fixed range bounds; 0 uses all selected frames.",
    )
    parser.add_argument(
        "--range-calibration-max-values-per-frame",
        type=int,
        default=20000,
        help="Maximum raw sensor values sampled from each calibration frame per case; 0 uses all pixels.",
    )
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
    parser.add_argument(
        "--noise-1f-density-1hz",
        type=float,
        default=None,
        help=(
            "Manual 1/f noise density at 1 Hz in A/Hz^0.5. If omitted, the script derives "
            "it from the case1 PSD using --case1-noise-scale-vs-case1."
        ),
    )
    parser.add_argument(
        "--case1-noise-scale-vs-case1",
        type=float,
        default=DEFAULT_CASE1_NOISE_SCALE_VS_CASE1,
        help=(
            "Scale factor applied to the measured case1 noise PSD before resampling the "
            "1 Hz 1/f density. Default 5.0 matches the case1Noise5x detection runs; set 0 "
            "to disable this automatic case1-derived noise density."
        ),
    )
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
            "If 1 and readout resolves to TIA, update each sample-and-hold frame in one "
            "closed-form frame step and use only the final-step noise sample. Default "
            "keeps the historical per-step trace path."
        ),
    )
    parser.add_argument(
        "--temporal-noise-mode",
        default="pixel_buffered",
        choices=["pixel_buffered", "pixel_repeated_window", "global_full_sequence", "global_repeated_window"],
        help=(
            "Temporal-noise synthesis mode. pixel_buffered keeps the historical per-pixel "
            "memory-bounded buffer. pixel_repeated_window synthesizes an independent "
            "per-pixel reusable trace. global_full_sequence synthesizes one full-duration "
            "1/f trace per channel and broadcasts it spatially. global_repeated_window "
            "synthesizes a shorter per-channel trace and reuses it cyclically."
        ),
    )
    parser.add_argument(
        "--temporal-noise-window-frames",
        type=int,
        default=10,
        help="Number of frames in the reusable temporal-noise window for repeated-window modes.",
    )
    parser.add_argument("--save-center-waveform", type=int, default=1)
    parser.add_argument("--center-waveform-channel", type=int, default=1)
    parser.add_argument("--center-waveform-max-frames", type=int, default=120)
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch. Use conda run -n xjhenv python ...") from exc
    return torch


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: ultralytics. Install it with pip install ultralytics.") from exc
    return YOLO


def import_sensor_pipeline():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import generate_sensor_verification_images as base_pipeline
    import generate_sensor_verification_images_video_sequence as video_pipeline

    return base_pipeline, video_pipeline


def resolve_device(device):
    device = str(device)
    if device.lower() != "auto":
        return device
    torch = import_torch()
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_torch_device(device):
    torch = import_torch()
    device = str(device)
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def frame_path_to_index(path):
    return int(path.stem)


def list_sequences(kitti_root):
    image_root = Path(kitti_root) / "training" / "image_02"
    if not image_root.is_dir():
        raise FileNotFoundError(f"KITTI tracking image_02 directory not found: {image_root}")
    return sorted([path.name for path in image_root.iterdir() if path.is_dir()])


def sequence_frame_paths(kitti_root, sequence):
    seq_dir = Path(kitti_root) / "training" / "image_02" / sequence
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"KITTI tracking sequence directory not found: {seq_dir}")
    frames = sorted(seq_dir.glob("*.png"), key=frame_path_to_index)
    if not frames:
        raise ValueError(f"No PNG frames found in {seq_dir}")
    return frames


def resolve_sequence(kitti_root, requested_sequence, num_frames, start_frame):
    sequences = list_sequences(kitti_root)
    if requested_sequence != "auto":
        return requested_sequence
    for sequence in sequences:
        frames = sequence_frame_paths(kitti_root, sequence)
        if len(frames) >= start_frame + num_frames:
            return sequence
    longest = max(sequences, key=lambda seq: len(sequence_frame_paths(kitti_root, seq)))
    return longest


def select_consecutive_frames(kitti_root, sequence, start_frame, num_frames):
    frames = sequence_frame_paths(kitti_root, sequence)
    if start_frame < 0:
        raise ValueError("start-frame must be non-negative")
    if start_frame + num_frames > len(frames):
        raise ValueError(
            f"Sequence {sequence} has {len(frames)} frames, cannot take "
            f"{num_frames} from start {start_frame}."
        )
    return frames[start_frame : start_frame + num_frames]


def parse_kitti_labels(label_path):
    labels_by_frame = {}
    if not Path(label_path).is_file():
        raise FileNotFoundError(f"KITTI tracking label file not found: {label_path}")
    with Path(label_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            frame_idx = int(parts[0])
            obj_type = parts[2]
            bbox = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
            labels_by_frame.setdefault(frame_idx, []).append({"type": obj_type, "bbox": bbox})
    return labels_by_frame


def scale_bbox_xyxy(box, scale_x, scale_y, width, height):
    x1, y1, x2, y2 = box
    return [
        float(np.clip(x1 * scale_x, 0.0, width)),
        float(np.clip(y1 * scale_y, 0.0, height)),
        float(np.clip(x2 * scale_x, 0.0, width)),
        float(np.clip(y2 * scale_y, 0.0, height)),
    ]


def prepare_ground_truth(frame_paths, labels_by_frame, output_width, output_height, target_classes):
    gt_by_class = {name: {} for name in target_classes}
    ignore_by_class = {name: {} for name in target_classes}
    image_meta = {}
    for image_path in frame_paths:
        frame_idx = frame_path_to_index(image_path)
        with Image.open(image_path) as img:
            src_w, src_h = img.size
        scale_x = output_width / float(src_w)
        scale_y = output_height / float(src_h)
        image_key = image_path.name
        image_meta[image_key] = {
            "frame_index": frame_idx,
            "source_path": str(image_path),
            "source_width": src_w,
            "source_height": src_h,
            "width": output_width,
            "height": output_height,
        }
        for class_name in target_classes:
            gt_by_class[class_name][image_key] = []
            ignore_by_class[class_name][image_key] = []
        for obj in labels_by_frame.get(frame_idx, []):
            scaled_box = scale_bbox_xyxy(obj["bbox"], scale_x, scale_y, output_width, output_height)
            if scaled_box[2] <= scaled_box[0] or scaled_box[3] <= scaled_box[1]:
                continue
            for class_name in target_classes:
                if obj["type"] in KITTI_GT_TYPES[class_name]:
                    gt_by_class[class_name][image_key].append(scaled_box)
                elif obj["type"] in KITTI_IGNORE_TYPES[class_name]:
                    ignore_by_class[class_name][image_key].append(scaled_box)
    return image_meta, gt_by_class, ignore_by_class


def make_sensor_args(args, base_pipeline, total_sequence_frames):
    return SimpleNamespace(
        params_csv=args.params_csv or str(DEFAULT_PARAMS_CSV),
        array_size=int(max(args.output_width, args.output_height)),
        output_channels=3,
        sensor_rng_seed=int(args.sensor_rng_seed),
        seed=1234,
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
        fast_tia_frame_step=int(getattr(args, "fast_tia_frame_step", 0)),
        temporal_noise_mode=getattr(args, "temporal_noise_mode", "pixel_buffered"),
        temporal_noise_window_frames=int(getattr(args, "temporal_noise_window_frames", 10)),
        i_thermal=0.0,
        bandwidth=5000.0,
        total_sequence_frames=int(total_sequence_frames),
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


def build_power_maps_rect(image, output_width, output_height, output_channels, base_params):
    resized = image.convert("RGB").resize((output_width, output_height), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float64) / 255.0
    if output_channels == 1:
        arr = np.mean(arr, axis=2, keepdims=True)
    arr = np.transpose(arr, (2, 0, 1))
    return base_params["prange1_w"] + arr * (base_params["prange2_w"] - base_params["prange1_w"])


def frame_to_rgb_array(unit_frame):
    frame = np.asarray(unit_frame, dtype=np.float32)
    if frame.ndim == 2:
        frame = frame[:, :, None]
    elif frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return np.clip(frame, 0.0, 1.0)


def rgb_array_to_image(array):
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")


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
    return base_pipeline.scaled_frame_to_unit_interval(
        scaled,
        sensor_args.readout,
        video_pipeline.effective_frame_range_mode(sensor_args),
    )


def effective_range_cases(eval_cases):
    range_cases = []
    if "ideal" in eval_cases:
        range_cases.append("ideal")
    if any(case in eval_cases for case in ("nonideal", "restored")):
        range_cases.append("nonideal")
    return range_cases


def sampled_values(frame, max_values, rng):
    values = np.asarray(frame, dtype=np.float32).reshape(-1)
    max_values = int(max_values)
    if max_values <= 0 or values.size <= max_values:
        return values
    indices = rng.choice(values.size, size=max_values, replace=False)
    return values[indices]


def resolve_range_calibration_sample_limit(args, num_frames):
    requested = int(getattr(args, "range_calibration_samples", 0))
    if requested > 0:
        return min(num_frames, requested)
    return num_frames


def compute_dataset_range_bounds(args, frame_paths, sensor_args, base_params, range_cases, base_pipeline, video_pipeline):
    if args.normalization_mode == "physical" and range_cases:
        return video_pipeline.compute_physical_case_range_bounds(sensor_args, base_params, range_cases)
    if args.normalization_mode != "calibration" or not range_cases:
        return {}
    if video_pipeline.effective_frame_range_mode(sensor_args) == "none":
        return {}

    sample_limit = resolve_range_calibration_sample_limit(args, len(frame_paths))
    rng = np.random.default_rng(int(args.sensor_rng_seed) + 1009)
    collected = {case_name: [] for case_name in range_cases}

    sequence_sensor = None
    if "nonideal" in range_cases:
        sequence_sensor = video_pipeline.StatefulNonidealVideoSensor(args=sensor_args, base_params=base_params)
        zero_power = np.zeros((3, args.output_height, args.output_width), dtype=np.float64)
        for _ in range(max(0, int(args.startup_dark_frames))):
            sequence_sensor.simulate_frame(zero_power)

    effective_mode = base_pipeline.resolve_effective_range_mode(
        sensor_args.readout,
        video_pipeline.effective_frame_range_mode(sensor_args),
    )
    print(
        f"Estimating fixed {effective_mode} range bounds from {sample_limit} KITTI frames "
        f"(cases={range_cases})",
        flush=True,
    )

    for idx, image_path in enumerate(frame_paths[:sample_limit]):
        source = Image.open(image_path).convert("RGB")
        power_maps = build_power_maps_rect(source, args.output_width, args.output_height, 3, base_params)

        if "ideal" in range_cases:
            ideal_raw = video_pipeline.simulate_ideal_video_frame(power_maps, sensor_args, base_params)
            collected["ideal"].append(
                sampled_values(ideal_raw, args.range_calibration_max_values_per_frame, rng)
            )

        if "nonideal" in range_cases:
            nonideal_raw = sequence_sensor.simulate_frame(power_maps)
            collected["nonideal"].append(
                sampled_values(nonideal_raw, args.range_calibration_max_values_per_frame, rng)
            )

        processed = idx + 1
        if processed % 100 == 0 or processed == sample_limit:
            print(f"  calibration progress: frames={processed}/{sample_limit}", flush=True)

    bounds = {}
    for case_name, arrays in collected.items():
        values = np.concatenate(arrays, axis=0)
        low = float(np.percentile(values, float(args.percentile_low)))
        high = float(np.percentile(values, float(args.percentile_high)))
        if high - low <= 1e-12:
            high = low + 1e-12
        bounds[case_name] = {
            "low": low,
            "high": high,
            "mode": effective_mode,
            "num_frames": int(sample_limit),
            "num_values": int(values.size),
            "max_values_per_frame": int(args.range_calibration_max_values_per_frame),
            "percentile_low": float(args.percentile_low),
            "percentile_high": float(args.percentile_high),
        }
        print(
            f"  fixed range {case_name}: low={low:.6e} high={high:.6e} span={high - low:.6e}",
            flush=True,
        )
    return bounds


def percentile_stretch(rgb, low=1.0, high=99.0):
    out = np.empty_like(rgb, dtype=np.float32)
    for channel_idx in range(rgb.shape[2]):
        channel = rgb[:, :, channel_idx]
        lo = float(np.percentile(channel, low))
        hi = float(np.percentile(channel, high))
        if hi <= lo + 1e-8:
            out[:, :, channel_idx] = channel
        else:
            out[:, :, channel_idx] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return out


def restore_rgb(rgb, mode):
    if mode == "none":
        return np.asarray(rgb, dtype=np.float32)
    stretched = percentile_stretch(rgb)
    if mode == "percentile":
        return stretched
    try:
        import cv2

        uint8 = (stretched * 255.0).round().astype(np.uint8)
        lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        restored_l = clahe.apply(l_channel)
        restored = cv2.merge((restored_l, a_channel, b_channel))
        restored = cv2.cvtColor(restored, cv2.COLOR_LAB2RGB)
        return restored.astype(np.float32) / 255.0
    except Exception:
        return stretched


class LearnedRestorationApplier:
    def __init__(self, model, torch_module, device, history_frames):
        self.model = model
        self.torch = torch_module
        self.device = device
        self.history_frames = int(history_frames)
        self.history = deque(maxlen=self.history_frames)

    def __call__(self, rgb):
        array = np.asarray(rgb, dtype=np.float32)
        tensor = self.torch.from_numpy(np.transpose(array, (2, 0, 1))).float()
        self.history.append(tensor)
        frames = list(self.history)
        if len(frames) < self.history_frames:
            frames = [frames[0]] * (self.history_frames - len(frames)) + frames
        batch = self.torch.stack(frames[-self.history_frames:], dim=0).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            restored = self.model(batch).squeeze(0).detach().cpu()
        return np.clip(np.transpose(restored.numpy(), (1, 2, 0)), 0.0, 1.0)


def load_learned_restoration(args):
    if args.restoration_mode != "learned":
        return None, None
    if not args.restoration_model_path:
        raise ValueError("--restoration-model-path is required when --restoration-mode learned")
    if int(args.restoration_history_frames) <= 0:
        raise ValueError("--restoration-history-frames must be positive")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.restoration import TemporalTinyRestorationCNN

    torch = import_torch()
    device = resolve_torch_device(args.restoration_device)
    model = TemporalTinyRestorationCNN(
        in_channels=3,
        hidden_channels=int(args.restoration_hidden_channels),
        num_blocks=int(args.restoration_num_blocks),
        history_frames=int(args.restoration_history_frames),
    ).to(device)
    checkpoint_path = Path(args.restoration_model_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "restoration_model" in checkpoint:
        checkpoint = checkpoint["restoration_model"]
    model.load_state_dict(checkpoint)
    model.eval()
    summary = {
        "mode": "learned",
        "model_path": str(checkpoint_path),
        "history_frames": int(args.restoration_history_frames),
        "hidden_channels": int(args.restoration_hidden_channels),
        "num_blocks": int(args.restoration_num_blocks),
        "device": str(device),
    }
    return LearnedRestorationApplier(model, torch, device, args.restoration_history_frames), summary


def prepare_case_images(args, frame_paths, output_dir):
    base_pipeline, video_pipeline = import_sensor_pipeline()
    learned_restoration, learned_restoration_summary = load_learned_restoration(args)
    sensor_args = make_sensor_args(
        args,
        base_pipeline,
        total_sequence_frames=len(frame_paths) + max(0, int(args.startup_dark_frames)),
    )
    if sensor_args.analog_readout is None:
        sensor_args.analog_readout = video_pipeline.resolve_analog_readout_mode(sensor_args)
    sensor_args.frame_range_mode_override = video_pipeline.effective_frame_range_mode(sensor_args)

    base_params = video_pipeline.resolve_sequence_base_params(sensor_args)
    base_params = apply_detection_x2_overrides(args, base_params)
    if sensor_args.tia_gain_ohm is None:
        sensor_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if sensor_args.integration_gain_v_per_c is None:
        sensor_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    noise_trace_summary = summarize_actual_noise_trace(sensor_args, base_params)

    range_cases = effective_range_cases(args.eval_cases)
    case_range_bounds = compute_dataset_range_bounds(
        args,
        frame_paths,
        sensor_args,
        base_params,
        range_cases,
        base_pipeline,
        video_pipeline,
    )

    sequence_sensor = None
    center_trace_recorder = CenterPixelWaveformRecorder(
        enabled=(any(case in args.eval_cases for case in ("nonideal", "restored")) and bool(args.save_center_waveform)),
        output_dir=output_dir / "center_pixel_waveform",
        target_channel=args.center_waveform_channel,
        max_plot_frames=args.center_waveform_max_frames,
        title="KITTI Tracking center pixel case2 nonideal waveform",
    )
    if any(case in args.eval_cases for case in ("nonideal", "restored")):
        sequence_sensor = video_pipeline.StatefulNonidealVideoSensor(args=sensor_args, base_params=base_params)
        zero_power = np.zeros((3, args.output_height, args.output_width), dtype=np.float64)
        for _ in range(max(0, int(args.startup_dark_frames))):
            sequence_sensor.simulate_frame(zero_power)

    case_payloads = {}
    for case_name in args.eval_cases:
        case_dir = output_dir / case_name
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        case_payloads[case_name] = {"case_dir": case_dir, "image_dir": image_dir, "image_paths": []}

    started = time.time()
    for idx, image_path in enumerate(frame_paths):
        source = Image.open(image_path).convert("RGB")
        generated = {}
        if "clean" in args.eval_cases:
            generated["clean"] = source.resize((args.output_width, args.output_height), resample=Image.Resampling.BILINEAR)

        power_maps = None
        if any(case in args.eval_cases for case in ("ideal", "nonideal", "restored")):
            power_maps = build_power_maps_rect(source, args.output_width, args.output_height, 3, base_params)

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
            generated["ideal"] = rgb_array_to_image(frame_to_rgb_array(ideal_unit))

        if any(case in args.eval_cases for case in ("nonideal", "restored")):
            if center_trace_recorder.enabled:
                nonideal_raw, center_trace = sequence_sensor.simulate_frame(
                    power_maps,
                    record_center_trace=True,
                )
                center_trace_recorder.record(
                    idx,
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
            nonideal_rgb = frame_to_rgb_array(nonideal_unit)
            if "nonideal" in args.eval_cases:
                generated["nonideal"] = rgb_array_to_image(nonideal_rgb)
            if "restored" in args.eval_cases:
                if learned_restoration is not None:
                    restored_rgb = learned_restoration(nonideal_rgb)
                else:
                    restored_rgb = restore_rgb(nonideal_rgb, args.restoration_mode)
                generated["restored"] = rgb_array_to_image(restored_rgb)

        for case_name, image in generated.items():
            output_path = case_payloads[case_name]["image_dir"] / image_path.name
            image.save(output_path)
            case_payloads[case_name]["image_paths"].append(output_path)

        processed = idx + 1
        if processed % 100 == 0 or processed == len(frame_paths):
            print(f"  image export progress: {processed}/{len(frame_paths)}", flush=True)

    center_trace_summary = center_trace_recorder.finalize(sensor_args, base_params)
    return {
        "elapsed_s": time.time() - started,
        "sensor_args": vars(sensor_args),
        "base_params": {
            key: (None if value is None else float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in base_params.items()
            if key not in {"spatial_variation_cache_dir"}
        },
        "case_range_bounds": case_range_bounds,
        "noise_trace": noise_trace_summary,
        "case_payloads": case_payloads,
        "center_pixel_waveform": center_trace_summary,
        "learned_restoration": learned_restoration_summary,
    }


def result_boxes_to_predictions(result, target_classes, image_name):
    boxes = result.boxes
    predictions = {class_name: [] for class_name in target_classes}
    if boxes is None or len(boxes) == 0:
        return predictions
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    for box, score, class_id in zip(xyxy, conf, cls):
        for class_name in target_classes:
            if int(class_id) in YOLO_CLASS_IDS[class_name]:
                predictions[class_name].append(
                    {
                        "image_name": image_name,
                        "confidence": float(score),
                        "bbox_xyxy": [float(v) for v in box],
                    }
                )
    return predictions


def save_annotated_preview(image_path, predictions_by_class, annotated_path, conf_threshold):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = {"car": (255, 80, 80), "person": (80, 180, 255)}
    for class_name, predictions in predictions_by_class.items():
        color = colors.get(class_name, (255, 220, 80))
        for pred in predictions:
            if float(pred["confidence"]) < conf_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in pred["bbox_xyxy"]]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            label = f"{class_name} {float(pred['confidence']):.2f}"
            if font is not None:
                text_box = draw.textbbox((x1, y1), label, font=font)
                text_h = text_box[3] - text_box[1]
                label_y = max(0.0, y1 - text_h - 2)
                draw.rectangle((x1, label_y, text_box[2] + 4, label_y + text_h + 2), fill=color)
                draw.text((x1 + 2, label_y + 1), label, fill=(0, 0, 0), font=font)
    image.save(annotated_path)


def run_predict(model, image_paths, args, device, case_dir):
    predict_dir = case_dir / "predict"
    annotated_dir = predict_dir / "annotated"
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    predict_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {class_name: [] for class_name in args.target_classes}
    per_image = []
    preview_predictions = []
    preview_detection_count = 0
    started = time.time()
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device,
        conf=args.predict_conf,
        iou=args.iou_nms,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )
    for idx, result in enumerate(results):
        image_name = image_paths[idx].name
        pred_by_class = result_boxes_to_predictions(result, args.target_classes, image_name)
        for class_name, preds in pred_by_class.items():
            all_predictions[class_name].extend(preds)
        per_image.append({"image_name": image_name, "predictions": pred_by_class})
        preview_by_class = {
            class_name: [
                pred for pred in preds
                if float(pred["confidence"]) >= float(args.annotated_conf)
            ]
            for class_name, preds in pred_by_class.items()
        }
        preview_detection_count += sum(len(preds) for preds in preview_by_class.values())
        preview_predictions.append({"image_name": image_name, "predictions": preview_by_class})
        if args.save_annotated and idx < max(0, args.max_save_images):
            save_annotated_preview(
                image_paths[idx],
                pred_by_class,
                annotated_dir / image_name,
                float(args.annotated_conf),
            )

    predictions_path = predict_dir / "predictions.json"
    predictions_path.write_text(json.dumps(per_image, indent=2), encoding="utf-8")
    preview_predictions_path = predict_dir / f"preview_predictions_conf{float(args.annotated_conf):.2f}.json"
    preview_predictions_path.write_text(json.dumps(preview_predictions, indent=2), encoding="utf-8")
    elapsed_s = time.time() - started
    return {
        "predictions": all_predictions,
        "predictions_path": str(predictions_path),
        "preview_predictions_path": str(preview_predictions_path),
        "annotated_dir": str(annotated_dir) if args.save_annotated else None,
        "elapsed_s": elapsed_s,
        "num_images": len(per_image),
        "num_detections": int(sum(len(v) for v in all_predictions.values())),
        "num_preview_detections": int(preview_detection_count),
        "ap_conf": float(args.predict_conf),
        "annotated_conf": float(args.annotated_conf),
    }


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    denom = np.maximum(box_area + boxes_area - inter, 1e-12)
    return inter / denom


def compute_ap_from_pr(precision, recall):
    thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for threshold in thresholds:
        valid = precision[recall >= threshold]
        ap += float(np.max(valid)) if valid.size else 0.0
    return ap / len(thresholds)


def evaluate_class_ap(predictions, gt_by_image, ignore_by_image, iou_threshold):
    total_gt = int(sum(len(boxes) for boxes in gt_by_image.values()))
    if total_gt == 0:
        return {"ap": None, "num_gt": 0, "num_predictions": len(predictions)}

    sorted_preds = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    matched = {image_name: np.zeros(len(boxes), dtype=bool) for image_name, boxes in gt_by_image.items()}
    tp = []
    fp = []

    for pred in sorted_preds:
        image_name = pred["image_name"]
        pred_box = pred["bbox_xyxy"]
        gt_boxes = gt_by_image.get(image_name, [])
        ignored_boxes = ignore_by_image.get(image_name, [])

        match_idx = -1
        match_iou = 0.0
        ious = box_iou(pred_box, gt_boxes)
        if ious.size:
            order = np.argsort(-ious)
            for idx in order:
                if ious[idx] >= iou_threshold and not matched[image_name][idx]:
                    match_idx = int(idx)
                    match_iou = float(ious[idx])
                    break

        if match_idx >= 0:
            matched[image_name][match_idx] = True
            tp.append(1.0)
            fp.append(0.0)
            continue

        ignore_ious = box_iou(pred_box, ignored_boxes)
        if ignore_ious.size and float(np.max(ignore_ious)) >= iou_threshold:
            continue

        tp.append(0.0)
        fp.append(1.0)

    if not tp:
        return {
            "ap": 0.0,
            "num_gt": total_gt,
            "num_predictions": len(predictions),
            "precision": 0.0,
            "recall": 0.0,
        }

    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return {
        "ap": float(compute_ap_from_pr(precision, recall)),
        "num_gt": total_gt,
        "num_predictions": len(predictions),
        "precision": float(precision[-1]) if precision.size else 0.0,
        "recall": float(recall[-1]) if recall.size else 0.0,
    }


def evaluate_predictions(predictions_by_class, gt_by_class, ignore_by_class, target_classes):
    iou_thresholds = [round(v, 2) for v in np.arange(0.5, 0.96, 0.05)]
    metrics = {}
    for class_name in target_classes:
        class_metrics = {}
        ap_values = []
        for threshold in iou_thresholds:
            result = evaluate_class_ap(
                predictions_by_class[class_name],
                gt_by_class[class_name],
                ignore_by_class[class_name],
                threshold,
            )
            class_metrics[f"AP{int(threshold * 100):02d}"] = result["ap"]
            if result["ap"] is not None:
                ap_values.append(result["ap"])
            if threshold == 0.5:
                class_metrics["num_gt"] = result["num_gt"]
                class_metrics["num_predictions"] = result["num_predictions"]
                class_metrics["precision_at_conf_sweep_end"] = result.get("precision")
                class_metrics["recall_at_conf_sweep_end"] = result.get("recall")
        class_metrics["AP50_95"] = float(np.mean(ap_values)) if ap_values else None
        metrics[class_name] = class_metrics
    valid_ap50 = [metrics[class_name]["AP50"] for class_name in target_classes if metrics[class_name]["AP50"] is not None]
    valid_map = [metrics[class_name]["AP50_95"] for class_name in target_classes if metrics[class_name]["AP50_95"] is not None]
    metrics["mean"] = {
        "AP50": float(np.mean(valid_ap50)) if valid_ap50 else None,
        "AP50_95": float(np.mean(valid_map)) if valid_map else None,
    }
    return metrics


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


def derive_case1_noise_density_1hz(scale_vs_case1):
    scale_vs_case1 = float(scale_vs_case1)
    if scale_vs_case1 <= 0.0:
        return None

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import photodetector_model as pm

    case1_results = pm.extract_case1_single_carrier_params(
        pm.CASE1_DATA_DIR,
        device_area_cm2=pm.DEVICE_AREA_CM2,
    )
    freqs_hz = case1_results["dataset"]["noise_freq_hz"]
    noise_density = case1_results["dataset"]["noise_density"]
    scaled_density = pm.scale_noise_density_components(
        freqs_hz,
        noise_density,
        white_scale=scale_vs_case1,
        flicker_scale=scale_vs_case1,
    )
    density_1hz = float(
        pm._resample_noise_density(
            np.asarray([1.0], dtype=np.float64),
            freqs_hz,
            scaled_density,
        )[0]
    )
    base_density_1hz = float(
        pm._resample_noise_density(
            np.asarray([1.0], dtype=np.float64),
            freqs_hz,
            noise_density,
        )[0]
    )
    return {
        "source": "case1_psd_scaled",
        "scale_vs_case1": scale_vs_case1,
        "base_density_1hz_a_root_hz": base_density_1hz,
        "density_1hz_a_root_hz": density_1hz,
        "case1_data_dir": str(pm.CASE1_DATA_DIR),
    }


def resolve_detection_noise_config(args):
    manual_density = args.noise_1f_density_1hz
    if manual_density is not None:
        density = float(manual_density)
        args.noise_1f_density_1hz = density
        if density > 0.0:
            args.use_noise_fn = 1
        return {
            "source": "manual_noise_1f_density_1hz",
            "scale_vs_case1": None,
            "density_1hz_a_root_hz": density,
            "use_noise_fn": int(args.use_noise_fn),
        }

    case1_noise = derive_case1_noise_density_1hz(args.case1_noise_scale_vs_case1)
    if case1_noise is None:
        return {
            "source": "disabled",
            "scale_vs_case1": float(args.case1_noise_scale_vs_case1),
            "density_1hz_a_root_hz": None,
            "use_noise_fn": int(args.use_noise_fn),
        }

    args.noise_1f_density_1hz = float(case1_noise["density_1hz_a_root_hz"])
    args.use_noise_fn = 1
    return {
        **case1_noise,
        "use_noise_fn": int(args.use_noise_fn),
    }


def summarize_actual_noise_trace(sensor_args, base_params):
    flicker_density = float(base_params.get("noise_1f_density_1hz_a_root_hz", 0.0))
    shot_requested = bool(getattr(sensor_args, "shot_noise", 0))
    use_noise_fn = bool(getattr(sensor_args, "use_noise_fn", 0))
    temporal_trace_enabled = use_noise_fn and (flicker_density > 0.0 or shot_requested)
    return {
        "temporal_trace_enabled": temporal_trace_enabled,
        "use_noise_fn": int(use_noise_fn),
        "shot_noise": int(shot_requested),
        "flicker_density_1hz_a_root_hz": flicker_density,
        "temporal_noise_mode": str(getattr(sensor_args, "temporal_noise_mode", "pixel_buffered")),
        "temporal_noise_window_frames": int(getattr(sensor_args, "temporal_noise_window_frames", 10)),
        "reason": (
            "nonzero_flicker_or_shot_density"
            if temporal_trace_enabled
            else "disabled_or_zero_noise_density"
        ),
    }


def main():
    args = parse_args()
    requested_noise_config = resolve_detection_noise_config(args)
    kitti_root = Path(args.kitti_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = resolve_sequence(kitti_root, args.sequence, args.num_frames, args.start_frame)
    frame_paths = select_consecutive_frames(kitti_root, sequence, args.start_frame, args.num_frames)
    label_path = kitti_root / "training" / "label_02" / f"{sequence}.txt"
    labels_by_frame = parse_kitti_labels(label_path)
    image_meta, gt_by_class, ignore_by_class = prepare_ground_truth(
        frame_paths,
        labels_by_frame,
        args.output_width,
        args.output_height,
        args.target_classes,
    )

    YOLO = import_yolo()
    device = resolve_device(args.device)
    model = YOLO(args.model)

    print("=" * 72, flush=True)
    print("KITTI Tracking sensor + YOLO11n detection AP", flush=True)
    print("Pipeline: sensor / model eval only; CIM/NeuroSim disabled", flush=True)
    print(f"KITTI root: {kitti_root}", flush=True)
    print(f"Sequence: {sequence} start={args.start_frame} frames={len(frame_paths)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Cases: {args.eval_cases}", flush=True)
    print(f"Output size: {args.output_width}x{args.output_height}", flush=True)
    print(f"Sensor params: {args.params_csv or DEFAULT_PARAMS_CSV}", flush=True)
    print(
        f"Sensor readout: readout={args.readout} analog_readout={args.analog_readout or 'auto'} "
        f"adc_enabled={args.adc_enabled} adc_bits={args.adc_bits}",
        flush=True,
    )
    print(
        f"Sensor nonideal extras: shot_noise={args.shot_noise} use_noise_fn={args.use_noise_fn} "
        f"flicker_1Hz={args.noise_1f_density_1hz} spatial_variation_r_pct={args.spatial_variation_r_pct}",
        flush=True,
    )
    print(
        f"Detection noise source: {requested_noise_config['source']} "
        f"scale_vs_case1={requested_noise_config.get('scale_vs_case1')} "
        f"density_1Hz={requested_noise_config.get('density_1hz_a_root_hz')}",
        flush=True,
    )
    print(
        f"Sensor x2 mode: trap_mode={args.x2_trap_mode} "
        f"trap_output_mode={args.x2_trap_output_mode} alpha={args.x2_alpha or 'case2_csv'} beta={args.x2_beta}",
        flush=True,
    )
    print(f"Center waveform export: {bool(args.save_center_waveform)}", flush=True)

    prep = prepare_case_images(args, frame_paths, output_dir)
    print(
        f"Actual temporal noise trace: enabled={prep['noise_trace']['temporal_trace_enabled']} "
        f"flicker_1Hz={prep['noise_trace']['flicker_density_1hz_a_root_hz']:.6e} A/Hz^0.5",
        flush=True,
    )
    summary = {
        "pipeline": "sensor / model eval",
        "cim_neurosim": "disabled",
        "task": "object_detection",
        "dataset": "KITTI Tracking training",
        "sequence": sequence,
        "start_frame": args.start_frame,
        "num_frames": len(frame_paths),
        "target_classes": args.target_classes,
        "model": args.model,
        "device": device,
        "output_size": [args.output_width, args.output_height],
        "restoration_mode": args.restoration_mode,
        "learned_restoration": prep["learned_restoration"],
        "args": vars(args),
        "noise_config": requested_noise_config,
        "sensor": {
            "elapsed_s": prep["elapsed_s"],
            "args": prep["sensor_args"],
            "base_params": prep["base_params"],
            "case_range_bounds": prep["case_range_bounds"],
            "noise_trace": prep["noise_trace"],
            "center_pixel_waveform": prep["center_pixel_waveform"],
        },
        "ground_truth": {
            class_name: {
                "num_gt": int(sum(len(v) for v in gt_by_class[class_name].values())),
                "num_ignore": int(sum(len(v) for v in ignore_by_class[class_name].values())),
            }
            for class_name in args.target_classes
        },
        "cases": {},
    }

    image_meta_path = output_dir / "image_meta.json"
    image_meta_path.write_text(json.dumps(json_safe(image_meta), indent=2), encoding="utf-8")
    summary["image_meta_path"] = str(image_meta_path)

    for case_name in args.eval_cases:
        payload = prep["case_payloads"][case_name]
        predict_result = run_predict(model, payload["image_paths"], args, device, payload["case_dir"])
        metrics = evaluate_predictions(
            predict_result["predictions"],
            gt_by_class,
            ignore_by_class,
            args.target_classes,
        )
        metrics_path = payload["case_dir"] / "ap_metrics.json"
        metrics_path.write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
        summary["cases"][case_name] = {
            "case_dir": str(payload["case_dir"]),
            "image_dir": str(payload["image_dir"]),
            "num_images": len(payload["image_paths"]),
            "predict": {
                key: value
                for key, value in predict_result.items()
                if key != "predictions"
            },
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        }
        print(
            f"{case_name}: mAP50-95={metrics['mean']['AP50_95']} "
            f"mAP50={metrics['mean']['AP50']} detections={predict_result['num_detections']}",
            flush=True,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
