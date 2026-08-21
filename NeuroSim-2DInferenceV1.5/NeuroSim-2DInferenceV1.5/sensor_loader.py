import os
import sys
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
for import_root in (THIS_DIR, WORKSPACE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import generate_sensor_verification_images as static_sensor_pipeline  # noqa: E402
import generate_sensor_verification_images_video_sequence as video_sequence_pipeline  # noqa: E402
from photodetector_array import (  # noqa: E402
    PhotodetectorArray,
    ReadoutADC,
    ReadoutIntegration,
    ReadoutTIA,
)
from sensor_video_sequence_backend import (  # noqa: E402
    StatefulNonidealVideoSensor,
    simulate_ideal_video_frame,
)


_POST_NORM_STATS = {
    "cifar10": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar100": (
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    ),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _as_bool(value):
    return bool(int(value)) if isinstance(value, (int, np.integer, bool)) else bool(value)


def _resolve_optional_local_path(path_str):
    if not path_str:
        return path_str

    expanded = Path(path_str).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    if expanded.exists():
        return str(expanded)

    for base in (THIS_DIR, WORKSPACE_ROOT):
        candidate = base / expanded
        if candidate.exists():
            return str(candidate)
    return str(expanded)


def _infer_target_size(model, source_dataset):
    if model == "cimae":
        return 96
    if model == "swin_t":
        return 256
    if source_dataset in {"cifar10", "cifar100"}:
        return 32
    return 224


def _infer_output_channels(model):
    if model == "cimae":
        return 1
    return 3


def _build_sensor_base_dataset(source_dataset, data_root, train):
    data_root = os.path.expanduser(data_root)

    if source_dataset == "cifar10":
        data_root = os.path.join(data_root, "cifar10-data")
        return datasets.CIFAR10(root=data_root, train=train, download=True, transform=None)

    if source_dataset == "cifar100":
        data_root = os.path.join(data_root, "cifar100-data")
        return datasets.CIFAR100(root=data_root, train=train, download=True, transform=None)

    if source_dataset == "imagenet":
        split_path = os.path.join(data_root, "val")
        if train:
            print(f"Warning: Using val set as train set from {split_path}, as download is too slow.")
        return datasets.ImageFolder(split_path, transform=None)

    raise ValueError(f"Unsupported sensor source dataset: {source_dataset}")


def _load_sensor_params_csv(csv_path):
    params = {}
    with open(_resolve_optional_local_path(csv_path), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sensor params csv is empty: {csv_path}")
        required = {"parameter", "value"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Sensor params csv {csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            key = row["parameter"].strip()
            if key in PhotodetectorArray.DEFAULT_PARAMS:
                params[key] = float(row["value"])
    if not params:
        raise ValueError(
            f"Sensor params csv {csv_path} did not contain any photodetector_model parameters. "
            f"Expected keys like {sorted(PhotodetectorArray.DEFAULT_PARAMS.keys())}."
        )
    return params


def _load_model_pipeline_params(csv_path):
    if csv_path:
        return static_sensor_pipeline.resolve_base_params(_resolve_optional_local_path(csv_path))
    return static_sensor_pipeline.resolve_base_params(str(static_sensor_pipeline.DEFAULT_PARAMS_CSV))


def _resolve_sensor_video_fps(args):
    fps = float(getattr(args, "sensor_video_fps", 0.0))
    if fps > 0:
        return fps

    exposure_time = float(getattr(args, "sensor_exposure_time", 0.0))
    if exposure_time <= 0:
        raise ValueError(
            "sensor_video_fps must be positive, or sensor_exposure_time must be positive when sensor_video_fps=0."
        )
    return 1.0 / exposure_time


def _apply_model_pipeline_overrides(base_params, args):
    params = dict(base_params)
    spatial_variation_r_pct = getattr(args, "sensor_spatial_variation_r_pct", None)
    if spatial_variation_r_pct is not None:
        params["spatial_variation_r_pct"] = float(spatial_variation_r_pct)
        params["spatial_variation_r_ratio"] = max(0.0, float(spatial_variation_r_pct) / 100.0)
    return params


class _BaseSensorReadoutDataset(Dataset):
    requires_sequential_access = False

    def __init__(self, base_dataset, args, model, source_dataset):
        self.base_dataset = base_dataset
        self.args = args
        self.model = model
        self.source_dataset = source_dataset
        self.sensor_nonideal = _as_bool(getattr(args, "sensor_nonideal", 1))

        self.target_size = args.sensor_target_size or _infer_target_size(model, source_dataset)
        self.array_size = args.sensor_array_size or self.target_size
        self.output_channels = args.sensor_output_channels or _infer_output_channels(model)

        self.exposure_time = args.sensor_exposure_time
        self.fps_sim = args.sensor_fps_sim
        self.n_steps = max(1, int(round(self.exposure_time * self.fps_sim)))
        self.dt = self.exposure_time / self.n_steps

        self.post_norm = args.sensor_post_norm
        self.range_mode = args.sensor_range_mode
        self.percentile_low = args.sensor_percentile_low
        self.percentile_high = args.sensor_percentile_high
        self.model_pipeline_params = _apply_model_pipeline_overrides(
            _load_model_pipeline_params(args.sensor_params_csv),
            args,
        )

    def __len__(self):
        return len(self.base_dataset)

    def _to_power_maps_static(self, image):
        image_tensor = TF.to_tensor(image)
        resized = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.array_size, self.array_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if resized.shape[0] == 1:
            resized = resized[:1]
        elif self.output_channels == 1:
            resized = TF.rgb_to_grayscale(resized, num_output_channels=1)
        elif resized.shape[0] > self.output_channels:
            resized = resized[:self.output_channels]

        power_maps = resized.cpu().numpy().astype(np.float64)
        power_maps *= self.args.sensor_power_max
        return power_maps

    def _to_power_maps_model_pipeline(self, image):
        return static_sensor_pipeline.build_power_maps(
            image,
            self.array_size,
            self.output_channels,
            self.args.sensor_power_max,
        )

    def _scale_frame(self, frame):
        return static_sensor_pipeline.scale_frame(
            frame,
            self.args.sensor_readout,
            self.range_mode,
            self.percentile_low,
            self.percentile_high,
        )

    def _apply_post_norm(self, tensor):
        if self.post_norm == "none":
            return tensor

        if self.post_norm == "auto":
            norm_key = self.source_dataset
        else:
            norm_key = self.post_norm

        if norm_key not in _POST_NORM_STATS:
            raise ValueError(f"Unsupported sensor post norm: {self.post_norm}")

        mean, std = _POST_NORM_STATS[norm_key]
        if tensor.shape[0] == 1:
            mean = (mean[0],)
            std = (std[0],)

        return TF.normalize(tensor, mean=mean, std=std)

    def _frame_to_tensor(self, sensor_frame):
        sensor_frame = static_sensor_pipeline.scaled_frame_to_unit_interval(
            sensor_frame,
            self.args.sensor_readout,
            self.range_mode,
        )
        tensor = torch.from_numpy(sensor_frame).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.output_channels == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        return self._apply_post_norm(tensor)

    def _simulate_sensor_frame(self, index, image):
        raise NotImplementedError

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        sensor_frame = self._simulate_sensor_frame(index, image)
        sensor_frame = self._scale_frame(sensor_frame)
        return self._frame_to_tensor(sensor_frame), label


class StaticSensorReadoutDataset(_BaseSensorReadoutDataset):
    def __init__(self, base_dataset, args, model, source_dataset):
        super().__init__(base_dataset, args, model, source_dataset)

        variation_params = {
            "responsivity_cv": args.sensor_pixel_var_resp,
            "eta_sigma": args.sensor_pixel_var_eta,
            "tau_cv": args.sensor_pixel_var_tau,
            "dark_current_cv": args.sensor_pixel_var_dark,
            "thermal_noise_cv": args.sensor_pixel_var_noise,
        }
        noise_params = {
            "i_thermal": args.sensor_i_thermal,
            "bandwidth": args.sensor_bandwidth,
            "shot_noise": bool(args.sensor_shot_noise),
        }
        base_params = None
        if args.sensor_params_csv:
            base_params = _load_sensor_params_csv(args.sensor_params_csv)

        self.array = PhotodetectorArray(
            H=self.array_size,
            W=self.array_size,
            params=base_params,
            noise_params=noise_params,
            variation_params=variation_params,
            rng_seed=args.sensor_rng_seed,
        )
        self.readout = self._build_readout(args.sensor_readout)
        self.static_model_args = SimpleNamespace(
            exposure_time=self.exposure_time,
            fps_sim=self.fps_sim,
            readout=self.args.sensor_readout,
            adc_bits=self.args.sensor_adc_bits,
            adc_full_scale=self.args.sensor_adc_full_scale,
            sensor_rng_seed=self.args.sensor_rng_seed,
            seed=getattr(self.args, "seed", 1234),
        )

    def _build_readout(self, mode):
        if mode == "tia":
            return ReadoutTIA(self.array)
        if mode == "integration":
            return ReadoutIntegration(self.array)
        if mode == "adc":
            return ReadoutADC(
                self.array,
                n_bits=self.args.sensor_adc_bits,
                full_scale=self.args.sensor_adc_full_scale,
            )
        raise ValueError(f"Unsupported sensor readout: {mode}")

    def _simulate_sensor_channel(self, power_map):
        self.array.reset()

        if hasattr(self.readout, "reset_accumulator"):
            self.readout.reset_accumulator()

        if isinstance(self.readout, ReadoutADC) and self.args.sensor_adc_full_scale is None:
            self.readout.full_scale = None

        for _ in range(self.n_steps):
            self.array.step(power_map, self.dt)
            if hasattr(self.readout, "accumulate"):
                self.readout.accumulate(self.dt)

        return self.readout.read_frame().astype(np.float32)

    def _simulate_sensor_nonideal(self, power_maps):
        if power_maps.ndim == 2:
            return self._simulate_sensor_channel(power_maps)

        frames = [self._simulate_sensor_channel(channel_map) for channel_map in power_maps]
        return np.stack(frames, axis=0)

    def _simulate_sensor_ideal(self, index, image):
        power_maps = self._to_power_maps_model_pipeline(image)
        return static_sensor_pipeline.simulate_static_frame(
            power_maps,
            self.static_model_args,
            self.model_pipeline_params,
            nonideal=False,
            seed_offset=index * 8,
        )

    def _simulate_sensor_frame(self, index, image):
        if not self.sensor_nonideal:
            return self._simulate_sensor_ideal(index, image)
        return self._simulate_sensor_nonideal(self._to_power_maps_static(image))


class VideoSequenceSensorReadoutDataset(_BaseSensorReadoutDataset):
    requires_sequential_access = True

    def __init__(self, base_dataset, args, model, source_dataset):
        if source_dataset not in {"cifar10", "cifar100"}:
            raise ValueError(
                "sensor_backend=video_sequence currently supports only cifar10 and cifar100."
            )

        super().__init__(base_dataset, args, model, source_dataset)
        self.sequence_args = SimpleNamespace(
            video_fps=_resolve_sensor_video_fps(args),
            fps_sim=self.args.sensor_fps_sim,
            sensor_rng_seed=self.args.sensor_rng_seed,
            seed=getattr(self.args, "seed", 1234),
            use_noise_fn=getattr(self.args, "sensor_use_noise_fn", 1),
            shot_noise=self.args.sensor_shot_noise,
            i_thermal=self.args.sensor_i_thermal,
            bandwidth=self.args.sensor_bandwidth,
            readout=self.args.sensor_readout,
            analog_readout=self.args.sensor_readout if self.args.sensor_readout in {"tia", "integration"} else "integration",
            adc_enabled=1 if self.args.sensor_readout == "adc" else 0,
            adc_bits=self.args.sensor_adc_bits,
            adc_full_scale=self.args.sensor_adc_full_scale,
            adc_calibration_low=None,
            adc_calibration_high=None,
            data_root=self.args.data_path,
            source_dataset=self.source_dataset,
            split="test",
            array_size=self.array_size,
            output_channels=self.output_channels,
            target_size=self.target_size,
            num_workers=0,
            post_norm=self.post_norm,
            normalization_mode="calibration",
            range_mode=self.range_mode,
            range_scope="calibration",
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            range_calibration_split="train",
            range_calibration_samples=100,
            startup_dark_frames=max(0, int(getattr(self.args, "sensor_startup_dark_frames", 0))),
        )
        self.case_range_bounds = self._compute_case_range_bounds()
        self._sequence_sensor = None
        self._sequence_cursor = 0
        self._zero_power = None
        self._reset_sequence_state()

    def _compute_case_range_bounds(self):
        calibration_dataset = _build_sensor_base_dataset(self.source_dataset, self.args.data_path, train=True)
        return video_sequence_pipeline.compute_case_range_bounds(
            args=self.sequence_args,
            calibration_dataset=calibration_dataset,
            base_params=self.model_pipeline_params,
            cases=video_sequence_pipeline.RANGE_CASES,
        )

    def _to_power_maps_model_pipeline(self, image):
        return video_sequence_pipeline.build_sequence_power_maps(
            image,
            self.sequence_args,
            self.model_pipeline_params,
        )

    def _get_zero_power_frame(self):
        if self._zero_power is None:
            if len(self.base_dataset) == 0:
                raise ValueError("Video-sequence sensor dataset is empty.")
            first_image, _ = self.base_dataset[0]
            self._zero_power = np.zeros_like(self._to_power_maps_model_pipeline(first_image), dtype=np.float64)
        return self._zero_power

    def _reset_sequence_state(self):
        self._sequence_cursor = 0
        self._sequence_sensor = None

        if not self.sensor_nonideal:
            return

        self._sequence_sensor = StatefulNonidealVideoSensor(
            args=self.sequence_args,
            base_params=self.model_pipeline_params,
        )
        startup_dark_frames = max(0, int(getattr(self.args, "sensor_startup_dark_frames", 0)))
        if startup_dark_frames <= 0:
            return

        zero_power = self._get_zero_power_frame()
        for _ in range(startup_dark_frames):
            self._sequence_sensor.simulate_frame(zero_power)

    def _simulate_sequence_frame(self, image):
        power_maps = self._to_power_maps_model_pipeline(image)
        if self.sensor_nonideal:
            return self._sequence_sensor.simulate_frame(power_maps)
        return simulate_ideal_video_frame(
            power_maps,
            self.sequence_args,
            self.model_pipeline_params,
        )

    def _advance_to_index(self, index, current_image):
        if index < self._sequence_cursor:
            self._reset_sequence_state()

        sensor_frame = None
        while self._sequence_cursor <= index:
            if self._sequence_cursor == index:
                image = current_image
            else:
                image, _ = self.base_dataset[self._sequence_cursor]
            sensor_frame = self._simulate_sequence_frame(image)
            self._sequence_cursor += 1
        return sensor_frame

    def _simulate_sensor_frame(self, index, image):
        return self._advance_to_index(index, image)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        sensor_frame = self._simulate_sensor_frame(index, image)
        case_name = "nonideal" if self.sensor_nonideal else "ideal"
        sensor_frame = video_sequence_pipeline.scale_case_frame(
            sensor_frame,
            case_name,
            self.sequence_args,
            self.case_range_bounds,
        )
        return self._frame_to_tensor(sensor_frame), label


def _build_sensor_dataset(base_dataset, args, model, source_dataset):
    backend = getattr(args, "sensor_backend", "static")
    if backend == "static":
        return StaticSensorReadoutDataset(base_dataset, args=args, model=model, source_dataset=source_dataset)
    if backend == "video_sequence":
        return VideoSequenceSensorReadoutDataset(base_dataset, args=args, model=model, source_dataset=source_dataset)
    raise ValueError(f"Unsupported sensor backend: {backend}")


def _make_sensor_data_loader(batch_size, dataset, train, sample, num_workers, **kwargs):
    requires_sequence = getattr(dataset, "requires_sequential_access", False)

    if requires_sequence:
        if sample:
            print("Warning: forcing sequential order for sensor_backend=video_sequence.")
        sampler = None
        shuffle = False
        num_workers = 0
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if sample else None
        shuffle = train and sampler is None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        **kwargs,
    )


def get_sensor_loader(batch_size, data_root, train=True, val=True, sample=False, model=None, args=None, **kwargs):
    if args is None:
        raise ValueError("get_sensor_loader requires the parsed argparse namespace in args.")

    kwargs.pop("input_size", None)
    num_workers = kwargs.pop("num_workers", args.sensor_num_workers)
    if num_workers != 0 and getattr(args, "sensor_backend", "static") == "video_sequence":
        print("Warning: forcing num_workers=0 for sensor_backend=video_sequence.")
        num_workers = 0
    elif num_workers != 0:
        print("Warning: sensor loader is most reliable with num_workers=0 due to the simulator state.")

    source_dataset = args.sensor_source_dataset
    ds = []

    if train:
        base_train = _build_sensor_base_dataset(source_dataset, data_root, train=True)
        sensor_train = _build_sensor_dataset(base_train, args=args, model=model, source_dataset=source_dataset)
        ds.append(
            _make_sensor_data_loader(
                batch_size=batch_size,
                dataset=sensor_train,
                train=True,
                sample=sample,
                num_workers=num_workers,
                **kwargs,
            )
        )

    if val:
        base_val = _build_sensor_base_dataset(source_dataset, data_root, train=False)
        sensor_val = _build_sensor_dataset(base_val, args=args, model=model, source_dataset=source_dataset)
        ds.append(
            _make_sensor_data_loader(
                batch_size=batch_size,
                dataset=sensor_val,
                train=False,
                sample=sample,
                num_workers=num_workers,
                **kwargs,
            )
        )

    return ds[0] if len(ds) == 1 else ds
