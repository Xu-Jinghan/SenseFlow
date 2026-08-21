import argparse
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as video_sequence_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ResNet18 with the stateful video-sequence sensor pipeline. "
            "Supports both CIFAR-10 and CIFAR-100 and delegates all work to "
            "generate_sensor_verification_images_video_sequence.py."
        )
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT.parent / ".datasets"),
        help="Dataset root. Data will be read from <data_path>/<source_dataset>-data",
    )
    parser.add_argument(
        "--source_dataset",
        "--sensor_source_dataset",
        dest="source_dataset",
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Base dataset used for sequence generation and evaluation",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to the pretrained ResNet18 checkpoint. Defaults to models/resnet18_<source_dataset>.pth",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["nonideal"],
        choices=["raw", "ideal", "nonideal"],
        help="Evaluation cases to run",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=50,
        help="If > 0, only evaluate this many batches per case",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--results_json",
        default=None,
        help="Optional path to save JSON results. Defaults to an artifacts/eval_runs path for the selected dataset.",
    )

    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--generate_images", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save generated comparison images when --generate_images=1. Defaults to a dataset-specific folder.",
    )
    parser.add_argument(
        "--analyze_center_pixel",
        type=int,
        default=0,
        help="Whether to save the center-pixel waveform analysis assets.",
    )

    parser.add_argument("--num_classes", type=int, default=0)
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
    parser.add_argument(
        "--power_max",
        type=float,
        default=video_sequence_pipeline.base_pipeline.DEFAULT_POWER_MAX_W,
    )
    parser.add_argument(
        "--params_csv",
        default=str(video_sequence_pipeline.base_pipeline.DEFAULT_PARAMS_CSV),
    )
    parser.add_argument("--normalization_mode", default="physical", choices=["physical", "calibration", "per_frame", "none"])
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
    parser.add_argument("--spatial_variation_r_pct", type=float, default=None)
    parser.add_argument("--tia_gain_ohm", type=float, default=None)
    parser.add_argument("--integration_gain_v_per_c", type=float, default=None)
    parser.add_argument("--video_fps", type=float, default=20.0)
    parser.add_argument("--fps_sim", type=float, default=1000.0)
    parser.add_argument("--adc_bits", type=int, default=8)
    parser.add_argument("--adc_full_scale", type=float, default=None)
    parser.add_argument("--adc_calibration_low", type=float, default=None)
    parser.add_argument("--adc_calibration_high", type=float, default=None)
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
    parser.add_argument("--sensor_rng_seed", type=int, default=42)
    parser.add_argument("--use_noise_fn", type=int, default=1)
    parser.add_argument("--shot_noise", type=int, default=1)
    parser.add_argument("--bandwidth", type=float, default=5000.0)
    parser.add_argument("--i_thermal", type=float, default=0.0)
    parser.add_argument("--startup_dark_frames", type=int, default=0)
    return parser.parse_args()


def _infer_num_images(args):
    if args.num_images > 0:
        return args.num_images
    if args.generate_images and args.max_eval_batches > 0:
        return args.max_eval_batches * args.batch_size
    return 0


def to_sequence_args(args):
    return SimpleNamespace(
        data_root=args.data_path,
        source_dataset=args.source_dataset,
        split=args.split,
        generate_images=bool(args.generate_images),
        run_eval=True,
        eval_cases=list(args.cases),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=args.model_path,
        num_classes=args.num_classes,
        results_json=args.results_json,
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        use_noise_fn=args.use_noise_fn,
        target_size=args.target_size,
        output_channels=args.output_channels,
        post_norm=args.post_norm,
        num_images=_infer_num_images(args),
        start_index=args.start_index,
        array_size=args.array_size,
        tile_size=args.tile_size,
        readout=args.analog_readout if args.analog_readout is not None else args.readout,
        analog_readout=args.analog_readout,
        adc_enabled=args.adc_enabled,
        power_max=args.power_max,
        params_csv=args.params_csv,
        normalization_mode=args.normalization_mode,
        prange1_density=args.prange1_density,
        prange2_density=args.prange2_density,
        pmin_density=args.pmin_density,
        pmax_density=args.pmax_density,
        device_area_cm2=args.device_area_cm2,
        force_single_carrier=args.force_single_carrier,
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
        video_fps=args.video_fps,
        fps_sim=args.fps_sim,
        adc_bits=args.adc_bits,
        adc_full_scale=args.adc_full_scale,
        adc_calibration_low=args.adc_calibration_low,
        adc_calibration_high=args.adc_calibration_high,
        range_mode=args.range_mode,
        range_scope=args.range_scope,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        range_calibration_split=args.range_calibration_split,
        range_calibration_samples=args.range_calibration_samples,
        i_thermal=args.i_thermal,
        bandwidth=args.bandwidth,
        shot_noise=args.shot_noise,
        startup_dark_frames=args.startup_dark_frames,
        output_dir=args.output_dir,
        analyze_center_pixel=args.analyze_center_pixel,
    )


def main():
    args = parse_args()
    video_sequence_pipeline.run_sequence_pipeline(to_sequence_args(args))


if __name__ == "__main__":
    main()
