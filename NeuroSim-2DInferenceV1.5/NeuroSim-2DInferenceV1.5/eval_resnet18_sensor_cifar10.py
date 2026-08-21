import argparse
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images as sensor_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CIFAR-10 ResNet18 with the unified sensor pipeline. "
            "This wrapper keeps the historical eval_* argument names while delegating "
            "all work to generate_sensor_verification_images.py."
        )
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT.parent / ".datasets"),
        help="Dataset root. CIFAR-10 will be read from <data_path>/cifar10-data",
    )
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "models" / "resnet18_cifar10.pth"),
        help="Path to the pretrained ResNet18 CIFAR-10 checkpoint",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["raw", "ideal", "nonideal"],
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
        default=str(sensor_pipeline.DEFAULT_RESULTS_JSON),
        help="Optional path to save JSON results",
    )

    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--generate_images", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        default=str(sensor_pipeline.DEFAULT_OUTPUT_DIR),
        help="Directory to save generated comparison images when --generate_images=1",
    )

    parser.add_argument("--sensor_source_dataset", default="cifar10")
    parser.add_argument("--sensor_readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--sensor_array_size", type=int, default=32)
    parser.add_argument("--sensor_target_size", type=int, default=32)
    parser.add_argument("--sensor_output_channels", type=int, default=3)
    parser.add_argument("--sensor_power_max", type=float, default=1.0)
    parser.add_argument(
        "--sensor_params_csv",
        default=str(sensor_pipeline.DEFAULT_PARAMS_CSV),
    )
    parser.add_argument("--sensor_exposure_time", type=float, default=1.0 / 30.0)
    parser.add_argument("--sensor_fps_sim", type=float, default=1000.0)
    parser.add_argument("--sensor_adc_bits", type=int, default=8)
    parser.add_argument("--sensor_adc_full_scale", type=float, default=None)
    parser.add_argument("--sensor_range_mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument("--sensor_percentile_low", type=float, default=1.0)
    parser.add_argument("--sensor_percentile_high", type=float, default=99.0)
    parser.add_argument(
        "--sensor_post_norm",
        default="cifar10",
        choices=["none", "auto", "cifar10", "cifar100", "imagenet"],
    )
    parser.add_argument("--sensor_rng_seed", type=int, default=42)
    parser.add_argument("--sensor_shot_noise", type=int, default=1)
    parser.add_argument("--sensor_bandwidth", type=float, default=5000.0)
    parser.add_argument("--sensor_i_thermal", type=float, default=5e-8)
    parser.add_argument("--sensor_pixel_var_resp", type=float, default=0.08)
    parser.add_argument("--sensor_pixel_var_eta", type=float, default=0.01)
    parser.add_argument("--sensor_pixel_var_tau", type=float, default=0.12)
    parser.add_argument("--sensor_pixel_var_dark", type=float, default=0.30)
    parser.add_argument("--sensor_pixel_var_noise", type=float, default=0.20)
    return parser.parse_args()


def _infer_num_images(args):
    if args.num_images > 0:
        return args.num_images
    if args.generate_images and args.max_eval_batches > 0:
        return args.max_eval_batches * args.batch_size
    return 0


def to_unified_args(args):
    return SimpleNamespace(
        data_root=args.data_path,
        source_dataset=args.sensor_source_dataset,
        split=args.split,
        generate_images=bool(args.generate_images),
        run_eval=True,
        eval_cases=list(args.cases),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=args.model_path,
        num_classes=10,
        results_json=args.results_json,
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=args.sensor_target_size,
        output_channels=args.sensor_output_channels,
        post_norm=args.sensor_post_norm,
        num_images=_infer_num_images(args),
        start_index=args.start_index,
        array_size=args.sensor_array_size,
        tile_size=args.tile_size,
        readout=args.sensor_readout,
        power_max=args.sensor_power_max,
        params_csv=args.sensor_params_csv,
        exposure_time=args.sensor_exposure_time,
        fps_sim=args.sensor_fps_sim,
        adc_bits=args.sensor_adc_bits,
        adc_full_scale=args.sensor_adc_full_scale,
        range_mode=args.sensor_range_mode,
        percentile_low=args.sensor_percentile_low,
        percentile_high=args.sensor_percentile_high,
        i_thermal=args.sensor_i_thermal,
        bandwidth=args.sensor_bandwidth,
        shot_noise=args.sensor_shot_noise,
        output_dir=args.output_dir,
        sensor_pixel_var_resp=args.sensor_pixel_var_resp,
        sensor_pixel_var_eta=args.sensor_pixel_var_eta,
        sensor_pixel_var_tau=args.sensor_pixel_var_tau,
        sensor_pixel_var_dark=args.sensor_pixel_var_dark,
        sensor_pixel_var_noise=args.sensor_pixel_var_noise,
    )


def main():
    args = parse_args()
    sensor_pipeline.run_pipeline(to_unified_args(args))


if __name__ == "__main__":
    main()
