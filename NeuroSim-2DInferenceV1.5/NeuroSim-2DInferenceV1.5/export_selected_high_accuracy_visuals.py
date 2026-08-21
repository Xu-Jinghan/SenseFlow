import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AGGREGATE = PROJECT_ROOT.parent.parent / "outputs" / "final_narrow_sweep_cifar100_nonideal_200" / "aggregate_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent.parent / "outputs" / "selected_high_accuracy_visuals_cifar100"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export comparison images and center-pixel analysis assets for a few high-accuracy scenarios "
            "selected from an existing sweep result table."
        )
    )
    parser.add_argument("--aggregate-csv", default=str(DEFAULT_AGGREGATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument(
        "--scenario-name",
        action="append",
        dest="scenario_names",
        default=[],
        help="Optional explicit scenario names. If omitted, the built-in curated high-accuracy set is used.",
    )
    return parser.parse_args()


def default_scenarios():
    return [
        "fps100_tia_sp01_R5e-02_eta0p7",
        "fps010_tia_sp01_R5e-02_eta0p7",
        "fps020_tia_sp01_R5e-02_eta0p7",
        "fps010_tia_sp02_R5e-02_eta0p7",
        "fps010_integration_adc4_sp01_R5e-02_eta0p7",
    ]


def load_rows(aggregate_csv):
    rows = {}
    with Path(aggregate_csv).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["scenario_name"]] = row
    return rows


def build_args(row, args, output_dir):
    adc_enabled = int(row["adc_enabled"])
    adc_bits = int(row["adc_bits"]) if row["adc_bits"] else 8
    analog_readout = row["analog_readout"]
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        split=args.split,
        generate_images=True,
        run_eval=False,
        eval_cases=["nonideal"],
        batch_size=20,
        num_workers=0,
        max_eval_batches=0,
        model_path=None,
        num_classes=0,
        results_json=str(output_dir / "results.json"),
        seed=1234,
        sensor_rng_seed=42,
        target_size=32,
        output_channels=3,
        post_norm="auto",
        num_images=args.num_images,
        start_index=0,
        array_size=32,
        tile_size=256,
        readout=analog_readout,
        analog_readout=analog_readout,
        adc_enabled=adc_enabled,
        power_max=0.001,
        params_csv=str(PROJECT_ROOT.parent.parent / "outputs" / "synthetic_image_fit_v3_params.csv"),
        normalization_mode="physical",
        prange1_density=5e-5,
        prange2_density=5e-4,
        pmin_density=float(row["pmin_density_w_cm2"]),
        pmax_density=float(row["pmax_density_w_cm2"]),
        device_area_cm2=None,
        force_single_carrier=1,
        single_r=float(row["R_single"]),
        single_eta=float(row["eta_single"]),
        single_trise=None,
        single_tfall=None,
        trap_saturation_time=float(row["trap_saturation_time_s"]),
        trap_amplitude_pct=float(row["trap_amplitude_ratio"]) * 100.0,
        noise_1f_density_1hz=float(row["noise_1f_density_1hz_a_root_hz"]),
        aging_tau_hours=136.5924901605282,
        r_degradation_pct=float(row["r_degradation_ratio"]) * 100.0,
        spatial_variation_r_pct=float(row["spatial_variation_r_ratio"]) * 100.0,
        tia_gain_ohm=1.0,
        integration_gain_v_per_c=1.0,
        video_fps=float(row["video_fps"]),
        fps_sim=1000.0,
        adc_bits=adc_bits,
        adc_full_scale=None,
        range_mode="minmax",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=1024,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=1,
        use_noise_fn=1,
        startup_dark_frames=0,
        output_dir=str(output_dir),
        analyze_center_pixel=1,
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def main():
    args = parse_args()
    rows = load_rows(args.aggregate_csv)
    scenario_names = args.scenario_names or default_scenarios()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for scenario_name in scenario_names:
        if scenario_name not in rows:
            raise KeyError(f"Scenario {scenario_name!r} not found in {args.aggregate_csv}")
        row = rows[scenario_name]
        scenario_output_dir = base_output_dir / scenario_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        run_args = build_args(row, args, scenario_output_dir)
        print(f"Exporting visuals for {scenario_name}", flush=True)
        result = pipeline.run_sequence_pipeline(run_args)
        summary.append(
            {
                "scenario_name": scenario_name,
                "accuracy_nonideal_from_sweep": float(row["accuracy_nonideal"]),
                "output_dir": str(scenario_output_dir),
                "results_json": str(scenario_output_dir / "results.json"),
                "comparison_dir": str(scenario_output_dir / "comparison"),
                "analysis_dir": str(scenario_output_dir / "analysis"),
                "image_generation": result.get("image_generation"),
                "center_pixel_analysis": result.get("center_pixel_analysis"),
            }
        )

    summary_path = base_output_dir / "selected_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
