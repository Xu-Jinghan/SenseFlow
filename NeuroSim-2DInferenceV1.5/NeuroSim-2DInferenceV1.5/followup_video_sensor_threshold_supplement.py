import argparse
import copy
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused second-round supplement after the first structured sweep and first supplement. "
            "Targets the spatial-variation threshold and the low-R/low-eta collapse boundary."
        )
    )
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--source-dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--params-csv", default=str(PROJECT_ROOT.parent.parent / "outputs" / "synthetic_image_fit_v3_params.csv"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--post-norm", default="auto", choices=["none", "auto", "cifar10", "cifar100", "imagenet"])
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--range-calibration-samples", type=int, default=1024)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def build_base_args(args, best_anchor):
    adc_bits = int(best_anchor["adc_bits"]) if best_anchor["adc_bits"] not in {"", None} else 8
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        split=args.split,
        generate_images=False,
        run_eval=True,
        eval_cases=["nonideal"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=args.model_path,
        num_classes=0,
        results_json=None,
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=args.target_size,
        output_channels=args.output_channels,
        post_norm=args.post_norm,
        num_images=0,
        start_index=0,
        array_size=args.array_size,
        tile_size=256,
        readout="integration",
        analog_readout="integration",
        adc_enabled=0,
        power_max=0.001,
        params_csv=args.params_csv,
        normalization_mode="physical",
        prange1_density=5e-5,
        prange2_density=5e-4,
        pmin_density=float(best_anchor["pmin_density_w_cm2"]),
        pmax_density=float(best_anchor["pmax_density_w_cm2"]),
        device_area_cm2=None,
        force_single_carrier=1,
        single_r=float(best_anchor["R_single"]),
        single_eta=float(best_anchor["eta_single"]),
        single_trise=float(best_anchor["tau_rise_single_s"]),
        single_tfall=float(best_anchor["tau_fall_single_s"]),
        trap_saturation_time=float(best_anchor["trap_saturation_time_s"]),
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=float(best_anchor["noise_1f_density_1hz_a_root_hz"]),
        aging_tau_hours=float(best_anchor["aging_tau_hours"]),
        r_degradation_pct=10.0,
        spatial_variation_r_pct=0.0,
        tia_gain_ohm=1.0,
        integration_gain_v_per_c=1.0,
        video_fps=float(best_anchor["video_fps"]),
        fps_sim=args.fps_sim,
        adc_bits=adc_bits,
        adc_full_scale=None,
        range_mode="minmax",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=args.range_calibration_samples,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=1,
        use_noise_fn=1,
        startup_dark_frames=args.startup_dark_frames,
        output_dir=None,
        analyze_center_pixel=0,
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def make_signature(cfg):
    keys = [
        "video_fps",
        "analog_readout",
        "adc_enabled",
        "adc_bits",
        "pmin_density",
        "pmax_density",
        "single_r",
        "single_eta",
        "trap_saturation_time",
        "trap_amplitude_pct",
        "noise_1f_density_1hz",
        "r_degradation_pct",
        "spatial_variation_r_pct",
    ]
    values = []
    for key in keys:
        value = getattr(cfg, key)
        if isinstance(value, float):
            value = round(value, 12)
        values.append((key, value))
    return tuple(values)


def build_scenarios(base_args):
    scenarios = []
    seen = set()

    def add(name, **overrides):
        cfg = copy.deepcopy(base_args)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        sig = make_signature(cfg)
        if sig in seen:
            return
        seen.add(sig)
        scenarios.append((name, cfg))

    for spatial in [2.0, 3.0, 5.0, 7.0]:
        add(
            f"spatial_threshold_nominal_s{int(spatial):02d}",
            spatial_variation_r_pct=spatial,
            trap_amplitude_pct=0.0,
            r_degradation_pct=10.0,
        )
        add(
            f"spatial_threshold_stress_s{int(spatial):02d}",
            spatial_variation_r_pct=spatial,
            trap_amplitude_pct=20.0,
            r_degradation_pct=80.0,
            trap_saturation_time=1.73,
        )

    for analog_readout, adc_enabled, adc_bits in [
        ("integration", 0, 8),
        ("integration", 1, 8),
        ("tia", 0, 8),
        ("tia", 1, 8),
    ]:
        add(
            f"readout_spatial5_{analog_readout}_adc{adc_bits if adc_enabled else 0}",
            analog_readout=analog_readout,
            readout=analog_readout,
            adc_enabled=adc_enabled,
            adc_bits=adc_bits,
            spatial_variation_r_pct=5.0,
            trap_amplitude_pct=20.0,
            r_degradation_pct=80.0,
        )

    frontier_pairs = [
        (0.005623413251903491, 0.35),
        (0.01, 0.5),
        (0.01778279410038923, 0.35),
        (0.01778279410038923, 0.5),
    ]
    frontier_modes = [
        ("integration", 0, 8, 100.0),
        ("integration", 1, 4, 10.0),
        ("tia", 0, 8, 100.0),
        ("tia", 1, 4, 10.0),
    ]
    for mode_idx, (analog_readout, adc_enabled, adc_bits, fps) in enumerate(frontier_modes):
        for pair_idx, (r_value, eta_value) in enumerate(frontier_pairs[:2]):
            add(
                f"frontier2_{mode_idx:02d}_{pair_idx:02d}",
                analog_readout=analog_readout,
                readout=analog_readout,
                adc_enabled=adc_enabled,
                adc_bits=adc_bits,
                video_fps=fps,
                pmin_density=1.25e-05,
                pmax_density=3.75e-04,
                single_r=r_value,
                single_eta=eta_value,
                noise_1f_density_1hz=1e-07,
                trap_amplitude_pct=20.0,
                trap_saturation_time=0.173,
                r_degradation_pct=80.0,
                spatial_variation_r_pct=10.0,
            )

    return scenarios


def main():
    args = parse_args()
    analysis = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    best_anchor = analysis["best_anchor"]
    base_args = build_base_args(args, best_anchor)
    scenarios = build_scenarios(base_args)

    results_dir = Path(args.results_dir)
    (results_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    records = []
    manifest = []
    started = time.time()

    for index, (name, cfg) in enumerate(scenarios, start=1):
        cfg.results_json = str(results_dir / "scenarios" / f"{name}.json")
        cfg.output_dir = str(results_dir / "visuals" / name)
        result_path = Path(cfg.results_json)
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            print(f"[{index}/{len(scenarios)}] resume {name}", flush=True)
        else:
            print(f"[{index}/{len(scenarios)}] run {name}", flush=True)
            result = pipeline.run_sequence_pipeline(cfg)
        base_params = result.get("base_params", {})
        accuracy = (((result.get("evaluation") or {}).get("cases") or {}).get("nonideal") or {}).get("accuracy")
        records.append(
            {
                "scenario_name": name,
                "results_json": cfg.results_json,
                "video_fps": cfg.video_fps,
                "analog_readout": cfg.analog_readout,
                "adc_enabled": int(bool(cfg.adc_enabled)),
                "adc_bits": cfg.adc_bits if cfg.adc_enabled else "",
                "pmin_density_w_cm2": base_params.get("pmin_density_w_cm2"),
                "pmax_density_w_cm2": base_params.get("pmax_density_w_cm2"),
                "R_single": base_params.get("R_single"),
                "eta_single": base_params.get("eta_single"),
                "trap_saturation_time_s": base_params.get("trap_saturation_time_s"),
                "trap_amplitude_ratio": base_params.get("trap_amplitude_ratio"),
                "noise_1f_density_1hz_a_root_hz": base_params.get("noise_1f_density_1hz_a_root_hz"),
                "r_degradation_ratio": base_params.get("r_degradation_ratio"),
                "spatial_variation_r_ratio": base_params.get("spatial_variation_r_ratio"),
                "accuracy_nonideal": accuracy,
            }
        )
        manifest.append({"name": name, "results_json": cfg.results_json})

    elapsed = time.time() - started
    (results_dir / "supplement_scenarios.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (results_dir / "aggregate_results.json").write_text(json.dumps({"elapsed_sec": elapsed, "records": records}, indent=2), encoding="utf-8")
    with (results_dir / "aggregate_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Follow-up aggregate: {results_dir / 'aggregate_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
