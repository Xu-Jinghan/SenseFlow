import argparse
import copy
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Final narrow sweep focused on the discovered sensitivity band: "
            "spatial variation 1%-5%, R 5e-3~5e-2, eta 0.3~0.7, and only the key readout paths."
        )
    )
    parser.add_argument("--source-dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--params-csv", default=str(PROJECT_ROOT.parent.parent / "outputs" / "synthetic_image_fit_v3_params.csv"))
    parser.add_argument("--results-dir", required=True)
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


def build_base_args(args):
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
        pmin_density=1.25e-5,
        pmax_density=5e-4,
        device_area_cm2=None,
        force_single_carrier=1,
        single_r=0.01,
        single_eta=0.5,
        single_trise=None,
        single_tfall=None,
        trap_saturation_time=0.00173,
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=1e-7,
        aging_tau_hours=136.5924901605282,
        r_degradation_pct=10.0,
        spatial_variation_r_pct=1.0,
        tia_gain_ohm=1.0,
        integration_gain_v_per_c=1.0,
        video_fps=10.0,
        fps_sim=args.fps_sim,
        adc_bits=8,
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


def build_scenarios(base_args):
    fps_values = [10.0, 20.0, 100.0]
    readout_configs = [
        {"label": "integration_adc4", "analog_readout": "integration", "readout": "integration", "adc_enabled": 1, "adc_bits": 4},
        {"label": "tia", "analog_readout": "tia", "readout": "tia", "adc_enabled": 0, "adc_bits": 8},
    ]
    spatial_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    r_values = [5e-3, 1e-2, 2e-2, 5e-2]
    eta_values = [0.3, 0.5, 0.7]

    scenarios = []
    seen = set()

    def add(name, **overrides):
        cfg = copy.deepcopy(base_args)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        signature = (
            round(cfg.video_fps, 12),
            cfg.analog_readout,
            int(bool(cfg.adc_enabled)),
            int(cfg.adc_bits),
            round(cfg.spatial_variation_r_pct, 12),
            round(cfg.single_r, 12),
            round(cfg.single_eta, 12),
        )
        if signature in seen:
            return
        seen.add(signature)
        scenarios.append((name, cfg))

    for fps in fps_values:
        for readout_config in readout_configs:
            for spatial in spatial_values:
                for r_value in r_values:
                    for eta_value in eta_values:
                        add(
                            f"fps{int(fps):03d}_{readout_config['label']}_sp{int(spatial):02d}_R{r_value:.0e}_eta{eta_value:.1f}".replace(".", "p"),
                            video_fps=fps,
                            single_r=r_value,
                            single_eta=eta_value,
                            spatial_variation_r_pct=spatial,
                            analog_readout=readout_config["analog_readout"],
                            readout=readout_config["readout"],
                            adc_enabled=readout_config["adc_enabled"],
                            adc_bits=readout_config["adc_bits"],
                        )
    return scenarios


def summarize_records(records):
    accuracies = [record["accuracy_nonideal"] for record in records]
    summary = {
        "num_scenarios": len(records),
        "mean_accuracy": statistics.mean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "by_fps": {},
        "by_readout": {},
        "by_spatial": {},
    }
    for key, field in [("by_fps", "video_fps"), ("by_readout", "readout_label"), ("by_spatial", "spatial_variation_r_pct")]:
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record["accuracy_nonideal"])
        summary[key] = {
            str(group_key): {
                "count": len(values),
                "mean_accuracy": statistics.mean(values),
                "min_accuracy": min(values),
                "max_accuracy": max(values),
            }
            for group_key, values in groups.items()
        }
    summary["best_10"] = sorted(
        [
            {
                "scenario_name": record["scenario_name"],
                "accuracy_nonideal": record["accuracy_nonideal"],
                "video_fps": record["video_fps"],
                "readout_label": record["readout_label"],
                "spatial_variation_r_pct": record["spatial_variation_r_pct"],
                "R_single": record["R_single"],
                "eta_single": record["eta_single"],
            }
            for record in records
        ],
        key=lambda item: item["accuracy_nonideal"],
        reverse=True,
    )[:10]
    summary["worst_10"] = sorted(
        [
            {
                "scenario_name": record["scenario_name"],
                "accuracy_nonideal": record["accuracy_nonideal"],
                "video_fps": record["video_fps"],
                "readout_label": record["readout_label"],
                "spatial_variation_r_pct": record["spatial_variation_r_pct"],
                "R_single": record["R_single"],
                "eta_single": record["eta_single"],
            }
            for record in records
        ],
        key=lambda item: item["accuracy_nonideal"],
    )[:10]
    return summary


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    (results_dir / "scenarios").mkdir(parents=True, exist_ok=True)

    base_args = build_base_args(args)
    scenarios = build_scenarios(base_args)
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
        readout_label = "integration_adc4" if cfg.analog_readout == "integration" else "tia"
        record = {
            "scenario_name": name,
            "results_json": cfg.results_json,
            "video_fps": cfg.video_fps,
            "readout_label": readout_label,
            "analog_readout": cfg.analog_readout,
            "adc_enabled": int(bool(cfg.adc_enabled)),
            "adc_bits": cfg.adc_bits if cfg.adc_enabled else "",
            "spatial_variation_r_pct": cfg.spatial_variation_r_pct,
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
        records.append(record)
        manifest.append({"name": name, "results_json": cfg.results_json})

    elapsed = time.time() - started
    summary = summarize_records(records)
    summary["elapsed_sec"] = elapsed

    (results_dir / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (results_dir / "aggregate_results.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (results_dir / "aggregate_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Narrow-sweep aggregate: {results_dir / 'aggregate_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
