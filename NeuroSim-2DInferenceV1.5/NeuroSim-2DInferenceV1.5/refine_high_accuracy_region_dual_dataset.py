import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refine the high-accuracy parameter region around the current best settings, "
            "evaluate every refined scenario on both CIFAR-100 and CIFAR-10, "
            "and summarize the usable parameter space."
        )
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
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


def build_base_args(args, source_dataset):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=source_dataset,
        split="test",
        generate_images=False,
        run_eval=True,
        eval_cases=["nonideal"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=None,
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
        single_r=0.05,
        single_eta=0.7,
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


def build_scenarios():
    scenarios = []
    seen = set()

    def add(name, analog_readout, adc_enabled, adc_bits, fps, spatial, r_value, eta_value):
        signature = (
            analog_readout,
            int(bool(adc_enabled)),
            int(adc_bits),
            round(float(fps), 12),
            round(float(spatial), 12),
            round(float(r_value), 12),
            round(float(eta_value), 12),
        )
        if signature in seen:
            return
        seen.add(signature)
        scenarios.append(
            {
                "scenario_name": name,
                "analog_readout": analog_readout,
                "readout": analog_readout,
                "adc_enabled": int(bool(adc_enabled)),
                "adc_bits": int(adc_bits),
                "video_fps": float(fps),
                "spatial_variation_r_pct": float(spatial),
                "single_r": float(r_value),
                "single_eta": float(eta_value),
            }
        )

    tia_fps_values = [10.0, 20.0, 50.0, 100.0]
    tia_spatial_values = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    tia_r_values = [0.03, 0.05, 0.07, 0.10]
    tia_eta_values = [0.70, 0.80, 0.90]

    for fps in tia_fps_values:
        for spatial in tia_spatial_values:
            for r_value in tia_r_values:
                for eta_value in tia_eta_values:
                    add(
                        f"tia_fps{int(fps):03d}_sp{spatial:.2f}_R{r_value:.2f}_eta{eta_value:.2f}".replace(".", "p"),
                        analog_readout="tia",
                        adc_enabled=0,
                        adc_bits=8,
                        fps=fps,
                        spatial=spatial,
                        r_value=r_value,
                        eta_value=eta_value,
                    )

    integ_fps_values = [10.0, 20.0, 50.0]
    integ_spatial_values = [0.25, 0.5, 1.0, 1.5, 2.0]
    integ_r_values = [0.05, 0.07, 0.10]
    integ_eta_values = [0.70, 0.80]

    for fps in integ_fps_values:
        for spatial in integ_spatial_values:
            for r_value in integ_r_values:
                for eta_value in integ_eta_values:
                    add(
                        f"int4_fps{int(fps):03d}_sp{spatial:.2f}_R{r_value:.2f}_eta{eta_value:.2f}".replace(".", "p"),
                        analog_readout="integration",
                        adc_enabled=1,
                        adc_bits=4,
                        fps=fps,
                        spatial=spatial,
                        r_value=r_value,
                        eta_value=eta_value,
                    )

    return scenarios


def apply_scenario(base_args, scenario):
    cfg = copy.deepcopy(base_args)
    for key, value in scenario.items():
        if key == "scenario_name":
            continue
        setattr(cfg, key, value)
    return cfg


def run_single_scenario(base_args, scenario, dataset_name, results_dir, resume):
    scenario_args = apply_scenario(base_args, scenario)
    scenario_results_dir = Path(results_dir) / dataset_name / "scenarios"
    scenario_results_dir.mkdir(parents=True, exist_ok=True)
    scenario_args.results_json = str(scenario_results_dir / f"{scenario['scenario_name']}.json")
    result_path = Path(scenario_args.results_json)
    if resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    else:
        result = pipeline.run_sequence_pipeline(scenario_args)
    evaluation = result.get("evaluation") or {}
    cases = evaluation.get("cases") or {}
    accuracy = cases.get("nonideal", {}).get("accuracy")
    return result, accuracy


def summarize_records(records):
    best_c100 = max(record["accuracy_nonideal_cifar100"] for record in records)
    best_c10 = max(record["accuracy_nonideal_cifar10"] for record in records)
    for record in records:
        record["joint_min_accuracy"] = min(record["accuracy_nonideal_cifar100"], record["accuracy_nonideal_cifar10"])
        record["joint_relative_score"] = min(
            record["accuracy_nonideal_cifar100"] / max(best_c100, 1e-12),
            record["accuracy_nonideal_cifar10"] / max(best_c10, 1e-12),
        )

    usable_records = []
    chosen_threshold = None
    for threshold in [0.95, 0.90, 0.85, 0.80]:
        candidate = [
            record for record in records
            if record["accuracy_nonideal_cifar100"] >= threshold * best_c100
            and record["accuracy_nonideal_cifar10"] >= threshold * best_c10
        ]
        if candidate:
            usable_records = candidate
            chosen_threshold = threshold
            break

    summary = {
        "num_scenarios": len(records),
        "best_cifar100": best_c100,
        "best_cifar10": best_c10,
        "mean_cifar100": statistics.mean(record["accuracy_nonideal_cifar100"] for record in records),
        "mean_cifar10": statistics.mean(record["accuracy_nonideal_cifar10"] for record in records),
        "best_by_cifar100": sorted(records, key=lambda item: item["accuracy_nonideal_cifar100"], reverse=True)[:10],
        "best_by_joint_min": sorted(records, key=lambda item: item["joint_min_accuracy"], reverse=True)[:10],
        "usable_threshold_relative_to_best": chosen_threshold,
        "usable_region_count": len(usable_records),
        "usable_region_examples": sorted(usable_records, key=lambda item: item["joint_min_accuracy"], reverse=True)[:20],
    }
    return summary


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()
    print(f"Refinement scenarios: {len(scenarios)}", flush=True)

    base_args_c100 = build_base_args(args, source_dataset="cifar100")
    base_args_c10 = build_base_args(args, source_dataset="cifar10")

    records = []
    started = time.time()
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] {scenario['scenario_name']}", flush=True)
        result_c100, accuracy_c100 = run_single_scenario(
            base_args_c100,
            scenario,
            dataset_name="cifar100",
            results_dir=results_dir,
            resume=bool(args.resume),
        )
        _result_c10, accuracy_c10 = run_single_scenario(
            base_args_c10,
            scenario,
            dataset_name="cifar10",
            results_dir=results_dir,
            resume=bool(args.resume),
        )
        base_params = result_c100.get("base_params", {})
        records.append(
            {
                "scenario_name": scenario["scenario_name"],
                "analog_readout": scenario["analog_readout"],
                "adc_enabled": scenario["adc_enabled"],
                "adc_bits": scenario["adc_bits"] if scenario["adc_enabled"] else "",
                "video_fps": scenario["video_fps"],
                "spatial_variation_r_pct": scenario["spatial_variation_r_pct"],
                "pmin_density_w_cm2": base_params.get("pmin_density_w_cm2"),
                "pmax_density_w_cm2": base_params.get("pmax_density_w_cm2"),
                "R_single": base_params.get("R_single"),
                "eta_single": base_params.get("eta_single"),
                "trap_saturation_time_s": base_params.get("trap_saturation_time_s"),
                "noise_1f_density_1hz_a_root_hz": base_params.get("noise_1f_density_1hz_a_root_hz"),
                "r_degradation_ratio": base_params.get("r_degradation_ratio"),
                "spatial_variation_r_ratio": base_params.get("spatial_variation_r_ratio"),
                "cifar100_results_json": str(results_dir / "cifar100" / "scenarios" / f"{scenario['scenario_name']}.json"),
                "cifar10_results_json": str(results_dir / "cifar10" / "scenarios" / f"{scenario['scenario_name']}.json"),
                "accuracy_nonideal_cifar100": accuracy_c100,
                "accuracy_nonideal_cifar10": accuracy_c10,
            }
        )

    summary = summarize_records(records)
    summary["elapsed_sec"] = time.time() - started

    aggregate_csv_path = results_dir / "aggregate_results_dual_dataset.csv"
    aggregate_json_path = results_dir / "aggregate_results_dual_dataset.json"
    summary_json_path = results_dir / "summary.json"

    with aggregate_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    aggregate_json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Dual aggregate CSV: {aggregate_csv_path}", flush=True)
    print(f"Summary JSON: {summary_json_path}", flush=True)


if __name__ == "__main__":
    main()
