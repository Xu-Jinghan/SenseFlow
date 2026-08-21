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
            "Scan the TIA high-accuracy region while forcing trise=tfall and sweeping 1/f noise, "
            "then report the usable parameter space satisfying CIFAR100>55 and CIFAR10>75."
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


def build_base_args(args, dataset_name):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=dataset_name,
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
        readout="tia",
        analog_readout="tia",
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
        single_r=0.1,
        single_eta=0.9,
        single_trise=1e-3,
        single_tfall=1e-3,
        trap_saturation_time=0.00173,
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=1e-8,
        aging_tau_hours=136.5924901605282,
        r_degradation_pct=10.0,
        spatial_variation_r_pct=0.25,
        tia_gain_ohm=1.0,
        integration_gain_v_per_c=1.0,
        video_fps=20.0,
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
    fps_values = [10.0, 20.0, 50.0, 100.0]
    spatial_values = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    r_values = [0.03, 0.05, 0.07, 0.10]
    eta_values = [0.7, 0.8, 0.9]
    tr_equal_values = [1e-4, 1e-3, 1e-2, 1e-1]
    noise_values = [1e-9, 1e-8, 1e-7]

    scenarios = []
    for fps in fps_values:
        for spatial in spatial_values:
            for r_value in r_values:
                for eta_value in eta_values:
                    for tr_value in tr_equal_values:
                        for noise_value in noise_values:
                            scenario_name = (
                                f"tia_fps{int(fps):03d}_sp{spatial:.2f}_R{r_value:.2f}_eta{eta_value:.2f}"
                                f"_tr{tr_value:.0e}_n{noise_value:.0e}"
                            ).replace(".", "p")
                            scenarios.append(
                                {
                                    "scenario_name": scenario_name,
                                    "video_fps": fps,
                                    "spatial_variation_r_pct": spatial,
                                    "single_r": r_value,
                                    "single_eta": eta_value,
                                    "single_trise": tr_value,
                                    "single_tfall": tr_value,
                                    "noise_1f_density_1hz": noise_value,
                                }
                            )
    return scenarios


def run_one(base_args, scenario, dataset_name, results_dir, resume):
    cfg = copy.deepcopy(base_args)
    for key, value in scenario.items():
        if key == "scenario_name":
            continue
        setattr(cfg, key, value)
    scenario_dir = Path(results_dir) / dataset_name / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_json = str(scenario_dir / f"{scenario['scenario_name']}.json")
    result_path = Path(cfg.results_json)
    if resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    else:
        result = pipeline.run_sequence_pipeline(cfg)
    accuracy = ((((result.get("evaluation") or {}).get("cases") or {}).get("nonideal")) or {}).get("accuracy")
    return result, accuracy


def summarize(records):
    usable = [
        record for record in records
        if record["accuracy_nonideal_cifar100"] > 55.0 and record["accuracy_nonideal_cifar10"] > 75.0
    ]
    usable_summary = {
        "count": len(usable),
        "best_joint": sorted(usable, key=lambda item: min(item["accuracy_nonideal_cifar100"], item["accuracy_nonideal_cifar10"]), reverse=True)[:20],
    }
    if usable:
        usable_summary["ranges"] = {
            "video_fps": sorted({record["video_fps"] for record in usable}),
            "spatial_variation_r_pct": [min(record["spatial_variation_r_pct"] for record in usable), max(record["spatial_variation_r_pct"] for record in usable)],
            "R_single": [min(record["R_single"] for record in usable), max(record["R_single"] for record in usable)],
            "eta_single": [min(record["eta_single"] for record in usable), max(record["eta_single"] for record in usable)],
            "trise_tfall_equal_s": [min(record["trise_tfall_equal_s"] for record in usable), max(record["trise_tfall_equal_s"] for record in usable)],
            "noise_1f_density_1hz_a_root_hz": [min(record["noise_1f_density_1hz_a_root_hz"] for record in usable), max(record["noise_1f_density_1hz_a_root_hz"] for record in usable)],
        }

        by_tr = defaultdict(list)
        by_noise = defaultdict(list)
        for record in usable:
            by_tr[record["trise_tfall_equal_s"]].append(record)
            by_noise[record["noise_1f_density_1hz_a_root_hz"]].append(record)
        usable_summary["by_trise_tfall"] = {
            str(key): {
                "count": len(values),
                "spatial_range": [min(item["spatial_variation_r_pct"] for item in values), max(item["spatial_variation_r_pct"] for item in values)],
                "R_range": [min(item["R_single"] for item in values), max(item["R_single"] for item in values)],
                "eta_range": [min(item["eta_single"] for item in values), max(item["eta_single"] for item in values)],
                "fps_values": sorted({item["video_fps"] for item in values}),
            }
            for key, values in by_tr.items()
        }
        usable_summary["by_noise"] = {
            str(key): {
                "count": len(values),
                "spatial_range": [min(item["spatial_variation_r_pct"] for item in values), max(item["spatial_variation_r_pct"] for item in values)],
                "R_range": [min(item["R_single"] for item in values), max(item["R_single"] for item in values)],
                "eta_range": [min(item["eta_single"] for item in values), max(item["eta_single"] for item in values)],
                "trise_tfall_values": sorted({item["trise_tfall_equal_s"] for item in values}),
            }
            for key, values in by_noise.items()
        }

    summary = {
        "num_scenarios": len(records),
        "best_cifar100": max(record["accuracy_nonideal_cifar100"] for record in records),
        "best_cifar10": max(record["accuracy_nonideal_cifar10"] for record in records),
        "mean_cifar100": statistics.mean(record["accuracy_nonideal_cifar100"] for record in records),
        "mean_cifar10": statistics.mean(record["accuracy_nonideal_cifar10"] for record in records),
        "usable_space": usable_summary,
    }
    return summary


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()
    print(f"Scenarios: {len(scenarios)}", flush=True)

    base_args_c100 = build_base_args(args, dataset_name="cifar100")
    base_args_c10 = build_base_args(args, dataset_name="cifar10")

    records = []
    started = time.time()
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] {scenario['scenario_name']}", flush=True)
        result_c100, acc_c100 = run_one(base_args_c100, scenario, "cifar100", results_dir, bool(args.resume))
        _result_c10, acc_c10 = run_one(base_args_c10, scenario, "cifar10", results_dir, bool(args.resume))
        base_params = result_c100.get("base_params", {})
        records.append(
            {
                "scenario_name": scenario["scenario_name"],
                "analog_readout": "tia",
                "adc_enabled": 0,
                "adc_bits": "",
                "video_fps": scenario["video_fps"],
                "spatial_variation_r_pct": scenario["spatial_variation_r_pct"],
                "pmin_density_w_cm2": base_params.get("pmin_density_w_cm2"),
                "pmax_density_w_cm2": base_params.get("pmax_density_w_cm2"),
                "R_single": base_params.get("R_single"),
                "eta_single": base_params.get("eta_single"),
                "trise_tfall_equal_s": scenario["single_trise"],
                "trap_saturation_time_s": base_params.get("trap_saturation_time_s"),
                "noise_1f_density_1hz_a_root_hz": base_params.get("noise_1f_density_1hz_a_root_hz"),
                "r_degradation_ratio": base_params.get("r_degradation_ratio"),
                "spatial_variation_r_ratio": base_params.get("spatial_variation_r_ratio"),
                "cifar100_results_json": str(results_dir / "cifar100" / "scenarios" / f"{scenario['scenario_name']}.json"),
                "cifar10_results_json": str(results_dir / "cifar10" / "scenarios" / f"{scenario['scenario_name']}.json"),
                "accuracy_nonideal_cifar100": acc_c100,
                "accuracy_nonideal_cifar10": acc_c10,
            }
        )

    summary = summarize(records)
    summary["elapsed_sec"] = time.time() - started

    csv_path = results_dir / "aggregate_results_dual_dataset.csv"
    json_path = results_dir / "aggregate_results_dual_dataset.json"
    summary_path = results_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Aggregate CSV: {csv_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
