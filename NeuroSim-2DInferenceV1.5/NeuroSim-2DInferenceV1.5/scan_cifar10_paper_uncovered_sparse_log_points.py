import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a sparse log-spaced CIFAR-10 scan for paper-covered points that fall outside "
            "the original R/tr sweep window."
        )
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--params-csv", default=str(PROJECT_ROOT.parent.parent / "outputs" / "synthetic_image_fit_v3_params.csv"))
    parser.add_argument("--paper-csv", default=str(PROJECT_ROOT.parent.parent / "data" / "photodetector_paper_dataset.csv"))
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--readout", default="integration", choices=["integration", "tia"])
    parser.add_argument("--spatial-values", nargs="+", type=float, default=[1.0, 3.0])
    parser.add_argument("--fps-values", nargs="+", type=float, default=[10.0, 20.0, 50.0, 100.0])
    parser.add_argument("--noise-values", nargs="+", type=float, default=[1e-9, 1e-7])
    parser.add_argument(
        "--structures",
        nargs="+",
        default=["Photoconductor", "Phototransistor", "Photodiode"],
    )
    parser.add_argument("--sweep-r-min", type=float, default=0.005)
    parser.add_argument("--sweep-r-max", type=float, default=1.0)
    parser.add_argument("--sweep-tr-min", type=float, default=1e-4)
    parser.add_argument("--sweep-tr-max", type=float, default=1e-1)
    parser.add_argument("--log-snap-decades", type=float, default=1.0)
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


def parse_float(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def snap_log_value(value, min_bound, max_bound, step_decades):
    exponent = math.log10(value)
    snapped = round(exponent / step_decades) * step_decades
    candidate = 10 ** snapped
    if value < min_bound:
        while candidate >= min_bound:
            snapped -= step_decades
            candidate = 10 ** snapped
    elif value > max_bound:
        while candidate <= max_bound:
            snapped += step_decades
            candidate = 10 ** snapped
    return float(candidate)


def load_sparse_pairs(args):
    requested = {str(structure).strip() for structure in args.structures}
    pairs = []
    with Path(args.paper_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entry_id = str(row.get("entry_id", "")).strip()
            if entry_id.startswith("default_") or entry_id.startswith("paper_stub_"):
                continue
            structure = str(row.get("structure", "")).strip()
            if structure not in requested:
                continue
            r_fast = parse_float(row.get("R_fast", ""))
            tau_rise = parse_float(row.get("tau_rise_fast", ""))
            tau_fall = parse_float(row.get("tau_fall_fast", ""))
            if r_fast is None or tau_rise is None or tau_fall is None:
                continue
            tau_avg = 0.5 * (tau_rise + tau_fall)
            if r_fast <= 0.0 or tau_avg <= 0.0:
                continue
            in_sweep = (
                args.sweep_r_min <= r_fast <= args.sweep_r_max
                and args.sweep_tr_min <= tau_avg <= args.sweep_tr_max
            )
            if in_sweep:
                continue
            r_snap = snap_log_value(r_fast, args.sweep_r_min, args.sweep_r_max, args.log_snap_decades)
            t_snap = snap_log_value(tau_avg, args.sweep_tr_min, args.sweep_tr_max, args.log_snap_decades)
            pairs.append(
                {
                    "entry_id": entry_id,
                    "structure": structure,
                    "specific_material": str(row.get("specific_material", "")).strip(),
                    "R_fast": r_fast,
                    "tau_avg_fast": tau_avg,
                    "R_sparse": r_snap,
                    "tr_sparse": t_snap,
                }
            )
    if not pairs:
        raise ValueError("No uncovered paper points were found.")
    unique_pairs = {}
    for pair in pairs:
        key = (pair["R_sparse"], pair["tr_sparse"])
        if key not in unique_pairs:
            unique_pairs[key] = {
                "R_sparse": pair["R_sparse"],
                "tr_sparse": pair["tr_sparse"],
                "covered_entries": [],
            }
        unique_pairs[key]["covered_entries"].append(pair["entry_id"])
    return pairs, [unique_pairs[key] for key in sorted(unique_pairs)]


def build_base_args(args):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset="cifar10",
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
        readout=args.readout,
        analog_readout=args.readout,
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
        single_eta=args.eta,
        single_trise=1e-3,
        single_tfall=1e-3,
        trap_saturation_time=0.00173,
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=1e-8,
        aging_tau_hours=136.5924901605282,
        r_degradation_pct=10.0,
        spatial_variation_r_pct=1.0,
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


def build_scenarios(args, unique_pairs):
    readout_tag = "int" if args.readout == "integration" else "tia"
    scenarios = []
    for noise_value in [float(value) for value in args.noise_values]:
        for fps in [float(value) for value in args.fps_values]:
            for spatial in [float(value) for value in args.spatial_values]:
                for pair in unique_pairs:
                    r_value = float(pair["R_sparse"])
                    tr_value = float(pair["tr_sparse"])
                    scenarios.append(
                        {
                            "scenario_name": (
                                f"supp_{readout_tag}_fps{int(fps):03d}_sp{int(round(spatial)):02d}_"
                                f"R{r_value:.3e}_eta{args.eta:.2f}_tr{tr_value:.0e}_n{noise_value:.0e}"
                            ).replace(".", "p"),
                            "video_fps": fps,
                            "spatial_variation_r_pct": spatial,
                            "single_r": r_value,
                            "single_eta": args.eta,
                            "single_trise": tr_value,
                            "single_tfall": tr_value,
                            "noise_1f_density_1hz": noise_value,
                        }
                    )
    return scenarios


def run_one(base_args, scenario, results_dir, resume):
    cfg = copy.deepcopy(base_args)
    for key, value in scenario.items():
        if key == "scenario_name":
            continue
        setattr(cfg, key, value)
    scenario_dir = Path(results_dir) / "cifar10" / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_json = str(scenario_dir / f"{scenario['scenario_name']}.json")
    result_path = Path(cfg.results_json)
    if resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    else:
        result = pipeline.run_sequence_pipeline(cfg)
    accuracy = ((((result.get("evaluation") or {}).get("cases") or {}).get("nonideal")) or {}).get("accuracy")
    return result, accuracy


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_pairs, unique_pairs = load_sparse_pairs(args)
    with (results_dir / "sparse_log_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["R_sparse", "tr_sparse", "covered_entries"])
        writer.writeheader()
        for row in unique_pairs:
            writer.writerow(
                {
                    "R_sparse": row["R_sparse"],
                    "tr_sparse": row["tr_sparse"],
                    "covered_entries": ";".join(row["covered_entries"]),
                }
            )
    with (results_dir / "paper_uncovered_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_pairs[0].keys()))
        writer.writeheader()
        writer.writerows(raw_pairs)

    scenarios = build_scenarios(args, unique_pairs)
    print(f"Scenarios: {len(scenarios)}", flush=True)

    base_args = build_base_args(args)
    records = []
    started = time.time()
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] {scenario['scenario_name']}", flush=True)
        result, accuracy = run_one(base_args, scenario, results_dir, bool(args.resume))
        base_params = result.get("base_params", {})
        records.append(
            {
                "scenario_name": scenario["scenario_name"],
                "analog_readout": args.readout,
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
                "cifar10_results_json": str(results_dir / "cifar10" / "scenarios" / f"{scenario['scenario_name']}.json"),
                "accuracy_nonideal_cifar10": accuracy,
                "scan_type": "paper_uncovered_sparse_log",
            }
        )

    csv_path = results_dir / "aggregate_results_cifar10.csv"
    json_path = results_dir / "aggregate_results_cifar10.json"
    summary_path = results_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "num_scenarios": len(records),
        "num_sparse_pairs": len(unique_pairs),
        "elapsed_sec": time.time() - started,
        "best_cifar10": max(record["accuracy_nonideal_cifar10"] for record in records),
        "structures": args.structures,
    }
    json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Aggregate CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
