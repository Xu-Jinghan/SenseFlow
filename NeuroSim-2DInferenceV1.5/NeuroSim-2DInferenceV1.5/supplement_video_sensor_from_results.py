import argparse
import copy
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import generate_sensor_verification_images_video_sequence as video_sequence_pipeline
import sweep_video_sensor_structured as structured_sweep


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an existing structured video-sensor sweep, generate a targeted supplementation set, "
            "and run the supplemental scenarios without touching the original files."
        )
    )
    parser.add_argument("--existing-aggregate-csv", required=True)
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
    parser.add_argument("--device-area-cm2", type=float, default=None)
    parser.add_argument("--prange1-density", type=float, default=5e-5)
    parser.add_argument("--prange2-density", type=float, default=5e-4)
    parser.add_argument("--aging-tau-hours", type=float, default=136.5924901605282)
    parser.add_argument("--tia-gain-ohm", type=float, default=1.0)
    parser.add_argument("--integration-gain-v-per-c", type=float, default=1.0)
    parser.add_argument("--drift-hours", type=float, default=0.0)
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
        power_max=video_sequence_pipeline.base_pipeline.DEFAULT_POWER_MAX_W,
        params_csv=args.params_csv,
        normalization_mode="physical",
        prange1_density=args.prange1_density,
        prange2_density=args.prange2_density,
        pmin_density=None,
        pmax_density=None,
        device_area_cm2=args.device_area_cm2,
        force_single_carrier=1,
        single_r=None,
        single_eta=None,
        single_trise=None,
        single_tfall=None,
        trap_saturation_time=None,
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=1e-8,
        aging_tau_hours=args.aging_tau_hours,
        r_degradation_pct=10.0,
        spatial_variation_r_pct=0.0,
        tia_gain_ohm=args.tia_gain_ohm,
        integration_gain_v_per_c=args.integration_gain_v_per_c,
        video_fps=50.0,
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
        drift_hours=[args.drift_hours],
        drift_aging_power_w=None,
    )


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def normalize_existing_row(row):
    out = dict(row)
    out["accuracy_nonideal"] = float(row["accuracy_nonideal"])
    out["video_fps"] = float(row["video_fps"])
    out["adc_enabled"] = bool(int(row["adc_enabled"]))
    out["adc_bits"] = None if row["adc_bits"] == "" else int(row["adc_bits"])
    for key in [
        "pmin_density_w_cm2",
        "pmax_density_w_cm2",
        "R_single",
        "eta_single",
        "tau_rise_single_s",
        "tau_fall_single_s",
        "trap_saturation_time_s",
        "trap_amplitude_ratio",
        "noise_1f_density_1hz_a_root_hz",
        "aging_tau_hours",
        "r_degradation_ratio",
        "spatial_variation_r_ratio",
    ]:
        out[key] = float(row[key]) if row[key] not in {"", None} else None
    return out


def scenario_signature(config):
    keys = [
        "video_fps",
        "analog_readout",
        "adc_enabled",
        "adc_bits",
        "pmin_density",
        "pmax_density",
        "single_r",
        "single_eta",
        "single_trise",
        "single_tfall",
        "trap_saturation_time",
        "trap_amplitude_pct",
        "noise_1f_density_1hz",
        "r_degradation_pct",
        "spatial_variation_r_pct",
    ]
    values = []
    for key in keys:
        value = config[key]
        if isinstance(value, float):
            value = round(value, 12)
        values.append((key, value))
    return tuple(values)


def summarize_effect(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row["accuracy_nonideal"])
    return [
        {"value": key_value, "count": len(values), "mean_accuracy": statistics.mean(values)}
        for key_value, values in groups.items()
    ]


def pick_anchor_rows(rows):
    best_row = max(rows, key=lambda row: row["accuracy_nonideal"])
    viable_rows = [row for row in rows if row["R_single"] >= 0.03162277660168379 and row["accuracy_nonideal"] < 60.0]
    if not viable_rows:
        viable_rows = sorted(rows, key=lambda row: row["accuracy_nonideal"])[:10]
    fragile_row = min(viable_rows, key=lambda row: row["accuracy_nonideal"])
    return best_row, fragile_row


def worst_combo_values(rows, key, count):
    summaries = summarize_effect(rows, key)
    summaries.sort(key=lambda item: item["mean_accuracy"])
    return [item["value"] for item in summaries[:count]]


def add_scenario(scenarios, seen, name, group, base_config, **overrides):
    config = dict(base_config)
    config.update(overrides)
    if not config["adc_enabled"]:
        config["adc_bits"] = 8
    signature = scenario_signature(config)
    if signature in seen:
        return
    seen.add(signature)
    scenarios.append({"name": name, "group": group, "config": config})


def build_supplement_scenarios(rows, base_args):
    rows = [normalize_existing_row(row) for row in rows]
    best_row, fragile_row = pick_anchor_rows(rows)
    worst_readouts = worst_combo_values(rows, "analog_readout", 2)
    worst_fps = worst_combo_values(rows, "video_fps", 2)
    low_pmin = min(row["pmin_density_w_cm2"] for row in rows)
    low_pmax = min(row["pmax_density_w_cm2"] for row in rows)
    high_noise = max(row["noise_1f_density_1hz_a_root_hz"] for row in rows)
    single_trise = float(best_row["tau_rise_single_s"])
    single_tfall = float(best_row["tau_fall_single_s"])

    best_config = {
        "video_fps": best_row["video_fps"],
        "analog_readout": best_row["analog_readout"],
        "readout": best_row["analog_readout"],
        "adc_enabled": best_row["adc_enabled"],
        "adc_bits": 8 if best_row["adc_bits"] is None else best_row["adc_bits"],
        "pmin_density": best_row["pmin_density_w_cm2"],
        "pmax_density": best_row["pmax_density_w_cm2"],
        "single_r": best_row["R_single"],
        "single_eta": best_row["eta_single"],
        "single_trise": single_trise,
        "single_tfall": single_tfall,
        "trap_saturation_time": single_trise,
        "trap_amplitude_pct": 0.0,
        "noise_1f_density_1hz": best_row["noise_1f_density_1hz_a_root_hz"],
        "r_degradation_pct": 10.0,
        "spatial_variation_r_pct": 0.0,
    }
    fragile_config = {
        "video_fps": fragile_row["video_fps"],
        "analog_readout": fragile_row["analog_readout"],
        "readout": fragile_row["analog_readout"],
        "adc_enabled": fragile_row["adc_enabled"],
        "adc_bits": 8 if fragile_row["adc_bits"] is None else fragile_row["adc_bits"],
        "pmin_density": fragile_row["pmin_density_w_cm2"],
        "pmax_density": fragile_row["pmax_density_w_cm2"],
        "single_r": fragile_row["R_single"],
        "single_eta": fragile_row["eta_single"],
        "single_trise": single_trise,
        "single_tfall": single_tfall,
        "trap_saturation_time": 10.0 * single_trise,
        "trap_amplitude_pct": 0.0,
        "noise_1f_density_1hz": fragile_row["noise_1f_density_1hz_a_root_hz"],
        "r_degradation_pct": 10.0,
        "spatial_variation_r_pct": 0.0,
    }

    scenarios = []
    seen = set()

    trap_amplitudes = [5.0, 20.0]
    trap_times = [10.0 * single_trise, 1000.0 * single_trise]
    degradations = [30.0, 80.0]
    spatials = [1.0, 10.0]

    for anchor_name, anchor_config in [("best_anchor", best_config), ("fragile_anchor", fragile_config)]:
        for trap_amp in trap_amplitudes:
            for trap_time in trap_times:
                for degradation in degradations:
                    for spatial in spatials:
                        add_scenario(
                            scenarios,
                            seen,
                            f"{anchor_name}_trap{int(trap_amp):02d}_deg{int(degradation):02d}_sp{int(spatial):02d}_t{int(round(trap_time / single_trise)):04d}",
                            "corrected_missing_dims",
                            anchor_config,
                            trap_amplitude_pct=trap_amp,
                            trap_saturation_time=trap_time,
                            r_degradation_pct=degradation,
                            spatial_variation_r_pct=spatial,
                        )

    frontier_readouts = []
    readout_summary = summarize_effect(rows, "analog_readout")
    readout_summary.sort(key=lambda item: item["mean_accuracy"])
    worst_readout_labels = [item["value"] for item in readout_summary[:2]]
    for readout_label in worst_readout_labels:
        frontier_readouts.append({"analog_readout": readout_label, "readout": readout_label, "adc_enabled": False, "adc_bits": 8})
        frontier_readouts.append({"analog_readout": readout_label, "readout": readout_label, "adc_enabled": True, "adc_bits": 4})

    frontier_fps = [float(value) for value in worst_fps]
    frontier_r = [10 ** -2.25, 10 ** -1.5]
    frontier_eta = [0.35, 0.65]
    frontier_pairs = list(zip(frontier_r * 2, frontier_eta * 2))

    for idx, readout_config in enumerate(frontier_readouts):
        fps_value = frontier_fps[idx % len(frontier_fps)]
        for pair_idx, (r_value, eta_value) in enumerate(frontier_pairs):
            add_scenario(
                scenarios,
                seen,
                f"frontier_{idx:02d}_{pair_idx:02d}",
                "frontier_refinement",
                fragile_config,
                **readout_config,
                video_fps=fps_value,
                pmin_density=low_pmin,
                pmax_density=low_pmax,
                single_r=float(r_value),
                single_eta=float(eta_value),
                noise_1f_density_1hz=high_noise,
                trap_amplitude_pct=20.0,
                trap_saturation_time=100.0 * single_trise,
                r_degradation_pct=80.0,
                spatial_variation_r_pct=10.0,
            )

    analysis_summary = {
        "num_existing_rows": len(rows),
        "best_anchor": best_row,
        "fragile_anchor": fragile_row,
        "worst_readouts": worst_readouts,
        "worst_fps": worst_fps,
        "supplement_counts": defaultdict(int),
    }
    for scenario in scenarios:
        analysis_summary["supplement_counts"][scenario["group"]] += 1
    analysis_summary["supplement_counts"] = dict(analysis_summary["supplement_counts"])
    return scenarios, analysis_summary


def build_scenario_namespace(base_args, scenario, results_dir):
    scenario_args = copy.deepcopy(base_args)
    for key, value in scenario["config"].items():
        setattr(scenario_args, key, value)
    scenario_args.results_json = str(results_dir / "scenarios" / f"{scenario['name']}.json")
    scenario_args.output_dir = str(results_dir / "visuals" / scenario["name"])
    return scenario_args


def extract_accuracy(result):
    evaluation = result.get("evaluation") or {}
    cases = evaluation.get("cases") or {}
    return cases.get("nonideal", {}).get("accuracy")


def flatten_record(scenario, scenario_args, result):
    base_params = result.get("base_params", {})
    return {
        "scenario_name": scenario["name"],
        "scenario_group": scenario["group"],
        "results_json": scenario_args.results_json,
        "video_fps": scenario["config"]["video_fps"],
        "analog_readout": scenario["config"]["analog_readout"],
        "adc_enabled": int(bool(scenario["config"]["adc_enabled"])),
        "adc_bits": int(scenario["config"]["adc_bits"]) if scenario["config"]["adc_enabled"] else "",
        "pmin_density_w_cm2": base_params.get("pmin_density_w_cm2"),
        "pmax_density_w_cm2": base_params.get("pmax_density_w_cm2"),
        "R_single": base_params.get("R_single"),
        "eta_single": base_params.get("eta_single"),
        "trap_saturation_time_s": base_params.get("trap_saturation_time_s"),
        "trap_amplitude_ratio": base_params.get("trap_amplitude_ratio"),
        "noise_1f_density_1hz_a_root_hz": base_params.get("noise_1f_density_1hz_a_root_hz"),
        "r_degradation_ratio": base_params.get("r_degradation_ratio"),
        "spatial_variation_r_ratio": base_params.get("spatial_variation_r_ratio"),
        "accuracy_nonideal": extract_accuracy(result),
    }


def write_outputs(results_dir, analysis_summary, scenarios, records):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    (results_dir / "supplement_scenarios.json").write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    (results_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")

    aggregate_json_path = results_dir / "aggregate_results.json"
    aggregate_csv_path = results_dir / "aggregate_results.csv"
    aggregate_json_path.write_text(
        json.dumps({"analysis_summary": analysis_summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    if records:
        with aggregate_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = load_rows(args.existing_aggregate_csv)
    base_args = build_base_args(args)
    scenarios, analysis_summary = build_supplement_scenarios(rows, base_args)
    print(f"Supplement scenarios: {len(scenarios)}", flush=True)

    started = time.time()
    records = []
    for index, scenario in enumerate(scenarios, start=1):
        scenario_args = build_scenario_namespace(base_args, scenario, results_dir)
        result_path = Path(scenario_args.results_json)
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            print(f"[{index}/{len(scenarios)}] resume {scenario['name']}", flush=True)
        else:
            print(f"[{index}/{len(scenarios)}] run {scenario['name']} ({scenario['group']})", flush=True)
            result = video_sequence_pipeline.run_sequence_pipeline(scenario_args)
        records.append(flatten_record(scenario, scenario_args, result))

    analysis_summary["elapsed_sec"] = time.time() - started
    write_outputs(results_dir, analysis_summary, scenarios, records)
    print(f"Supplement aggregate: {results_dir / 'aggregate_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
