import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_CASE2_CSV = REPO_ROOT / "outputs" / "case2_fit" / "case2_fit_parameters.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "case2_noiseless_x2_eval_cifar10"
READOUT_GROUPS = (
    ("tia", "TIA"),
    ("integration", "Integration"),
    ("adc4", "ADC 4-bit"),
    ("adc8", "ADC 8-bit"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate case2 native parameters versus an x2-enhanced variant on the "
            "stateful video-sequence CIFAR-10 pipeline without temporal noise."
        )
    )
    parser.add_argument("--params-csv", default=str(DEFAULT_CASE2_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--visual-index", type=int, default=23)
    parser.add_argument("--num-visual-images", type=int, default=1)
    parser.add_argument("--fps", nargs="+", type=float, default=[20.0, 50.0, 100.0])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--x2-delta-scale", type=float, default=50.0)
    return parser.parse_args()


def load_parameter_rows(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row["parameter"].strip()
            if not key:
                continue
            rows.append((key, row["value"]))
    return rows


def raw_parameter_dict(rows):
    values = {}
    for key, value in rows:
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def write_parameter_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


def derive_case2_eval_csvs(args, output_dir):
    base_rows = load_parameter_rows(args.params_csv)
    base_values = raw_parameter_dict(base_rows)
    if "delta" not in base_values:
        raise ValueError(f"Expected a case2 parameter CSV with a delta entry: {args.params_csv}")
    if "device_area_cm2" not in base_values:
        raise ValueError(f"Expected device_area_cm2 in {args.params_csv}")

    native_csv = output_dir / "case2_native_eval_params.csv"
    write_parameter_csv(native_csv, base_rows)

    boosted_rows = []
    boosted_delta = float(base_values["delta"]) * float(args.x2_delta_scale)
    for key, value in base_rows:
        if key == "delta":
            boosted_rows.append((key, f"{boosted_delta:.16e}"))
        else:
            boosted_rows.append((key, value))
    boosted_csv = output_dir / "case2_x2_enhanced_eval_params.csv"
    write_parameter_csv(boosted_csv, boosted_rows)

    power_on_w = float(base_values.get("power_on_w", base_values.get("power_ref_w")))
    device_area_cm2 = float(base_values["device_area_cm2"])
    if power_on_w <= 0 or device_area_cm2 <= 0:
        raise ValueError("case2 power_on_w/power_ref_w and device_area_cm2 must be positive")
    pmax_density_w_cm2 = power_on_w / device_area_cm2

    summary = {
        "native_csv": str(native_csv),
        "x2_enhanced_csv": str(boosted_csv),
        "native_delta_a": float(base_values["delta"]),
        "x2_enhanced_delta_a": boosted_delta,
        "x2_delta_scale": float(args.x2_delta_scale),
        "device_area_cm2": device_area_cm2,
        "power_on_w": power_on_w,
        "mapped_pmax_density_w_cm2": pmax_density_w_cm2,
        "mapped_pmin_density_w_cm2": 0.0,
        "mapped_prange1_density_w_cm2": 0.0,
        "mapped_prange2_density_w_cm2": pmax_density_w_cm2,
    }
    summary_path = output_dir / "derived_case2_eval_params.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_args_namespace(args, params_csv, results_dir, fps, readout_key, pmax_density_w_cm2):
    if readout_key == "tia":
        readout = "tia"
        analog_readout = "tia"
        adc_enabled = 0
        adc_bits = 8
    elif readout_key == "integration":
        readout = "integration"
        analog_readout = "integration"
        adc_enabled = 0
        adc_bits = 8
    elif readout_key == "adc4":
        readout = "adc"
        analog_readout = "integration"
        adc_enabled = 1
        adc_bits = 4
    elif readout_key == "adc8":
        readout = "adc"
        analog_readout = "integration"
        adc_enabled = 1
        adc_bits = 8
    else:
        raise ValueError(f"Unsupported readout_key: {readout_key}")

    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        split=args.split,
        generate_images=True,
        run_eval=True,
        eval_cases=["nonideal"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=None,
        num_classes=0,
        results_json=str(results_dir / "results.json"),
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=args.target_size,
        output_channels=args.output_channels,
        post_norm="auto",
        num_images=args.num_visual_images,
        start_index=args.visual_index,
        array_size=args.array_size,
        tile_size=args.tile_size,
        readout=readout,
        analog_readout=analog_readout,
        adc_enabled=adc_enabled,
        power_max=pmax_density_w_cm2 * 1.0,
        params_csv=str(params_csv),
        normalization_mode="physical",
        prange1_density=0.0,
        prange2_density=pmax_density_w_cm2,
        pmin_density=0.0,
        pmax_density=pmax_density_w_cm2,
        device_area_cm2=None,
        force_single_carrier=0,
        single_r=None,
        single_eta=None,
        single_trise=None,
        single_tfall=None,
        trap_saturation_time=None,
        trap_amplitude_pct=None,
        noise_1f_density_1hz=0.0,
        aging_tau_hours=None,
        r_degradation_pct=None,
        spatial_variation_r_pct=0.0,
        tia_gain_ohm=1.0,
        integration_gain_v_per_c=1.0,
        video_fps=float(fps),
        fps_sim=args.fps_sim,
        adc_bits=adc_bits,
        adc_full_scale=None,
        range_mode="auto",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=1024,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=0,
        use_noise_fn=0,
        startup_dark_frames=0,
        output_dir=str(results_dir),
        analyze_center_pixel=0,
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def load_manifest_sample(manifest_path):
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"No exported samples found in {manifest_path}")
    return samples[0]


def run_all_scenarios(args, derived_summary):
    output_dir = Path(args.output_dir)
    scenario_rows = []
    families = [
        ("native", Path(derived_summary["native_csv"])),
        ("x2_enhanced", Path(derived_summary["x2_enhanced_csv"])),
    ]
    pmax_density_w_cm2 = float(derived_summary["mapped_pmax_density_w_cm2"])

    for family_name, params_csv in families:
        for fps in args.fps:
            for readout_key, readout_label in READOUT_GROUPS:
                scenario_name = f"{family_name}_fps{int(round(fps)):03d}_{readout_key}"
                scenario_dir = output_dir / family_name / scenario_name
                scenario_dir.mkdir(parents=True, exist_ok=True)
                run_args = build_args_namespace(
                    args,
                    params_csv=params_csv,
                    results_dir=scenario_dir,
                    fps=fps,
                    readout_key=readout_key,
                    pmax_density_w_cm2=pmax_density_w_cm2,
                )
                print(f"Running {scenario_name}", flush=True)
                result = pipeline.run_sequence_pipeline(run_args)
                accuracy = float(result["evaluation"]["cases"]["nonideal"]["accuracy"])
                manifest_path = Path(result["image_generation"]["manifest_path"])
                sample_item = load_manifest_sample(manifest_path)
                scenario_rows.append(
                    {
                        "family": family_name,
                        "scenario_name": scenario_name,
                        "fps": float(fps),
                        "readout_key": readout_key,
                        "readout_label": readout_label,
                        "accuracy_nonideal": accuracy,
                        "results_json": str(scenario_dir / "results.json"),
                        "manifest_path": str(manifest_path),
                        "input_path": sample_item["input_path"],
                        "nonideal_path": sample_item["nonideal_path"],
                        "compare_path": sample_item.get("compare_path"),
                        "psnr_db": sample_item.get("psnr_db"),
                        "dataset_index": sample_item["dataset_index"],
                        "label": sample_item["label"],
                    }
                )
    return scenario_rows


def save_aggregate_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "scenario_name",
        "fps",
        "readout_key",
        "readout_label",
        "accuracy_nonideal",
        "results_json",
        "manifest_path",
        "input_path",
        "nonideal_path",
        "compare_path",
        "psnr_db",
        "dataset_index",
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_bars(rows, output_path):
    family_titles = {
        "native": "Case2 Native",
        "x2_enhanced": "Case2 Enhanced x2",
    }
    colors = {
        "tia": "#d64550",
        "integration": "#2a9d8f",
        "adc4": "#457b9d",
        "adc8": "#8d6cab",
    }

    fps_values = sorted({float(row["fps"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.8), sharey=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    bar_width = 0.18
    centers = np.arange(len(fps_values), dtype=np.float64)

    for ax, family_name in zip(axes, ["native", "x2_enhanced"]):
        family_rows = [row for row in rows if row["family"] == family_name]
        for offset_idx, (readout_key, readout_label) in enumerate(READOUT_GROUPS):
            heights = []
            for fps in fps_values:
                match = next(
                    row for row in family_rows if row["readout_key"] == readout_key and abs(row["fps"] - fps) <= 1e-9
                )
                heights.append(float(match["accuracy_nonideal"]))
            x = centers + (offset_idx - 1.5) * bar_width
            ax.bar(
                x,
                heights,
                width=bar_width,
                color=colors[readout_key],
                edgecolor="#222222",
                linewidth=1.2,
                label=readout_label,
            )

        ax.set_xticks(centers)
        ax.set_xticklabels([f"FPS {int(fps)}" for fps in fps_values], fontsize=12, fontweight="bold")
        ax.set_xlabel("Video Frame Rate", fontsize=13, fontweight="bold")
        ax.set_ylabel("Accuracy on 200 CIFAR-10 Test Images (%)", fontsize=13, fontweight="bold")
        ax.set_title(family_titles[family_name], fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
        ax.tick_params(axis="y", labelsize=11, width=1.4)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

    axes[1].legend(loc="upper right", fontsize=11, frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_pair_image(input_path, nonideal_path):
    input_img = Image.open(input_path).convert("RGB")
    nonideal_img = Image.open(nonideal_path).convert("RGB")
    tile_w, tile_h = input_img.size
    canvas = Image.new("RGB", (tile_w * 2, tile_h), (255, 255, 255))
    canvas.paste(input_img, (0, 0))
    canvas.paste(nonideal_img, (tile_w, 0))
    return np.asarray(canvas)


def plot_visual_grid(rows, family_name, output_path):
    fps_values = sorted({float(row["fps"]) for row in rows})
    family_rows = [row for row in rows if row["family"] == family_name]
    fig, axes = plt.subplots(len(fps_values), len(READOUT_GROUPS), figsize=(15.5, 10.8), constrained_layout=True)
    if len(fps_values) == 1:
        axes = np.asarray([axes])

    for row_idx, fps in enumerate(fps_values):
        for col_idx, (readout_key, readout_label) in enumerate(READOUT_GROUPS):
            ax = axes[row_idx, col_idx]
            scenario = next(
                row for row in family_rows if row["readout_key"] == readout_key and abs(row["fps"] - fps) <= 1e-9
            )
            pair = make_pair_image(scenario["input_path"], scenario["nonideal_path"])
            ax.imshow(pair)
            ax.set_xticks([])
            ax.set_yticks([])
            title = f"FPS {int(fps)} | {readout_label}\nAcc={scenario['accuracy_nonideal']:.2f}%"
            if scenario["psnr_db"] is not None:
                title += f" | PSNR={float(scenario['psnr_db']):.2f} dB"
            ax.set_title(title, fontsize=10, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("#222222")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    derived_summary = derive_case2_eval_csvs(args, output_dir)
    scenario_rows = run_all_scenarios(args, derived_summary)

    aggregate_csv = output_dir / "aggregate_results.csv"
    save_aggregate_csv(scenario_rows, aggregate_csv)

    bar_chart_path = output_dir / "case2_native_vs_x2_accuracy_bars.png"
    plot_accuracy_bars(scenario_rows, bar_chart_path)

    native_visual_path = output_dir / "case2_native_visual_comparisons.png"
    enhanced_visual_path = output_dir / "case2_x2_enhanced_visual_comparisons.png"
    plot_visual_grid(scenario_rows, "native", native_visual_path)
    plot_visual_grid(scenario_rows, "x2_enhanced", enhanced_visual_path)

    summary = {
        "params_csv": str(Path(args.params_csv).expanduser()),
        "aggregate_csv": str(aggregate_csv),
        "bar_chart_path": str(bar_chart_path),
        "native_visual_path": str(native_visual_path),
        "x2_enhanced_visual_path": str(enhanced_visual_path),
        "derived_summary": derived_summary,
        "fps_values": [float(value) for value in args.fps],
        "readout_groups": [label for _, label in READOUT_GROUPS],
        "visual_index": int(args.visual_index),
        "eval_images": int(args.batch_size * args.max_eval_batches),
        "noise_disabled": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Aggregate CSV: {aggregate_csv}", flush=True)
    print(f"Bar chart: {bar_chart_path}", flush=True)
    print(f"Native visuals: {native_visual_path}", flush=True)
    print(f"Enhanced visuals: {enhanced_visual_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
