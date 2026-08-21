import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_case1_native_vs_x2_video_sequence as eval_cmp  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Supplement the native case1 experiment with FPS=200 and redraw a native-only "
            "bar chart containing FPS 20/50/100/200."
        )
    )
    parser.add_argument(
        "--base-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence"),
        help="Existing native-eta, no-noise case1 comparison directory.",
    )
    parser.add_argument(
        "--supplement-dir",
        default="auto",
        help="Directory for the newly run FPS=200 native scenarios.",
    )
    parser.add_argument(
        "--chart-output",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence_eta0p2" / "case1_native_vs_x2_accuracy_bars.png"),
        help="Where to write the redrawn native-only chart. This can overwrite an existing png if desired.",
    )
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def resolve_supplement_dir(args):
    if args.supplement_dir not in {None, "", "auto"}:
        return Path(args.supplement_dir).expanduser()
    return Path(args.base_results_dir) / "native_fps200_supplement"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_experiment(base_results_dir):
    summary = load_json(Path(base_results_dir) / "case1_native_vs_x2_accuracy_summary.json")
    parameter_summary = load_json(Path(base_results_dir) / "parameter_summary.json")
    return summary, parameter_summary


def build_base_args_from_summary(summary):
    args_ns = argparse.Namespace(**summary["args"])
    base_args = eval_cmp.build_base_args(args_ns)
    for field in [
        "range_calibration_samples",
        "seed",
        "sensor_rng_seed",
        "post_norm",
        "data_root",
        "source_dataset",
        "split",
        "fps_sim",
        "array_size",
        "target_size",
        "output_channels",
        "tile_size",
        "normalization_mode",
        "range_scope",
        "range_calibration_split",
        "use_noise_fn",
        "shot_noise",
        "i_thermal",
        "bandwidth",
        "batch_size",
        "max_eval_batches",
        "num_workers",
        "visual_start_index",
        "visual_num_images",
    ]:
        if field in summary["args"]:
            value = summary["args"][field]
            if hasattr(base_args, field):
                setattr(base_args, field, value)
    return base_args


def load_existing_records(csv_path):
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_native_fps200(base_args, native_params_csv, supplement_dir, resume):
    supplement_dir.mkdir(parents=True, exist_ok=True)
    adc_window_cache = {}
    records = []

    for readout_cfg in eval_cmp.READOUT_CONFIGS:
        cfg = copy.deepcopy(base_args)
        cfg.params_csv = str(native_params_csv)
        cfg.generate_images = True
        cfg.run_eval = True
        cfg.eval_cases = ["nonideal"]
        cfg.video_fps = 200.0
        cfg.readout = readout_cfg["readout"]
        cfg.analog_readout = readout_cfg["analog_readout"]
        cfg.adc_enabled = int(readout_cfg["adc_enabled"])
        cfg.adc_bits = int(readout_cfg["adc_bits"])
        cfg.results_json = None
        scenario_name = eval_cmp.scenario_name("native", 200.0, readout_cfg["label"])
        scenario_dir = supplement_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir = str(scenario_dir)
        cfg.results_json = str(scenario_dir / "results.json")

        if cfg.adc_enabled:
            adc_low, adc_high = eval_cmp.estimate_adc_quantization_bounds(cfg, adc_window_cache)
            cfg.adc_calibration_low = adc_low
            cfg.adc_calibration_high = adc_high
            cfg.adc_full_scale = None
        else:
            cfg.adc_calibration_low = None
            cfg.adc_calibration_high = None

        print(f"Supplement native FPS=200 | {readout_cfg['display']}", flush=True)
        result_payload, manifest_payload = eval_cmp.run_scenario(cfg, resume=bool(resume))
        accuracy = eval_cmp.accuracy_from_result(result_payload)
        sample_paths = eval_cmp.first_sample_paths(manifest_payload)
        records.append(
            {
                "param_group": "native",
                "param_group_display": "Case1 Native",
                "video_fps": 200.0,
                "readout_label": readout_cfg["label"],
                "readout_display": readout_cfg["display"],
                "accuracy_nonideal": float(accuracy),
                "results_json": str((scenario_dir / "results.json").resolve()) if (scenario_dir / "results.json").is_file() else "",
                "manifest_path": str((scenario_dir / "manifest.json").resolve()),
                "sample_index": sample_paths["dataset_index"],
                "sample_label": sample_paths["label"],
                "adc_calibration_low": cfg.adc_calibration_low,
                "adc_calibration_high": cfg.adc_calibration_high,
                "input_path": sample_paths["input_path"],
                "nonideal_path": sample_paths["nonideal_path"],
                "compare_path": sample_paths["compare_path"],
            }
        )
    return records


def render_native_only_chart(records, output_path):
    readout_order = eval_cmp.READOUT_CONFIGS
    fps_values = [20.0, 50.0, 100.0, 200.0]
    grouped = {(float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"]) for r in records}

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.4), constrained_layout=True)
    colors = {
        "tia": "#D55E00",
        "integration": "#0072B2",
        "adc4": "#009E73",
        "adc8": "#CC79A7",
    }
    x = np.arange(len(fps_values), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for readout_idx, readout_cfg in enumerate(readout_order):
        values = [grouped[(fps, readout_cfg["label"])] for fps in fps_values]
        bars = ax.bar(
            x + offsets[readout_idx],
            values,
            width=width,
            color=colors[readout_cfg["label"]],
            edgecolor="#222222",
            linewidth=1.0,
            label=readout_cfg["display"],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                value + 0.35,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    ax.set_title("Case1 Native", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"FPS {int(fps)}" for fps in fps_values], fontsize=11, fontweight="bold")
    ax.set_xlabel("Video FPS", fontsize=12, fontweight="bold")
    ax.set_ylabel("CIFAR-10 Accuracy (%) on 200 test images", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(axis="y", labelsize=10, width=1.3)
    ax.tick_params(axis="x", width=1.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base_results_dir = Path(args.base_results_dir)
    supplement_dir = resolve_supplement_dir(args)
    chart_output = Path(args.chart_output).expanduser()

    summary, parameter_summary = load_experiment(base_results_dir)
    base_args = build_base_args_from_summary(summary)
    native_params_csv = Path(parameter_summary["native_csv"]).resolve()

    fps200_records = run_native_fps200(base_args, native_params_csv, supplement_dir, resume=args.resume)

    existing_records = load_existing_records(base_results_dir / "case1_native_vs_x2_accuracy_records.csv")
    native_records = [record for record in existing_records if record["param_group"] == "native"]
    merged_records = [record for record in native_records if float(record["video_fps"]) in {20.0, 50.0, 100.0}]
    merged_records.extend(fps200_records)

    supplement_csv = supplement_dir / "native_fps20_200_accuracy_records.csv"
    with supplement_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_records[0].keys()))
        writer.writeheader()
        writer.writerows(merged_records)

    native_chart_path = supplement_dir / "case1_native_accuracy_bars_fps20_200.png"
    render_native_only_chart(merged_records, native_chart_path)
    render_native_only_chart(merged_records, chart_output)

    summary_payload = {
        "base_results_dir": str(base_results_dir),
        "supplement_dir": str(supplement_dir),
        "chart_output": str(chart_output),
        "native_chart_path": str(native_chart_path),
        "records_csv": str(supplement_csv),
        "fps200_records": fps200_records,
    }
    summary_path = supplement_dir / "fps200_native_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Native chart: {native_chart_path}", flush=True)
    print(f"Requested chart output: {chart_output}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
