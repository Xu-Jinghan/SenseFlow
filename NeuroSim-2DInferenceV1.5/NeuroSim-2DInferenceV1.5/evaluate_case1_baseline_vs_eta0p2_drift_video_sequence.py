import argparse
import copy
import csv
import json
import sys
import time
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


PARAM_GROUPS = [
    {"key": "baseline", "display": "Case1 Native Baseline"},
    {"key": "eta0p2", "display": "Case1 Native eta=0.2"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CIFAR-10 video-sequence accuracy for the original native baseline "
            "versus the eta=0.2 native case under long-term drift, then render grouped bar charts."
        )
    )
    parser.add_argument(
        "--baseline-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence"),
        help="Existing original-eta native results directory.",
    )
    parser.add_argument(
        "--eta-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence_eta0p2"),
        help="Existing eta=0.2 native results directory.",
    )
    parser.add_argument(
        "--results-dir",
        default="auto",
        help="Directory where the drifted comparison artifacts will be written.",
    )
    parser.add_argument("--drift-hours", type=float, default=500.0)
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def resolve_results_dir(args):
    if args.results_dir not in {None, "", "auto"}:
        return Path(args.results_dir).expanduser()
    drift_tag = pipeline.format_drift_scenario_tag(args.drift_hours)
    return THIS_DIR / "artifacts" / f"case1_native_baseline_vs_eta0p2_{drift_tag}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_base_args_from_summary(summary, drift_hours):
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
    ]:
        if field in summary["args"] and hasattr(base_args, field):
            setattr(base_args, field, summary["args"][field])
    base_args.generate_images = False
    base_args.run_eval = True
    base_args.eval_cases = ["nonideal"]
    base_args.num_images = 0
    base_args.drift_hours = [float(drift_hours)]
    return base_args


def load_param_info(base_results_dir):
    summary = load_json(base_results_dir / "case1_native_vs_x2_accuracy_summary.json")
    parameter_summary = load_json(base_results_dir / "parameter_summary.json")
    params_csv = Path(parameter_summary["native_csv"]).resolve()
    eta_single = float(parameter_summary["native_params"]["eta_single"])
    return summary, params_csv, eta_single


def load_existing_result(results_json_path, expected_drift_hours):
    if not results_json_path.is_file():
        return None
    try:
        payload = load_json(results_json_path)
    except Exception:
        return None
    if abs(float(payload.get("drift_hours", 0.0)) - float(expected_drift_hours)) > 1e-9:
        return None
    return payload


def run_scenario(cfg, resume):
    results_json_path = Path(cfg.results_json)
    if resume:
        existing = load_existing_result(results_json_path, cfg.drift_hours[0])
        if existing is not None:
            return existing
    return pipeline.run_sequence_pipeline(cfg)


def render_accuracy_bar_chart(records, output_path, drift_hours):
    grouped = {}
    eta_by_group = {}
    for record in records:
        grouped[(record["param_group"], record["video_fps"], record["readout_label"])] = float(record["accuracy_nonideal"])
        eta_by_group[record["param_group"]] = float(record["eta_single"])

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.1), sharey=True, constrained_layout=True)
    colors = {
        "tia": "#D55E00",
        "integration": "#0072B2",
        "adc4": "#009E73",
        "adc8": "#CC79A7",
    }
    x = np.arange(len(eval_cmp.FPS_VALUES), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for ax, param_group in zip(axes, PARAM_GROUPS):
        for readout_idx, readout_cfg in enumerate(eval_cmp.READOUT_CONFIGS):
            values = [
                grouped[(param_group["key"], fps, readout_cfg["label"])]
                for fps in eval_cmp.FPS_VALUES
            ]
            bars = ax.bar(
                x + offsets[readout_idx],
                values,
                width=width,
                color=colors[readout_cfg["label"]],
                label=readout_cfg["display"],
                edgecolor="#222222",
                linewidth=1.0,
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

        eta_single = eta_by_group[param_group["key"]]
        ax.set_title(
            f"{param_group['display']}\neta={eta_single:.4f}, drift={float(drift_hours):.1f} h",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"FPS {int(fps)}" for fps in eval_cmp.FPS_VALUES], fontsize=11, fontweight="bold")
        ax.set_xlabel("Video FPS", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        ax.tick_params(axis="y", labelsize=10, width=1.3)
        ax.tick_params(axis="x", width=1.3)

    axes[0].set_ylabel("CIFAR-10 Accuracy (%) on 200 test images", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_records_csv(records, output_path):
    fieldnames = [
        "param_group",
        "param_group_display",
        "eta_single",
        "drift_hours",
        "video_fps",
        "readout_label",
        "readout_display",
        "accuracy_nonideal",
        "results_json",
        "adc_calibration_low",
        "adc_calibration_high",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    baseline_results_dir = Path(args.baseline_results_dir)
    eta_results_dir = Path(args.eta_results_dir)
    results_dir = resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary, baseline_params_csv, baseline_eta = load_param_info(baseline_results_dir)
    _eta_summary, eta_params_csv, eta_eta = load_param_info(eta_results_dir)
    base_args = build_base_args_from_summary(baseline_summary, drift_hours=args.drift_hours)

    param_groups = {
        "baseline": {
            "display": "Case1 Native Baseline",
            "params_csv": baseline_params_csv,
            "eta_single": baseline_eta,
        },
        "eta0p2": {
            "display": "Case1 Native eta=0.2",
            "params_csv": eta_params_csv,
            "eta_single": eta_eta,
        },
    }

    records = []
    adc_window_cache = {}
    total = len(PARAM_GROUPS) * len(eval_cmp.FPS_VALUES) * len(eval_cmp.READOUT_CONFIGS)
    counter = 0
    start_time = time.time()
    for param_group in PARAM_GROUPS:
        group_info = param_groups[param_group["key"]]
        for fps in eval_cmp.FPS_VALUES:
            for readout_cfg in eval_cmp.READOUT_CONFIGS:
                counter += 1
                scenario_name = eval_cmp.scenario_name(param_group["key"], fps, readout_cfg["label"])
                scenario_dir = results_dir / scenario_name
                scenario_dir.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(base_args)
                cfg.params_csv = str(group_info["params_csv"])
                cfg.video_fps = float(fps)
                cfg.readout = readout_cfg["readout"]
                cfg.analog_readout = readout_cfg["analog_readout"]
                cfg.adc_enabled = int(readout_cfg["adc_enabled"])
                cfg.adc_bits = int(readout_cfg["adc_bits"])
                cfg.results_json = str(scenario_dir / "results.json")
                cfg.output_dir = str(scenario_dir)
                if cfg.adc_enabled:
                    adc_low, adc_high = eval_cmp.estimate_adc_quantization_bounds(cfg, adc_window_cache)
                    cfg.adc_calibration_low = adc_low
                    cfg.adc_calibration_high = adc_high
                    cfg.adc_full_scale = None
                else:
                    cfg.adc_calibration_low = None
                    cfg.adc_calibration_high = None

                print(
                    f"[{counter}/{total}] {group_info['display']} | drift={float(args.drift_hours):.1f}h | "
                    f"FPS={int(fps)} | {readout_cfg['display']}",
                    flush=True,
                )
                result_payload = run_scenario(cfg, resume=bool(args.resume))
                accuracy = eval_cmp.accuracy_from_result(result_payload)
                elapsed = time.time() - start_time
                print(f"  accuracy={accuracy:.2f}% elapsed={elapsed:.1f}s", flush=True)

                records.append(
                    {
                        "param_group": param_group["key"],
                        "param_group_display": group_info["display"],
                        "eta_single": float(group_info["eta_single"]),
                        "drift_hours": float(args.drift_hours),
                        "video_fps": float(fps),
                        "readout_label": readout_cfg["label"],
                        "readout_display": readout_cfg["display"],
                        "accuracy_nonideal": float(accuracy),
                        "results_json": str(Path(cfg.results_json).resolve()),
                        "adc_calibration_low": cfg.adc_calibration_low,
                        "adc_calibration_high": cfg.adc_calibration_high,
                    }
                )

    records_csv = results_dir / "case1_baseline_vs_eta0p2_drift_accuracy_records.csv"
    save_records_csv(records, records_csv)

    chart_path = results_dir / "case1_baseline_vs_eta0p2_drift_accuracy_bars.png"
    render_accuracy_bar_chart(records, chart_path, drift_hours=args.drift_hours)

    summary_payload = {
        "baseline_results_dir": str(baseline_results_dir),
        "eta_results_dir": str(eta_results_dir),
        "results_dir": str(results_dir),
        "drift_hours": float(args.drift_hours),
        "baseline_params_csv": str(baseline_params_csv),
        "eta0p2_params_csv": str(eta_params_csv),
        "records_csv": str(records_csv),
        "chart_path": str(chart_path),
        "records": records,
    }
    summary_path = results_dir / "case1_baseline_vs_eta0p2_drift_accuracy_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Records CSV: {records_csv}", flush=True)
    print(f"Chart: {chart_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
