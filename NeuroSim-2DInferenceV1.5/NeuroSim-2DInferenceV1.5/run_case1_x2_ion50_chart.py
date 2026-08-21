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
import generate_fig3_arbitrary_case1_on_state_drift as fig3_case1  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402
import photodetector_model as pm  # noqa: E402


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a case1 single-group experiment where x2 is set to about 50% of Ion at pmax, "
            "then render a native-style bar chart over FPS 20/50/100/200."
        )
    )
    parser.add_argument(
        "--base-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence"),
        help="Reference native case1 experiment directory for shared runtime settings.",
    )
    parser.add_argument(
        "--results-dir",
        default="auto",
        help="Directory for the new x2=50% Ion experiment.",
    )
    parser.add_argument(
        "--chart-output",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence_eta0p2" / "case1_native_vs_x2_accuracy_bars.png"),
        help="PNG path for the redrawn single-group chart.",
    )
    parser.add_argument(
        "--params-csv",
        default=None,
        help="Optional explicit parameter CSV for the single-group experiment. If omitted, the built-in x2~50%Ion parameters are used.",
    )
    parser.add_argument(
        "--chart-title",
        default="Case1 Native + x2 at ~50% of Ion",
        help="Chart title for the rendered bar plot.",
    )
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def resolve_results_dir(args):
    if args.results_dir not in {None, "", "auto"}:
        return Path(args.results_dir).expanduser()
    return THIS_DIR / "artifacts" / "case1_x2_ion50_video_sequence"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parameter_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


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
    ]:
        if field in summary["args"] and hasattr(base_args, field):
            setattr(base_args, field, summary["args"][field])
    base_args.generate_images = False
    base_args.run_eval = True
    base_args.eval_cases = ["nonideal"]
    base_args.num_images = 0
    return base_args


def derive_x2_ion50_params(output_dir):
    case1_results, params_with_x2, _ = fig3_case1.build_case1_parameter_sets()
    native_params = dict(case1_results["params"])

    model_config_native = pm.prepare_model_config(
        pm.params_to_vec(native_params),
        n_carrier=1,
    )
    main_photo_pmax_a = float(
        pm.steady_state_current_from_power(
            np.asarray([pm.compute_total_power_from_density_w_cm2(5e-4, pm.DEVICE_AREA_CM2)], dtype=np.float64),
            model_config_native,
            dark_current=0.0,
            include_drift=False,
        )[0]
    )
    # Set delta so that when x2 is fully charged, x2 contributes about half of Ion at pmax:
    # Ion = I_main + I_x2, choose I_x2 ~= I_main.
    params = dict(native_params)
    params["alpha"] = float(params_with_x2["alpha"])
    params["beta"] = float(params_with_x2["beta"])
    params["delta"] = float(main_photo_pmax_a)
    params["device_area_cm2"] = float(pm.DEVICE_AREA_CM2)
    params["power_ref_w"] = float(case1_results["power_ref_w"])
    params["noise_1f_density_1hz_a_root_hz"] = 0.0
    params["noise_scale_vs_case1"] = 0.0

    ordered_keys = [
        "R_single",
        "eta_single",
        "tau_rise_single",
        "tau_fall_single",
        "alpha",
        "beta",
        "delta",
        "gamma",
        "tau_drift",
        "drift_scale",
        "device_area_cm2",
        "power_ref_w",
        "noise_1f_density_1hz_a_root_hz",
        "noise_scale_vs_case1",
    ]
    csv_path = output_dir / "params_case1_x2_ion50_no_noise.csv"
    write_parameter_csv(csv_path, [(key, params[key]) for key in ordered_keys])

    summary = {
        "params_csv": str(csv_path),
        "power_ref_w": float(case1_results["power_ref_w"]),
        "device_area_cm2": float(pm.DEVICE_AREA_CM2),
        "target_condition": "x2 current ~= main photocurrent at pmax, so x2 ~= 50% of Ion",
        "pmax_density_w_cm2": 5e-4,
        "pmax_w": float(pm.compute_total_power_from_density_w_cm2(5e-4, pm.DEVICE_AREA_CM2)),
        "main_photo_pmax_a": main_photo_pmax_a,
        "derived_params": {key: float(params[key]) for key in ordered_keys},
    }
    (output_dir / "parameter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary


def run_scenarios(base_args, params_csv, results_dir, resume):
    adc_window_cache = {}
    records = []
    for fps in FPS_VALUES:
        for readout_cfg in eval_cmp.READOUT_CONFIGS:
            cfg = copy.deepcopy(base_args)
            cfg.params_csv = str(params_csv)
            cfg.video_fps = float(fps)
            cfg.readout = readout_cfg["readout"]
            cfg.analog_readout = readout_cfg["analog_readout"]
            cfg.adc_enabled = int(readout_cfg["adc_enabled"])
            cfg.adc_bits = int(readout_cfg["adc_bits"])
            scenario_name = f"x2ion50_fps{int(fps):03d}_{readout_cfg['label']}"
            scenario_dir = results_dir / scenario_name
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

            results_json_path = Path(cfg.results_json)
            if bool(resume) and results_json_path.is_file():
                result_payload = json.loads(results_json_path.read_text(encoding="utf-8"))
            else:
                print(f"Run x2~50% Ion | FPS={int(fps)} | {readout_cfg['display']}", flush=True)
                result_payload = pipeline.run_sequence_pipeline(cfg)
            accuracy = eval_cmp.accuracy_from_result(result_payload)
            records.append(
                {
                    "video_fps": float(fps),
                    "readout_label": readout_cfg["label"],
                    "readout_display": readout_cfg["display"],
                    "accuracy_nonideal": float(accuracy),
                    "results_json": str(Path(cfg.results_json).resolve()),
                    "adc_calibration_low": cfg.adc_calibration_low,
                    "adc_calibration_high": cfg.adc_calibration_high,
                }
            )
    return records


def render_chart(records, output_path, chart_title):
    grouped = {(float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"]) for r in records}
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.4), constrained_layout=True)
    colors = {
        "tia": "#D55E00",
        "integration": "#0072B2",
        "adc4": "#009E73",
        "adc8": "#CC79A7",
    }
    x = np.arange(len(FPS_VALUES), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for readout_idx, readout_cfg in enumerate(eval_cmp.READOUT_CONFIGS):
        values = [grouped[(fps, readout_cfg["label"])] for fps in FPS_VALUES]
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

    ax.set_title(chart_title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"FPS {int(fps)}" for fps in FPS_VALUES], fontsize=11, fontweight="bold")
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
    results_dir = resolve_results_dir(args)
    chart_output = Path(args.chart_output).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(base_results_dir / "case1_native_vs_x2_accuracy_summary.json")
    base_args = build_base_args_from_summary(summary)
    if args.params_csv:
        params_csv = Path(args.params_csv).expanduser().resolve()
        param_summary = {
            "params_csv": str(params_csv),
            "source": "user_provided",
        }
    else:
        params_csv, param_summary = derive_x2_ion50_params(results_dir)
    records = run_scenarios(base_args, params_csv, results_dir, resume=args.resume)

    records_csv = results_dir / "case1_x2_ion50_accuracy_records.csv"
    with records_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    native_chart_path = results_dir / "case1_x2_ion50_accuracy_bars_fps20_200.png"
    render_chart(records, native_chart_path, chart_title=args.chart_title)
    render_chart(records, chart_output, chart_title=args.chart_title)

    summary_payload = {
        "base_results_dir": str(base_results_dir),
        "results_dir": str(results_dir),
        "chart_output": str(chart_output),
        "native_chart_path": str(native_chart_path),
        "records_csv": str(records_csv),
        "parameter_summary": param_summary,
    }
    summary_path = results_dir / "case1_x2_ion50_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Chart: {native_chart_path}", flush=True)
    print(f"Requested chart output: {chart_output}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
