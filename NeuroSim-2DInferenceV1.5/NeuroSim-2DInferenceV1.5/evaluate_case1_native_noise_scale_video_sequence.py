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


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original case1 native baseline under multiple flicker-noise scale "
            "factors and render one accuracy bar chart per noise condition."
        )
    )
    parser.add_argument(
        "--base-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence"),
        help="Existing original baseline results directory used as the runtime reference.",
    )
    parser.add_argument(
        "--results-dir",
        default="auto",
        help="Directory where the noise-scale sweep artifacts will be written.",
    )
    parser.add_argument(
        "--noise-scales",
        type=float,
        nargs="+",
        default=[10.0, 100.0, 1000.0],
        help="Noise multipliers relative to the case1 baseline PSD.",
    )
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def resolve_results_dir(args):
    if args.results_dir not in {None, "", "auto"}:
        return Path(args.results_dir).expanduser()
    return THIS_DIR / "artifacts" / "case1_native_noise_scale_sweep"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_scale_label(scale):
    scale = float(scale)
    if abs(scale - round(scale)) <= 1e-9:
        return f"x{int(round(scale))}"
    return f"x{scale:g}".replace(".", "p")


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
    base_args.use_noise_fn = 1
    return base_args


def load_existing_result(results_json_path):
    if not results_json_path.is_file():
        return None
    try:
        return load_json(results_json_path)
    except Exception:
        return None


def run_scenario(cfg, resume):
    results_json_path = Path(cfg.results_json)
    if resume:
        existing = load_existing_result(results_json_path)
        if existing is not None:
            return existing
    return pipeline.run_sequence_pipeline(cfg)


def save_records_csv(records, output_path):
    fieldnames = [
        "noise_scale_vs_case1",
        "noise_1f_density_1hz_a_root_hz",
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


def render_chart(records, output_path, chart_title):
    grouped = {(float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"]) for r in records}

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.8), constrained_layout=True)
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
    ax.legend(loc="upper right", fontsize=10, frameon=False, ncol=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(axis="y", labelsize=10, width=1.3)
    ax.tick_params(axis="x", width=1.3)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base_results_dir = Path(args.base_results_dir)
    results_dir = resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(base_results_dir / "case1_native_vs_x2_accuracy_summary.json")
    base_args = build_base_args_from_summary(summary)

    sweep_manifest = []
    for noise_scale in args.noise_scales:
        scale_label = format_scale_label(noise_scale)
        scale_dir = results_dir / f"noise_{scale_label}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        param_info = eval_cmp.build_param_csvs(
            scale_dir,
            x2_delta_scale=summary["args"].get("x2_delta_scale", 200.0),
            noise_enabled=True,
            noise_scale_vs_case1=noise_scale,
            eta_override=None,
        )
        native_params_csv = Path(param_info["native"]).resolve()
        noise_density_1hz = float(param_info["summary"]["noise_1f_density_1hz_a_root_hz"])
        records = []
        adc_window_cache = {}
        total = len(FPS_VALUES) * len(eval_cmp.READOUT_CONFIGS)
        counter = 0
        start_time = time.time()

        for fps in FPS_VALUES:
            for readout_cfg in eval_cmp.READOUT_CONFIGS:
                counter += 1
                scenario_name = f"native_noise_{scale_label}_fps{int(fps):03d}_{readout_cfg['label']}"
                scenario_dir = scale_dir / scenario_name
                scenario_dir.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(base_args)
                cfg.params_csv = str(native_params_csv)
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
                    f"[{counter}/{total}] Baseline noise {scale_label} | FPS={int(fps)} | {readout_cfg['display']}",
                    flush=True,
                )
                result_payload = run_scenario(cfg, resume=bool(args.resume))
                accuracy = eval_cmp.accuracy_from_result(result_payload)
                elapsed = time.time() - start_time
                print(f"  accuracy={accuracy:.2f}% elapsed={elapsed:.1f}s", flush=True)

                records.append(
                    {
                        "noise_scale_vs_case1": float(noise_scale),
                        "noise_1f_density_1hz_a_root_hz": noise_density_1hz,
                        "video_fps": float(fps),
                        "readout_label": readout_cfg["label"],
                        "readout_display": readout_cfg["display"],
                        "accuracy_nonideal": float(accuracy),
                        "results_json": str(Path(cfg.results_json).resolve()),
                        "adc_calibration_low": cfg.adc_calibration_low,
                        "adc_calibration_high": cfg.adc_calibration_high,
                    }
                )

        records_csv = scale_dir / f"case1_native_noise_{scale_label}_accuracy_records.csv"
        save_records_csv(records, records_csv)
        chart_output = scale_dir / f"case1_native_noise_{scale_label}_accuracy_bars.png"
        render_chart(
            records,
            chart_output,
            chart_title=(
                f"Case1 Native Baseline | Noise {scale_label}\n"
                f"flicker_1Hz={noise_density_1hz:.3e} A/Hz^0.5"
            ),
        )

        summary_payload = {
            "base_results_dir": str(base_results_dir),
            "results_dir": str(scale_dir),
            "noise_scale_vs_case1": float(noise_scale),
            "noise_1f_density_1hz_a_root_hz": noise_density_1hz,
            "native_params_csv": str(native_params_csv),
            "records_csv": str(records_csv),
            "chart_output": str(chart_output),
            "records": records,
        }
        summary_path = scale_dir / f"case1_native_noise_{scale_label}_summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        sweep_manifest.append(summary_payload)

        print(f"Chart: {chart_output}", flush=True)
        print(f"Summary JSON: {summary_path}", flush=True)

    manifest_path = results_dir / "case1_native_noise_scale_sweep_manifest.json"
    manifest_path.write_text(json.dumps(sweep_manifest, indent=2), encoding="utf-8")
    print(f"Sweep manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
