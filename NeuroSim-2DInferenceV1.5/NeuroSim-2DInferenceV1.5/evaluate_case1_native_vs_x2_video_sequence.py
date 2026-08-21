import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_fig3_arbitrary_case1_on_state_drift as fig3_case1  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402
import photodetector_model as pm  # noqa: E402


FPS_VALUES = [20.0, 50.0, 100.0]
READOUT_CONFIGS = [
    {"label": "tia", "display": "TIA", "readout": "tia", "analog_readout": "tia", "adc_enabled": 0, "adc_bits": 8},
    {
        "label": "integration",
        "display": "Integration",
        "readout": "integration",
        "analog_readout": "integration",
        "adc_enabled": 0,
        "adc_bits": 8,
    },
    {
        "label": "adc4",
        "display": "ADC 4-bit",
        "readout": "integration",
        "analog_readout": "integration",
        "adc_enabled": 1,
        "adc_bits": 4,
    },
    {
        "label": "adc8",
        "display": "ADC 8-bit",
        "readout": "integration",
        "analog_readout": "integration",
        "adc_enabled": 1,
        "adc_bits": 8,
    },
]
PARAM_GROUPS = [
    {"key": "native", "display": "Case1 Native"},
    {"key": "x2_enhanced", "display": "Case1 + Enhanced x2"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CIFAR-10 video-sequence accuracy for native case1 parameters versus "
            "an x2-enhanced case1 parameter set, then render grouped bar charts and "
            "input/nonideal comparison panels."
        )
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--results-dir", default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=10, help="10 batches * batch_size 20 = 200 samples")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--post-norm", default="auto", choices=["none", "auto", "cifar10", "cifar100", "imagenet"])
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--range-calibration-samples", type=int, default=100)
    parser.add_argument("--visual-start-index", type=int, default=23)
    parser.add_argument("--visual-num-images", type=int, default=1)
    parser.add_argument(
        "--x2-delta-scale",
        type=float,
        default=200.0,
        help="Extra multiplier applied to the updated fig3 case1 delta term for the x2-enhanced parameter set.",
    )
    parser.add_argument(
        "--eta-override",
        type=float,
        default=None,
        help="Optional override applied to eta_single for both native and x2-enhanced case1 parameters.",
    )
    parser.add_argument(
        "--noise-enabled",
        type=int,
        default=0,
        help="Enable the video-sequence temporal noise trace for all scenarios.",
    )
    parser.add_argument(
        "--noise-scale-vs-case1",
        type=float,
        default=0.0,
        help="Case1 PSD scale factor used to derive noise_1f_density_1hz_a_root_hz.",
    )
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def write_parameter_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key, value in rows:
            writer.writerow([key, value])


def build_param_csvs(output_dir, x2_delta_scale, noise_enabled=False, noise_scale_vs_case1=0.0, eta_override=None):
    case1_results, params_with_x2, _ = fig3_case1.build_case1_parameter_sets()
    native_params = dict(case1_results["params"])
    x2_params = dict(params_with_x2)
    x2_params["delta"] = float(x2_params["delta"]) * float(x2_delta_scale)
    if eta_override is not None:
        native_params["eta_single"] = float(eta_override)
        x2_params["eta_single"] = float(eta_override)

    noise_enabled = bool(noise_enabled)
    noise_scale_vs_case1 = float(noise_scale_vs_case1)
    if noise_enabled and noise_scale_vs_case1 > 0.0:
        scaled_density = pm.scale_noise_density_components(
            case1_results["dataset"]["noise_freq_hz"],
            case1_results["dataset"]["noise_density"],
            white_scale=noise_scale_vs_case1,
            flicker_scale=noise_scale_vs_case1,
        )
        noise_density_1hz = float(
            pm._resample_noise_density(
                np.asarray([1.0], dtype=np.float64),
                case1_results["dataset"]["noise_freq_hz"],
                scaled_density,
            )[0]
        )
    else:
        noise_density_1hz = 0.0

    def finalize_params(params):
        merged = dict(params)
        merged.update(
            {
                "device_area_cm2": float(pm.DEVICE_AREA_CM2),
                "power_ref_w": float(case1_results["power_ref_w"]),
                "noise_1f_density_1hz_a_root_hz": float(noise_density_1hz),
                "noise_scale_vs_case1": float(noise_scale_vs_case1 if noise_enabled else 0.0),
            }
        )
        return merged

    native_params = finalize_params(native_params)
    x2_params = finalize_params(x2_params)

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

    suffix = "noise" if noise_enabled else "no_noise"
    native_csv = output_dir / f"params_case1_native_{suffix}.csv"
    x2_csv = output_dir / f"params_case1_x2_enhanced_{suffix}.csv"
    write_parameter_csv(native_csv, [(key, native_params[key]) for key in ordered_keys])
    write_parameter_csv(x2_csv, [(key, x2_params[key]) for key in ordered_keys])

    summary = {
        "native_csv": str(native_csv),
        "x2_enhanced_csv": str(x2_csv),
        "power_ref_w": float(case1_results["power_ref_w"]),
        "device_area_cm2": float(pm.DEVICE_AREA_CM2),
        "native_params": {key: float(native_params[key]) for key in ordered_keys if key in native_params},
        "x2_enhanced_params": {key: float(x2_params[key]) for key in ordered_keys if key in x2_params},
        "x2_delta_scale": float(x2_delta_scale),
        "eta_override": None if eta_override is None else float(eta_override),
        "noise_enabled": bool(noise_enabled),
        "noise_scale_vs_case1": float(noise_scale_vs_case1 if noise_enabled else 0.0),
        "noise_1f_density_1hz_a_root_hz": float(noise_density_1hz),
    }
    (output_dir / "parameter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "native": native_csv,
        "x2_enhanced": x2_csv,
        "summary": summary,
    }


def build_base_args(args):
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
        model_path=args.model_path,
        num_classes=0,
        results_json=None,
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=args.target_size,
        output_channels=args.output_channels,
        post_norm=args.post_norm,
        num_images=args.visual_num_images,
        start_index=args.visual_start_index,
        array_size=args.array_size,
        tile_size=args.tile_size,
        readout="integration",
        analog_readout="integration",
        adc_enabled=0,
        power_max=pipeline.base_pipeline.DEFAULT_POWER_MAX_W,
        params_csv="",
        normalization_mode="calibration",
        prange1_density=None,
        prange2_density=None,
        pmin_density=None,
        pmax_density=None,
        device_area_cm2=None,
        force_single_carrier=0,
        single_r=None,
        single_eta=None,
        single_trise=None,
        single_tfall=None,
        trap_saturation_time=None,
        trap_amplitude_pct=None,
        noise_1f_density_1hz=None,
        aging_tau_hours=None,
        r_degradation_pct=None,
        spatial_variation_r_pct=None,
        tia_gain_ohm=None,
        integration_gain_v_per_c=None,
        video_fps=20.0,
        fps_sim=args.fps_sim,
        adc_bits=8,
        adc_full_scale=None,
        adc_calibration_low=None,
        adc_calibration_high=None,
        range_mode="auto",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=args.range_calibration_samples,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=0,
        use_noise_fn=int(bool(args.noise_enabled)),
        startup_dark_frames=0,
        output_dir=None,
        analyze_center_pixel=0,
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def scenario_name(param_group, fps, readout_label):
    return f"{param_group}_fps{int(fps):03d}_{readout_label}"


def estimate_adc_quantization_bounds(cfg, cache):
    cache_key = (
        str(cfg.params_csv),
        float(cfg.video_fps),
        str(cfg.analog_readout),
        str(cfg.range_calibration_split),
        int(cfg.range_calibration_samples),
        float(cfg.percentile_low),
        float(cfg.percentile_high),
    )
    if cache_key in cache:
        return cache[cache_key]

    adc_calib_cfg = copy.deepcopy(cfg)
    adc_calib_cfg.adc_enabled = 0
    adc_calib_cfg.readout = adc_calib_cfg.analog_readout or adc_calib_cfg.readout
    adc_calib_cfg.adc_calibration_low = None
    adc_calib_cfg.adc_calibration_high = None
    base_params = pipeline.resolve_sequence_base_params(adc_calib_cfg)
    if getattr(adc_calib_cfg, "tia_gain_ohm", None) is None:
        adc_calib_cfg.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if getattr(adc_calib_cfg, "integration_gain_v_per_c", None) is None:
        adc_calib_cfg.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))

    calibration_dataset = pipeline.base_pipeline.load_base_dataset(
        adc_calib_cfg.source_dataset,
        adc_calib_cfg.data_root,
        adc_calib_cfg.range_calibration_split,
    )
    bounds = pipeline.compute_case_range_bounds(
        args=adc_calib_cfg,
        calibration_dataset=calibration_dataset,
        base_params=base_params,
        cases=["nonideal"],
    )
    low = float(bounds["nonideal"]["low"])
    high = float(bounds["nonideal"]["high"])
    cache[cache_key] = (low, high)
    return cache[cache_key]


def load_existing_result(results_json_path, manifest_path):
    if not results_json_path.is_file() or not manifest_path.is_file():
        return None
    try:
        results_payload = json.loads(results_json_path.read_text(encoding="utf-8"))
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return results_payload, manifest_payload


def run_scenario(cfg, resume):
    results_json_path = Path(cfg.results_json)
    manifest_path = Path(cfg.output_dir) / "manifest.json"
    if resume:
        existing = load_existing_result(results_json_path, manifest_path)
        if existing is not None:
            return existing

    result_payload = pipeline.run_sequence_pipeline(cfg)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return result_payload, manifest_payload


def accuracy_from_result(result_payload):
    return float(((((result_payload.get("evaluation") or {}).get("cases") or {}).get("nonideal")) or {})["accuracy"])


def first_sample_paths(manifest_payload):
    samples = manifest_payload.get("samples") or []
    if not samples:
        raise ValueError("Manifest did not contain exported samples.")
    sample = samples[0]
    return {
        "dataset_index": int(sample["dataset_index"]),
        "label": sample["label"],
        "input_path": sample["input_path"],
        "nonideal_path": sample["nonideal_path"],
        "compare_path": sample.get("compare_path"),
    }


def save_records_csv(records, output_path):
    fieldnames = [
        "param_group",
        "param_group_display",
        "video_fps",
        "readout_label",
        "readout_display",
        "accuracy_nonideal",
        "results_json",
        "manifest_path",
        "sample_index",
        "sample_label",
        "adc_calibration_low",
        "adc_calibration_high",
        "input_path",
        "nonideal_path",
        "compare_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def render_accuracy_bar_chart(records, output_path):
    grouped = {}
    for record in records:
        grouped[(record["param_group"], record["video_fps"], record["readout_label"])] = float(record["accuracy_nonideal"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), sharey=True, constrained_layout=True)
    colors = {
        "tia": "#D55E00",
        "integration": "#0072B2",
        "adc4": "#009E73",
        "adc8": "#CC79A7",
    }
    x = np.arange(len(FPS_VALUES), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for ax, param_group in zip(axes, PARAM_GROUPS):
        for readout_idx, readout_cfg in enumerate(READOUT_CONFIGS):
            values = [
                grouped[(param_group["key"], fps, readout_cfg["label"])]
                for fps in FPS_VALUES
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

        ax.set_title(param_group["display"], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"FPS {int(fps)}" for fps in FPS_VALUES], fontsize=11, fontweight="bold")
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


def open_rgb_image(path):
    return Image.open(path).convert("RGB")


def render_visual_panel(records, param_group_key, output_path):
    filtered = [record for record in records if record["param_group"] == param_group_key]
    lookup = {(record["video_fps"], record["readout_label"]): record for record in filtered}

    n_rows = len(FPS_VALUES)
    n_cols = len(READOUT_CONFIGS) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.15 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.asarray(axes).reshape(1, n_cols)

    for row_idx, fps in enumerate(FPS_VALUES):
        input_record = lookup[(fps, READOUT_CONFIGS[0]["label"])]
        input_img = open_rgb_image(input_record["input_path"])
        axes[row_idx, 0].imshow(input_img)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[row_idx, 0].set_ylabel(f"FPS {int(fps)}", fontsize=12, fontweight="bold")

        for col_idx, readout_cfg in enumerate(READOUT_CONFIGS, start=1):
            record = lookup[(fps, readout_cfg["label"])]
            nonideal_img = open_rgb_image(record["nonideal_path"])
            axes[row_idx, col_idx].imshow(nonideal_img)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(readout_cfg["display"], fontsize=12, fontweight="bold")
            axes[row_idx, col_idx].set_xlabel(f"Acc {record['accuracy_nonideal']:.1f}%", fontsize=10, fontweight="bold")

        for col_idx in range(n_cols):
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_linewidth(1.5)

    group_name = next(group["display"] for group in PARAM_GROUPS if group["key"] == param_group_key)
    fig.suptitle(f"{group_name}: Input vs Non-Ideal", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    param_info = build_param_csvs(
        results_dir,
        x2_delta_scale=args.x2_delta_scale,
        noise_enabled=bool(args.noise_enabled),
        noise_scale_vs_case1=args.noise_scale_vs_case1,
        eta_override=args.eta_override,
    )
    base_args = build_base_args(args)

    records = []
    adc_window_cache = {}
    total = len(PARAM_GROUPS) * len(FPS_VALUES) * len(READOUT_CONFIGS)
    counter = 0
    start_time = time.time()
    for param_group in PARAM_GROUPS:
        for fps in FPS_VALUES:
            for readout_cfg in READOUT_CONFIGS:
                counter += 1
                name = scenario_name(param_group["key"], fps, readout_cfg["label"])
                scenario_dir = results_dir / name
                scenario_dir.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(base_args)
                cfg.params_csv = str(param_info[param_group["key"]])
                cfg.video_fps = float(fps)
                cfg.readout = readout_cfg["readout"]
                cfg.analog_readout = readout_cfg["analog_readout"]
                cfg.adc_enabled = int(readout_cfg["adc_enabled"])
                cfg.adc_bits = int(readout_cfg["adc_bits"])
                if cfg.adc_enabled:
                    adc_low, adc_high = estimate_adc_quantization_bounds(cfg, adc_window_cache)
                    cfg.adc_calibration_low = adc_low
                    cfg.adc_calibration_high = adc_high
                    cfg.adc_full_scale = None
                else:
                    cfg.adc_calibration_low = None
                    cfg.adc_calibration_high = None
                cfg.results_json = str(scenario_dir / "results.json")
                cfg.output_dir = str(scenario_dir)

                print(
                    f"[{counter}/{total}] {param_group['display']} | FPS={int(fps)} | {readout_cfg['display']}",
                    flush=True,
                )
                result_payload, manifest_payload = run_scenario(cfg, resume=bool(args.resume))
                accuracy = accuracy_from_result(result_payload)
                sample_paths = first_sample_paths(manifest_payload)
                elapsed = time.time() - start_time
                print(
                    f"  accuracy={accuracy:.2f}% sample={sample_paths['dataset_index']}:{sample_paths['label']} elapsed={elapsed:.1f}s",
                    flush=True,
                )

                records.append(
                    {
                        "param_group": param_group["key"],
                        "param_group_display": param_group["display"],
                        "video_fps": float(fps),
                        "readout_label": readout_cfg["label"],
                        "readout_display": readout_cfg["display"],
                        "accuracy_nonideal": float(accuracy),
                        "results_json": str(Path(cfg.results_json).resolve()),
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

    records_csv = results_dir / "case1_native_vs_x2_accuracy_records.csv"
    save_records_csv(records, records_csv)
    summary_json = results_dir / "case1_native_vs_x2_accuracy_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "args": vars(args),
                "parameter_files": param_info["summary"],
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bar_chart_path = results_dir / "case1_native_vs_x2_accuracy_bars.png"
    render_accuracy_bar_chart(records, bar_chart_path)

    native_panel_path = results_dir / "case1_native_input_vs_nonideal.png"
    x2_panel_path = results_dir / "case1_x2_enhanced_input_vs_nonideal.png"
    render_visual_panel(records, "native", native_panel_path)
    render_visual_panel(records, "x2_enhanced", x2_panel_path)

    manifest = {
        "records_csv": str(records_csv),
        "summary_json": str(summary_json),
        "bar_chart": str(bar_chart_path),
        "native_visual_panel": str(native_panel_path),
        "x2_visual_panel": str(x2_panel_path),
        "results_dir": str(results_dir),
    }
    manifest_path = results_dir / "artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Records CSV: {records_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Bar chart: {bar_chart_path}", flush=True)
    print(f"Native visuals: {native_panel_path}", flush=True)
    print(f"X2 visuals: {x2_panel_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
