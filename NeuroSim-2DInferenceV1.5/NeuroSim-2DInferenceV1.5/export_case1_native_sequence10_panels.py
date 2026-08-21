import argparse
import copy
import json
import sys
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

import evaluate_case1_native_vs_x2_video_sequence as eval_cmp  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402


PAPER_TEXT_SCALE = 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export one native-case panel per sample for 10 consecutive video-sequence cases "
            "using the same FPS/readout matrix as case1_native_input_vs_nonideal.png."
        )
    )
    parser.add_argument(
        "--base-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence_eta0p2"),
        help="Existing native-vs-x2 eta=0.2 experiment directory used as the reference configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="auto",
        help="Directory for the new 10-sample scenario exports and composed panels.",
    )
    parser.add_argument("--start-index", type=int, default=23)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--drift-hours", type=float, default=0.0)
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def format_drift_tag(drift_hours):
    drift_hours = float(drift_hours)
    if abs(drift_hours - round(drift_hours)) <= 1e-9:
        return f"drift_{int(round(drift_hours)):04d}h"
    return f"drift_{drift_hours:.1f}h".replace(".", "p")


def resolve_output_dir(args):
    if args.output_dir not in {None, "", "auto"}:
        return Path(args.output_dir).expanduser()
    drift_prefix = "" if abs(float(args.drift_hours)) <= 1e-12 else f"{format_drift_tag(args.drift_hours)}_"
    return Path(args.base_results_dir) / (
        f"{drift_prefix}native_sequence_{args.start_index:04d}_{args.start_index + args.num_images - 1:04d}"
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_experiment_config(base_results_dir):
    summary = load_json(Path(base_results_dir) / "case1_native_vs_x2_accuracy_summary.json")
    param_summary = load_json(Path(base_results_dir) / "parameter_summary.json")
    args_ns = SimpleNamespace(**summary["args"])
    return args_ns, param_summary


def manifest_is_valid(manifest_path, expected_num_images, expected_start_index, expected_drift_hours):
    if not manifest_path.is_file():
        return False
    try:
        payload = load_json(manifest_path)
    except Exception:
        return False
    samples = payload.get("samples") or []
    if len(samples) != expected_num_images:
        return False
    expected_indices = list(range(expected_start_index, expected_start_index + expected_num_images))
    actual_indices = [int(sample.get("dataset_index", -1)) for sample in samples]
    if actual_indices != expected_indices:
        return False
    payload_drift_hours = float(payload.get("drift_hours", 0.0))
    if isinstance(expected_drift_hours, (list, tuple)):
        expected_drift_hours = expected_drift_hours[0] if expected_drift_hours else 0.0
    if abs(payload_drift_hours - float(expected_drift_hours)) > 1e-9:
        return False
    return all("psnr_db" in sample for sample in samples)


def build_native_export_configs(base_args, native_params_csv, output_dir, start_index, num_images):
    configs = []
    adc_window_cache = {}
    for fps in eval_cmp.FPS_VALUES:
        for readout_cfg in eval_cmp.READOUT_CONFIGS:
            cfg = copy.deepcopy(base_args)
            cfg.params_csv = str(native_params_csv)
            cfg.generate_images = True
            cfg.run_eval = False
            cfg.eval_cases = ["nonideal"]
            cfg.num_images = int(num_images)
            cfg.start_index = int(start_index)
            cfg.video_fps = float(fps)
            cfg.readout = readout_cfg["readout"]
            cfg.analog_readout = readout_cfg["analog_readout"]
            cfg.adc_enabled = int(readout_cfg["adc_enabled"])
            cfg.adc_bits = int(readout_cfg["adc_bits"])
            cfg.results_json = None

            scenario_name = eval_cmp.scenario_name("native", fps, readout_cfg["label"])
            scenario_dir = output_dir / "scenarios" / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            cfg.output_dir = str(scenario_dir)

            if cfg.adc_enabled:
                adc_low, adc_high = eval_cmp.estimate_adc_quantization_bounds(cfg, adc_window_cache)
                cfg.adc_calibration_low = adc_low
                cfg.adc_calibration_high = adc_high
                cfg.adc_full_scale = None
            else:
                cfg.adc_calibration_low = None
                cfg.adc_calibration_high = None

            configs.append((fps, readout_cfg, cfg, scenario_dir))
    return configs


def run_exports(configs, resume):
    manifests = {}
    for fps, readout_cfg, cfg, scenario_dir in configs:
        scenario_key = (fps, readout_cfg["label"])
        manifest_path = scenario_dir / "manifest.json"
        if resume and manifest_is_valid(manifest_path, cfg.num_images, cfg.start_index, cfg.drift_hours):
            manifest_payload = load_json(manifest_path)
        else:
            print(
                f"Export native visuals | FPS={int(fps)} | {readout_cfg['display']} | "
                f"start={cfg.start_index} num_images={cfg.num_images}",
                flush=True,
            )
            pipeline.run_sequence_pipeline(cfg)
            manifest_payload = load_json(manifest_path)
        manifests[scenario_key] = {
            "display": readout_cfg["display"],
            "manifest": manifest_payload,
        }
    return manifests


def open_rgb(path):
    return Image.open(path).convert("RGB")


def format_psnr(psnr_db):
    return f"PSNR {float(psnr_db):.2f} dB"


def build_sample_lookup(manifests):
    lookup = {}
    for scenario_key, payload in manifests.items():
        sample_map = {}
        for sample in payload["manifest"].get("samples") or []:
            sample_map[int(sample["dataset_index"])] = sample
        lookup[scenario_key] = {
            "display": payload["display"],
            "samples": sample_map,
        }
    return lookup


def compose_panel_for_sample(sample_index, sample_label, scenario_lookup, output_path, panel_title_prefix):
    n_rows = len(eval_cmp.FPS_VALUES)
    n_cols = len(eval_cmp.READOUT_CONFIGS) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.6 * n_cols, 5.8 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.asarray(axes).reshape(1, n_cols)

    for row_idx, fps in enumerate(eval_cmp.FPS_VALUES):
        # Reuse the input image from the first readout; it is the same across scenarios.
        reference_sample = scenario_lookup[(fps, eval_cmp.READOUT_CONFIGS[0]["label"])]["samples"][sample_index]
        axes[row_idx, 0].imshow(open_rgb(reference_sample["input_path"]))
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 0].set_title(
                "Input Image",
                fontsize=12 * PAPER_TEXT_SCALE,
                fontweight="bold",
                pad=22,
            )
        axes[row_idx, 0].set_ylabel(
            f"FPS {int(fps)}",
            fontsize=12 * PAPER_TEXT_SCALE,
            fontweight="bold",
            labelpad=18,
        )

        for col_idx, readout_cfg in enumerate(eval_cmp.READOUT_CONFIGS, start=1):
            scenario_sample = scenario_lookup[(fps, readout_cfg["label"])]["samples"][sample_index]
            axes[row_idx, col_idx].imshow(open_rgb(scenario_sample["nonideal_path"]))
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            psnr_label = format_psnr(scenario_sample["psnr_db"])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    f"{readout_cfg['display']}\n{psnr_label}",
                    fontsize=11 * PAPER_TEXT_SCALE,
                    fontweight="bold",
                    pad=22,
                )
            else:
                axes[row_idx, col_idx].text(
                    0.5,
                    1.08,
                    psnr_label,
                    transform=axes[row_idx, col_idx].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10 * PAPER_TEXT_SCALE,
                    fontweight="bold",
                )

        for col_idx in range(n_cols):
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_linewidth(2.4)

    fig.suptitle(
        f"{panel_title_prefix} | sample {sample_index}: {sample_label}",
        fontsize=14 * PAPER_TEXT_SCALE,
        fontweight="bold",
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base_results_dir = Path(args.base_results_dir)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_args, param_summary = load_experiment_config(base_results_dir)
    base_args = eval_cmp.build_base_args(reference_args)
    base_args.range_calibration_samples = int(getattr(reference_args, "range_calibration_samples", base_args.range_calibration_samples))
    base_args.seed = int(getattr(reference_args, "seed", base_args.seed))
    base_args.sensor_rng_seed = int(getattr(reference_args, "sensor_rng_seed", base_args.sensor_rng_seed))
    base_args.post_norm = getattr(reference_args, "post_norm", base_args.post_norm)
    base_args.data_root = getattr(reference_args, "data_root", base_args.data_root)
    base_args.source_dataset = getattr(reference_args, "source_dataset", base_args.source_dataset)
    base_args.split = getattr(reference_args, "split", base_args.split)
    base_args.fps_sim = float(getattr(reference_args, "fps_sim", base_args.fps_sim))
    base_args.array_size = int(getattr(reference_args, "array_size", base_args.array_size))
    base_args.target_size = int(getattr(reference_args, "target_size", base_args.target_size))
    base_args.output_channels = int(getattr(reference_args, "output_channels", base_args.output_channels))
    base_args.tile_size = int(getattr(reference_args, "tile_size", base_args.tile_size))
    base_args.percentile_low = float(getattr(reference_args, "percentile_low", base_args.percentile_low))
    base_args.percentile_high = float(getattr(reference_args, "percentile_high", base_args.percentile_high))
    base_args.normalization_mode = getattr(reference_args, "normalization_mode", base_args.normalization_mode)
    base_args.range_scope = getattr(reference_args, "range_scope", base_args.range_scope)
    base_args.range_calibration_split = getattr(reference_args, "range_calibration_split", base_args.range_calibration_split)
    base_args.use_noise_fn = int(getattr(reference_args, "use_noise_fn", base_args.use_noise_fn))
    base_args.shot_noise = int(getattr(reference_args, "shot_noise", base_args.shot_noise))
    base_args.i_thermal = float(getattr(reference_args, "i_thermal", base_args.i_thermal))
    base_args.bandwidth = float(getattr(reference_args, "bandwidth", base_args.bandwidth))
    base_args.drift_hours = [float(args.drift_hours)]

    native_params_csv = Path(param_summary["native_csv"]).resolve()
    configs = build_native_export_configs(
        base_args=base_args,
        native_params_csv=native_params_csv,
        output_dir=output_dir,
        start_index=args.start_index,
        num_images=args.num_images,
    )
    manifests = run_exports(configs, resume=bool(args.resume))
    scenario_lookup = build_sample_lookup(manifests)
    eta_single = float(param_summary["native_params"]["eta_single"])
    panel_title_prefix = f"Case1 Native | eta={eta_single:.4f} | drift={float(args.drift_hours):.1f}h"

    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_manifest = []
    first_key = (eval_cmp.FPS_VALUES[0], eval_cmp.READOUT_CONFIGS[0]["label"])
    reference_samples = scenario_lookup[first_key]["samples"]
    ordered_indices = sorted(reference_samples.keys())
    for sample_index in ordered_indices:
        sample_label = reference_samples[sample_index]["label"]
        panel_path = panel_dir / f"sample_{sample_index:04d}_{sample_label}.png"
        compose_panel_for_sample(
            sample_index,
            sample_label,
            scenario_lookup,
            panel_path,
            panel_title_prefix,
        )
        panel_manifest.append(
            {
                "sample_index": sample_index,
                "sample_label": sample_label,
                "panel_path": str(panel_path),
            }
        )

    summary = {
        "base_results_dir": str(base_results_dir),
        "output_dir": str(output_dir),
        "native_params_csv": str(native_params_csv),
        "start_index": int(args.start_index),
        "num_images": int(args.num_images),
        "drift_hours": float(args.drift_hours),
        "scenario_dirs": {
            f"fps{int(fps):03d}_{readout_cfg['label']}": str(Path(cfg.output_dir))
            for fps, readout_cfg, cfg, _scenario_dir in configs
        },
        "panels": panel_manifest,
    }
    summary_path = output_dir / "sequence10_panels_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Panel directory: {panel_dir}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
