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
from mpl_toolkits.mplot3d import proj3d


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_case1_native_vs_x2_video_sequence as eval_cmp  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]
READOUTS = [
    ("tia", "TIA", "#F8D89B"),
    ("integration", "Integral", "#B8D6CC"),
]
SPATIAL_VARIATION_LEVELS = [1.0, 3.0, 5.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate case1 native baseline accuracy under spatial variation of R and render a 3D summary chart."
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
        help="Directory where the spatial-variation sweep artifacts will be written.",
    )
    parser.add_argument(
        "--spatial-variation-pcts",
        type=float,
        nargs="+",
        default=SPATIAL_VARIATION_LEVELS,
        help="Spatial variation percentages applied to R.",
    )
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def resolve_results_dir(args):
    if args.results_dir not in {None, "", "auto"}:
        return Path(args.results_dir).expanduser()
    return THIS_DIR / "artifacts" / "case1_native_spatial_variation_sweep"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    base_args.use_noise_fn = 0
    base_args.shot_noise = 0
    base_args.spatial_variation_r_pct = 0.0
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
        "spatial_variation_r_pct",
        "video_fps",
        "readout_label",
        "readout_display",
        "accuracy_nonideal",
        "results_json",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def overlay_projected_text_labels(ax, labels):
    ax.figure.canvas.draw()
    for x, y, z, text in labels:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.annotate(
            text,
            xy=(x2, y2),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#111111",
            annotation_clip=False,
            zorder=1000,
        )


def render_3d_chart(records, output_path):
    grouped = {
        (float(r["spatial_variation_r_pct"]), float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"])
        for r in records
    }

    fig = plt.figure(figsize=(16.8, 10.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.90)

    group_spacing = 1.65
    readout_spacing = 0.78
    dx = 0.42
    dy = 0.68
    y_positions = {5.0: 0.00, 3.0: 1.05, 1.0: 2.10}
    x_group_starts = np.arange(len(FPS_VALUES), dtype=np.float64) * (group_spacing + readout_spacing)
    x_positions_by_readout = {
        "tia": x_group_starts,
        "integration": x_group_starts + readout_spacing,
    }
    projected_labels = []

    for readout_key, _readout_label, color in ["tia", "TIA", "#F8D89B"], ["integration", "Integral", "#B8D6CC"]:
        xs = x_positions_by_readout[readout_key]
        for variation_pct in sorted(y_positions.keys(), reverse=True):
            ys = np.full_like(xs, y_positions[variation_pct])
            dz = np.array([grouped[(variation_pct, fps, readout_key)] for fps in FPS_VALUES], dtype=np.float64)
            ax.bar3d(
                xs,
                ys,
                np.zeros_like(xs),
                dx,
                dy,
                dz,
                color=color,
                alpha=1.0,
                edgecolor="#222222",
                linewidth=0.9,
                shade=True,
                zsort="average",
            )
            for x, y, z in zip(xs, ys, dz):
                projected_labels.append((x + dx * 0.5, y + dy * 0.5, min(z + 1.0, 103.0), f"{z:.1f}"))

    ax.set_title("Case1 Native Baseline Accuracy", pad=26, fontsize=24, fontweight="bold")

    xticks = []
    xticklabels = []
    for fps, tia_x, integration_x in zip(FPS_VALUES, x_positions_by_readout["tia"], x_positions_by_readout["integration"]):
        xticks.append((tia_x + integration_x) * 0.5 + dx * 0.5)
        xticklabels.append(f"FPS {int(fps)}")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=18, fontweight="bold")
    ax.set_xlabel("Video FPS", labelpad=24, fontsize=20, fontweight="bold")

    ytick_centers = [y_positions[key] + dy * 0.5 for key in [5.0, 3.0, 1.0]]
    ax.set_yticks(ytick_centers)
    ax.set_yticklabels(["5%", "3%", "1%"], fontsize=16, fontweight="bold")
    ax.set_ylabel("Spatial Variation of R", labelpad=24, fontsize=20, fontweight="bold")

    ax.set_zlabel("Accuracy (%)", labelpad=18, fontsize=20, fontweight="bold")
    ax.set_zlim(0.0, 104.0)
    ax.set_zticks(np.arange(0, 101, 10))
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=22)
    ax.tick_params(axis="z", labelsize=16, width=1.4)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((6.4, 2.7, 2.9))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_alpha(0.08)
        axis.pane.set_edgecolor("#999999")

    overlay_projected_text_labels(ax, projected_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base_results_dir = Path(args.base_results_dir)
    results_dir = resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(base_results_dir / "case1_native_vs_x2_accuracy_summary.json")
    parameter_summary = load_json(base_results_dir / "parameter_summary.json")
    native_params_csv = Path(parameter_summary["native_csv"]).resolve()
    base_args = build_base_args_from_summary(summary)

    records = []
    total = len(args.spatial_variation_pcts) * len(FPS_VALUES) * len(READOUTS)
    counter = 0
    start_time = time.time()

    for spatial_variation_pct in args.spatial_variation_pcts:
        for readout_key, readout_display, _color in READOUTS:
            for fps in FPS_VALUES:
                counter += 1
                scenario_dir = results_dir / f"svr{str(spatial_variation_pct).replace('.', 'p')}_fps{int(fps):03d}_{readout_key}"
                scenario_dir.mkdir(parents=True, exist_ok=True)
                cfg = copy.deepcopy(base_args)
                cfg.params_csv = str(native_params_csv)
                cfg.video_fps = float(fps)
                cfg.readout = readout_key
                cfg.analog_readout = readout_key
                cfg.adc_enabled = 0
                cfg.adc_bits = 8
                cfg.spatial_variation_r_pct = float(spatial_variation_pct)
                cfg.results_json = str(scenario_dir / "results.json")
                cfg.output_dir = str(scenario_dir)

                print(
                    f"[{counter}/{total}] Spatial variation {float(spatial_variation_pct):.1f}% | "
                    f"FPS={int(fps)} | {readout_display}",
                    flush=True,
                )
                result_payload = run_scenario(cfg, resume=bool(args.resume))
                accuracy = eval_cmp.accuracy_from_result(result_payload)
                elapsed = time.time() - start_time
                print(f"  accuracy={accuracy:.2f}% elapsed={elapsed:.1f}s", flush=True)

                records.append(
                    {
                        "spatial_variation_r_pct": float(spatial_variation_pct),
                        "video_fps": float(fps),
                        "readout_label": readout_key,
                        "readout_display": readout_display,
                        "accuracy_nonideal": float(accuracy),
                        "results_json": str(Path(cfg.results_json).resolve()),
                    }
                )

    records_csv = results_dir / "case1_native_spatial_variation_accuracy_records.csv"
    save_records_csv(records, records_csv)
    chart_path = results_dir / "case1_native_spatial_variation_3d_tia_integration.png"
    render_3d_chart(records, chart_path)

    summary_payload = {
        "base_results_dir": str(base_results_dir),
        "results_dir": str(results_dir),
        "native_params_csv": str(native_params_csv),
        "spatial_variation_pcts": [float(v) for v in args.spatial_variation_pcts],
        "records_csv": str(records_csv),
        "chart_path": str(chart_path),
        "records": records,
    }
    summary_path = results_dir / "case1_native_spatial_variation_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Records CSV: {records_csv}", flush=True)
    print(f"Chart: {chart_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
