import argparse
import copy
import json
from pathlib import Path

import torch

import generate_sensor_verification_images as base_pipeline
import generate_sensor_verification_images_video_sequence as video_sequence_pipeline
from finetune_resnet18_sensor_video_sequence import DATASET_CONFIG, build_sensor_args, load_weights, select_calibration_dataset
from train_restoration_resnet18_sensor_video_sequence import (
    build_models,
    evaluate_model,
    format_eval_summary,
    freeze_module,
    resolve_device,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_CASES = ("raw", "ideal", "nonideal", "restored")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current temporal restoration frontend + frozen ResNet18 classifier "
            "under the same stateful sensor nonidealities modeled by "
            "generate_sensor_verification_images_video_sequence.py. "
            "Supports single-run evaluation and multi-scenario sweeps."
        )
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT.parent / ".datasets"),
        help="Dataset root. Data will be read from <data_path>/<dataset>-data",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--classifier_path",
        default=None,
        help="Frozen pretrained ResNet18 checkpoint. Defaults to models/resnet18_<dataset>.pth",
    )
    parser.add_argument(
        "--restoration_model_path",
        default=None,
        help=(
            "Restoration frontend checkpoint. Defaults to "
            "models/tiny_restoration_frontend_resnet18_<dataset>_video_sequence_h<history_frames>.pth"
        ),
    )
    parser.add_argument(
        "--results_json",
        default=None,
        help="Optional path to save JSON results. Defaults to artifacts/eval_runs/restoration_video_sequence_eval_<dataset>.json",
    )
    parser.add_argument(
        "--scenario_name",
        default="default",
        help="Name used for a single CLI-configured run when --scenario_json is not provided",
    )
    parser.add_argument(
        "--scenario_json",
        default=None,
        help=(
            "Optional JSON file describing multiple sensor-evaluation scenarios. "
            "Supported formats: a list of dicts, or {'scenarios': [...]}. "
            "Each scenario can be {'name': 'foo', 'video_fps': 20.0, 'readout': 'tia', ...} "
            "or {'name': 'foo', 'overrides': {...}}."
        ),
    )
    parser.add_argument(
        "--eval_cases",
        nargs="+",
        default=list(DEFAULT_EVAL_CASES),
        choices=DEFAULT_EVAL_CASES,
        help="Evaluation cases to report",
    )
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="If > 0, only evaluate this many batches per scenario",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor_rng_seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. Defaults to cpu to avoid interfering with active GPU jobs.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="If > 0, call torch.set_num_threads(cpu_threads). Useful for CPU evaluation control.",
    )

    parser.add_argument("--target_size", type=int, default=32)
    parser.add_argument("--output_channels", type=int, default=3)
    parser.add_argument(
        "--post_norm",
        default="auto",
        choices=["none", "auto", "cifar10", "cifar100", "imagenet"],
    )
    parser.add_argument("--array_size", type=int, default=32)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog_readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc_enabled", type=int, default=0)
    parser.add_argument("--power_max", type=float, default=0.1)
    parser.add_argument("--params_csv", default=str(base_pipeline.DEFAULT_PARAMS_CSV))
    parser.add_argument(
        "--normalization_mode",
        default="calibration",
        choices=["physical", "calibration", "per_frame", "none"],
        help="Normalization path used by the current video-sequence pipeline; default keeps the calibrated legacy behavior.",
    )
    parser.add_argument("--prange1_density", type=float, default=None)
    parser.add_argument("--prange2_density", type=float, default=None)
    parser.add_argument("--pmin_density", type=float, default=None)
    parser.add_argument("--pmax_density", type=float, default=None)
    parser.add_argument("--device_area_cm2", type=float, default=None)
    parser.add_argument("--force_single_carrier", type=int, default=0)
    parser.add_argument("--single_r", type=float, default=None)
    parser.add_argument("--single_eta", type=float, default=None)
    parser.add_argument("--single_trise", type=float, default=None)
    parser.add_argument("--single_tfall", type=float, default=None)
    parser.add_argument("--trap_saturation_time", type=float, default=None)
    parser.add_argument("--trap_amplitude_pct", type=float, default=None)
    parser.add_argument("--noise_1f_density_1hz", type=float, default=None)
    parser.add_argument("--aging_tau_hours", type=float, default=None)
    parser.add_argument("--r_degradation_pct", type=float, default=None)
    parser.add_argument("--video_fps", type=float, default=50.0)
    parser.add_argument("--fps_sim", type=float, default=1000.0)
    parser.add_argument("--adc_bits", type=int, default=8)
    parser.add_argument("--adc_full_scale", type=float, default=None)
    parser.add_argument("--range_mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument(
        "--range_scope",
        default="calibration",
        choices=["per_frame", "calibration"],
        help="Whether range scaling is computed per frame or fixed from a calibration set.",
    )
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument("--range_calibration_split", default="train", choices=["train", "test"])
    parser.add_argument("--range_calibration_samples", type=int, default=1024)
    parser.add_argument("--range_calibration_skip_frames", type=int, default=0)
    parser.add_argument("--use_noise_fn", type=int, default=1)
    parser.add_argument("--shot_noise", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=5000.0)
    parser.add_argument("--i_thermal", type=float, default=0.0)
    parser.add_argument("--startup_dark_frames", type=int, default=0)
    parser.add_argument(
        "--temporal_noise_mode",
        default="pixel_buffered",
        choices=[
            "pixel_buffered",
            "pixel_repeated_window",
            "global_full_sequence",
            "global_repeated_window",
        ],
    )
    parser.add_argument("--temporal_noise_window_frames", type=int, default=10)
    parser.add_argument("--fast_tia_frame_step", type=int, default=0)
    parser.add_argument(
        "--spatial_variation_r_pct",
        type=float,
        default=None,
        help="Fixed pixel-wise responsivity variation percentage used by the stateful sensor backend.",
    )
    parser.add_argument(
        "--spatial_variation_cache_dir",
        default=None,
        help="Optional cache directory for fixed responsivity-variation random maps.",
    )
    parser.add_argument("--tia_gain_ohm", type=float, default=None)
    parser.add_argument("--integration_gain_v_per_c", type=float, default=None)

    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--history_frames", type=int, default=4)
    return parser.parse_args()


def resolve_default_classifier_path(dataset_name):
    return PROJECT_ROOT / "models" / f"resnet18_{dataset_name}.pth"


def resolve_default_restoration_path(dataset_name, history_frames):
    history_tag = f"h{history_frames}"
    return PROJECT_ROOT / "models" / f"tiny_restoration_frontend_resnet18_{dataset_name}_video_sequence_{history_tag}.pth"


def resolve_default_results_json(dataset_name):
    return PROJECT_ROOT / "artifacts" / "eval_runs" / f"restoration_video_sequence_eval_{dataset_name}.json"


def resolve_model_paths(args):
    dataset_name = args.dataset.lower()
    classifier_path = Path(args.classifier_path) if args.classifier_path else resolve_default_classifier_path(dataset_name)
    restoration_model_path = (
        Path(args.restoration_model_path)
        if args.restoration_model_path
        else resolve_default_restoration_path(dataset_name, args.history_frames)
    )
    return classifier_path, restoration_model_path


def resolve_results_path(args):
    dataset_name = args.dataset.lower()
    return Path(args.results_json) if args.results_json else resolve_default_results_json(dataset_name)


def resolve_sequence_base_params(args):
    return video_sequence_pipeline.resolve_sequence_base_params(args)


def load_scenarios(args):
    if not args.scenario_json:
        return [{"name": args.scenario_name, "overrides": {}}]

    payload = json.loads(Path(args.scenario_json).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "scenarios" in payload:
        raw_scenarios = payload["scenarios"]
    elif isinstance(payload, list):
        raw_scenarios = payload
    elif isinstance(payload, dict):
        raw_scenarios = [payload]
    else:
        raise ValueError(f"Unsupported scenario JSON root type: {type(payload)!r}")

    scenarios = []
    for index, item in enumerate(raw_scenarios):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario at index {index} must be a JSON object, got {type(item)!r}")
        name = item.get("name", f"scenario_{index:02d}")
        if "overrides" in item:
            overrides = item["overrides"]
        else:
            overrides = {key: value for key, value in item.items() if key not in {"name", "description"}}
        if not isinstance(overrides, dict):
            raise ValueError(f"Scenario {name!r} overrides must be a JSON object")
        scenarios.append({"name": name, "overrides": overrides})
    return scenarios


def apply_overrides(base_args, overrides):
    scenario_args = copy.deepcopy(base_args)
    for key, value in overrides.items():
        if not hasattr(scenario_args, key):
            raise ValueError(f"Unknown scenario override key: {key!r}")
        setattr(scenario_args, key, value)
    return scenario_args


def compute_derived_metrics(evaluation):
    cases = evaluation["cases"]
    derived = {}

    def case_acc(case_name):
        if case_name not in cases:
            return None
        return float(cases[case_name]["accuracy"])

    nonideal_acc = case_acc("nonideal")
    restored_acc = case_acc("restored")
    ideal_acc = case_acc("ideal")
    raw_acc = case_acc("raw")

    if nonideal_acc is not None and restored_acc is not None:
        derived["restored_minus_nonideal_acc"] = restored_acc - nonideal_acc
    else:
        derived["restored_minus_nonideal_acc"] = None

    if raw_acc is not None and restored_acc is not None:
        derived["restored_minus_raw_acc"] = restored_acc - raw_acc
    else:
        derived["restored_minus_raw_acc"] = None

    if ideal_acc is not None and restored_acc is not None:
        derived["ideal_minus_restored_acc"] = ideal_acc - restored_acc
    else:
        derived["ideal_minus_restored_acc"] = None

    if ideal_acc is not None and nonideal_acc is not None and restored_acc is not None:
        gap = ideal_acc - nonideal_acc
        if abs(gap) > 1e-12:
            derived["restoration_gap_recovery_ratio"] = (restored_acc - nonideal_acc) / gap
        else:
            derived["restoration_gap_recovery_ratio"] = None
    else:
        derived["restoration_gap_recovery_ratio"] = None

    return derived


def print_scenario_header(name, scenario_args, classifier_path, restoration_model_path):
    print("\n" + "=" * 88, flush=True)
    print(f"Scenario: {name}", flush=True)
    print(f"  dataset            {scenario_args.dataset}", flush=True)
    print(f"  split              {scenario_args.split}", flush=True)
    print(f"  device             {scenario_args.device}", flush=True)
    print(f"  history_frames     {scenario_args.history_frames}", flush=True)
    print(f"  readout            {scenario_args.readout}", flush=True)
    print(f"  video_fps          {scenario_args.video_fps}", flush=True)
    print(f"  fps_sim            {scenario_args.fps_sim}", flush=True)
    print(f"  use_noise_fn       {scenario_args.use_noise_fn}", flush=True)
    print(f"  shot_noise         {scenario_args.shot_noise}", flush=True)
    print(f"  i_thermal          {scenario_args.i_thermal}", flush=True)
    print(
        f"  spatial_var_R_pct  "
        f"{0.0 if scenario_args.spatial_variation_r_pct is None else float(scenario_args.spatial_variation_r_pct):.2f}",
        flush=True,
    )
    print(f"  bandwidth          {scenario_args.bandwidth}", flush=True)
    print(f"  startup_dark_frames {scenario_args.startup_dark_frames}", flush=True)
    print(f"  classifier_path    {classifier_path}", flush=True)
    print(f"  restoration_path   {restoration_model_path}", flush=True)
    print("=" * 88, flush=True)


def evaluate_single_scenario(name, scenario_args):
    dataset_name = scenario_args.dataset.lower()
    num_classes = DATASET_CONFIG[dataset_name]["num_classes"]
    classifier_path, restoration_model_path = resolve_model_paths(scenario_args)
    device = resolve_device(scenario_args.device)

    if not classifier_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    if not restoration_model_path.is_file():
        raise FileNotFoundError(f"Restoration checkpoint not found: {restoration_model_path}")

    print_scenario_header(name, scenario_args, classifier_path, restoration_model_path)

    eval_dataset = base_pipeline.load_base_dataset(dataset_name, scenario_args.data_path, split=scenario_args.split)
    train_dataset = base_pipeline.load_base_dataset(dataset_name, scenario_args.data_path, split="train")
    test_dataset = base_pipeline.load_base_dataset(dataset_name, scenario_args.data_path, split="test")
    calibration_dataset = select_calibration_dataset(scenario_args, train_dataset, test_dataset)
    base_params = resolve_sequence_base_params(scenario_args)
    if scenario_args.tia_gain_ohm is None:
        scenario_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if scenario_args.integration_gain_v_per_c is None:
        scenario_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))

    calibration_sensor_args = build_sensor_args(scenario_args, scenario_args.sensor_rng_seed)
    case_range_bounds = video_sequence_pipeline.compute_case_range_bounds(
        args=calibration_sensor_args,
        calibration_dataset=calibration_dataset,
        base_params=base_params,
        cases=["ideal", "nonideal"],
    )

    restoration_model, classifier_model = build_models(scenario_args, device, num_classes)
    load_weights(classifier_model, classifier_path)
    freeze_module(classifier_model)
    restoration_model.load_state_dict(torch.load(restoration_model_path, map_location="cpu", weights_only=False))
    restoration_model = restoration_model.to(device).eval()
    classifier_model = classifier_model.to(device).eval()

    evaluation = evaluate_model(
        restoration_model=restoration_model,
        classifier_model=classifier_model,
        base_dataset=eval_dataset,
        args=scenario_args,
        base_params=base_params,
        device=device,
        cases=scenario_args.eval_cases,
        case_range_bounds=case_range_bounds,
    )
    derived = compute_derived_metrics(evaluation)
    print(format_eval_summary(evaluation), flush=True)
    if derived["restored_minus_nonideal_acc"] is not None:
        print(
            f"  restored - nonideal acc = {derived['restored_minus_nonideal_acc']:.4f} percentage points",
            flush=True,
        )
    if derived["restoration_gap_recovery_ratio"] is not None:
        print(
            f"  recovered ideal gap ratio = {derived['restoration_gap_recovery_ratio']:.4f}",
            flush=True,
        )

    return {
        "name": name,
        "effective_args": vars(scenario_args),
        "classifier_path": str(classifier_path),
        "restoration_model_path": str(restoration_model_path),
        "case_range_bounds": case_range_bounds,
        "evaluation": evaluation,
        "derived_metrics": derived,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    scenarios = load_scenarios(args)
    results = {
        "base_args": vars(args),
        "scenarios": [],
    }

    for scenario in scenarios:
        scenario_args = apply_overrides(args, scenario["overrides"])
        scenario_result = evaluate_single_scenario(scenario["name"], scenario_args)
        results["scenarios"].append(scenario_result)

    results_json = resolve_results_path(args)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {results_json}", flush=True)


if __name__ == "__main__":
    main()
