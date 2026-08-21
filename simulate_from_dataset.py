"""
Batch simulation entry points for photodetector paper datasets.
"""

from __future__ import annotations

import argparse

from photodetector_dataset import (
    DATASET_PATH,
    filter_entries,
    iter_simulation_configs,
    load_photodetector_dataset,
)
from photodetector_model import (
    NOISE_FN,
    PARAMS_TRUE,
    build_complete_params,
    make_noise_function_from_config,
    noise_fn_to_config,
    params_to_vec,
    plot_arbitrary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run photodetector simulations from a paper dataset.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to the CSV dataset.")
    parser.add_argument("--entry-id", default=None, help="Run only one dataset entry.")
    parser.add_argument("--material", default=None, help="Filter by material.")
    parser.add_argument("--material-subcategory", default=None, help="Filter by material subcategory.")
    parser.add_argument("--specific-material", default=None, help="Filter by the specific material name.")
    parser.add_argument("--structure", default=None, help="Filter by structure.")
    parser.add_argument("--year", type=int, default=None, help="Filter by paper year.")
    parser.add_argument(
        "--fill-mode",
        default="fill_default",
        choices=["fill_default", "strict", "metadata_only"],
        help="How to handle missing parameters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to simulate after filtering.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    entries = load_photodetector_dataset(args.dataset)
    if args.entry_id is not None:
        entries = [entry for entry in entries if entry.get("entry_id") == args.entry_id]
    else:
        entries = filter_entries(
            entries,
            material=args.material,
            material_subcategory=args.material_subcategory,
            specific_material=args.specific_material,
            structure=args.structure,
            year=args.year,
        )

    if args.limit is not None:
        entries = entries[: args.limit]

    if not entries:
        raise SystemExit("No dataset entries matched the given filters.")

    default_noise_config = noise_fn_to_config(NOISE_FN)
    for config in iter_simulation_configs(
        entries,
        default_params=PARAMS_TRUE,
        default_noise_config=default_noise_config,
        fill_mode=args.fill_mode,
    ):
        if config["params"] is None or any(value is None for value in config["params"].values()):
            print(f"[skip] {config['entry_id']}: incomplete params under fill_mode={args.fill_mode}")
            continue

        full_params = build_complete_params(config["params"])
        noise_fn = make_noise_function_from_config(
            config["noise_config"],
            label=f"dataset_noise_{config['entry_id']}",
        )
        print(
            f"[run] entry_id={config['entry_id']} material={config['material']} "
            f"subcategory={config['material_subcategory']} specific={config['specific_material']} "
            f"structure={config['structure']} year={config['paper_year']}"
        )
        plot_arbitrary(params_to_vec(full_params), noise_fn=noise_fn)


if __name__ == "__main__":
    main()
