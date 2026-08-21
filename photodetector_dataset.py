"""
Dataset utilities for paper-driven photodetector non-ideal parameter studies.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

DATASET_PATH = Path(__file__).resolve().parent / "data" / "photodetector_paper_dataset.csv"

META_KEYS = [
    "entry_id",
    "material",
    "material_subcategory",
    "specific_material",
    "structure",
    "paper_year",
    "paper_title",
    "doi",
    "notes",
    "substrate",
    "waveband",
    "responsivity_bias_V",
    "stability",
    "ref_id",
    "D_star_jones",
    "D_note",
    "LDR_dB",
    "LDR_note",
    "EQE_pct",
    "f3db_Hz",
]

MODEL_PARAM_KEYS = [
    "R_fast",
    "eta_fast",
    "tau_rise_fast",
    "tau_fall_fast",
    "R_slow",
    "eta_slow",
    "tau_rise_slow",
    "tau_fall_slow",
    "alpha",
    "beta",
    "delta",
    "gamma",
    "tau_drift",
    "drift_scale",
]

NOISE_PARAM_KEYS = [
    "white_sigma",
    "flicker_sigma",
    "kappa",
    "low_freq_amp",
    "low_freq_hz",
]

NUMERIC_KEYS = [
    "paper_year",
    "responsivity_bias_V",
    "D_star_jones",
    "LDR_dB",
    "EQE_pct",
    "f3db_Hz",
] + MODEL_PARAM_KEYS + NOISE_PARAM_KEYS


def _normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_number_or_none(value):
    text = _normalize_text(value)
    if text is None:
        return None
    return float(text)


def load_photodetector_dataset(path: Optional[Path] = None) -> List[Dict[str, object]]:
    dataset_path = Path(path) if path is not None else DATASET_PATH
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in NUMERIC_KEYS:
                    parsed[key] = _to_number_or_none(value)
                else:
                    parsed[key] = _normalize_text(value)
            rows.append(parsed)
    return rows


def save_photodetector_dataset(entries: Iterable[Dict[str, object]], path: Optional[Path] = None) -> Path:
    dataset_path = Path(path) if path is not None else DATASET_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = META_KEYS + MODEL_PARAM_KEYS + NOISE_PARAM_KEYS
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {}
            for key in fieldnames:
                value = entry.get(key)
                row[key] = "" if value is None else value
            writer.writerow(row)
    return dataset_path


def find_entry(entries: Iterable[Dict[str, object]], entry_id: str) -> Dict[str, object]:
    for entry in entries:
        if entry.get("entry_id") == entry_id:
            return entry
    raise KeyError(f"entry_id not found: {entry_id}")


def filter_entries(
    entries: Iterable[Dict[str, object]],
    material: Optional[str] = None,
    material_subcategory: Optional[str] = None,
    specific_material: Optional[str] = None,
    structure: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict[str, object]]:
    results = []
    for entry in entries:
        if material is not None and entry.get("material") != material:
            continue
        if material_subcategory is not None and entry.get("material_subcategory") != material_subcategory:
            continue
        if specific_material is not None and entry.get("specific_material") != specific_material:
            continue
        if structure is not None and entry.get("structure") != structure:
            continue
        if year is not None and entry.get("paper_year") != float(year):
            continue
        results.append(entry)
    return results


def build_params_from_entry(
    entry: Dict[str, object],
    default_params: Dict[str, float],
    fill_mode: str = "fill_default",
) -> Dict[str, object]:
    if fill_mode not in {"fill_default", "strict", "metadata_only"}:
        raise ValueError(f"unsupported fill_mode: {fill_mode}")

    params = {}
    missing = []
    for key in MODEL_PARAM_KEYS:
        value = entry.get(key)
        if value is None:
            if fill_mode == "fill_default":
                params[key] = float(default_params[key])
            elif fill_mode == "metadata_only":
                params[key] = None
            else:
                missing.append(key)
        else:
            params[key] = float(value)

    if missing:
        raise ValueError(
            f"entry {entry.get('entry_id')} is missing required model params in strict mode: {', '.join(missing)}"
        )

    return params


def build_noise_config_from_entry(
    entry: Dict[str, object],
    default_noise_config: Optional[Dict[str, float]] = None,
    fill_mode: str = "fill_default",
) -> Dict[str, Optional[float]]:
    if fill_mode not in {"fill_default", "strict", "metadata_only"}:
        raise ValueError(f"unsupported fill_mode: {fill_mode}")

    default_noise_config = default_noise_config or {}
    noise = {}
    missing = []
    for key in NOISE_PARAM_KEYS:
        value = entry.get(key)
        if value is None:
            if fill_mode == "fill_default":
                noise[key] = default_noise_config.get(key)
            elif fill_mode == "metadata_only":
                noise[key] = None
            else:
                missing.append(key)
        else:
            noise[key] = float(value)

    if fill_mode == "strict" and missing:
        raise ValueError(
            f"entry {entry.get('entry_id')} is missing required noise params in strict mode: {', '.join(missing)}"
        )

    return noise


def iter_simulation_configs(
    entries: Iterable[Dict[str, object]],
    default_params: Dict[str, float],
    default_noise_config: Optional[Dict[str, float]] = None,
    fill_mode: str = "fill_default",
) -> Iterator[Dict[str, object]]:
    for entry in entries:
        yield {
            "entry_id": entry.get("entry_id"),
            "material": entry.get("material"),
            "material_subcategory": entry.get("material_subcategory"),
            "specific_material": entry.get("specific_material"),
            "structure": entry.get("structure"),
            "paper_year": entry.get("paper_year"),
            "paper_title": entry.get("paper_title"),
            "doi": entry.get("doi"),
            "notes": entry.get("notes"),
            "params": build_params_from_entry(entry, default_params, fill_mode=fill_mode),
            "noise_config": build_noise_config_from_entry(
                entry,
                default_noise_config=default_noise_config,
                fill_mode=fill_mode,
            ),
        }


def append_entry(
    entry: Dict[str, object],
    path: Optional[Path] = None,
) -> Path:
    entries = load_photodetector_dataset(path=path)
    entries.append(entry)
    return save_photodetector_dataset(entries, path=path)
