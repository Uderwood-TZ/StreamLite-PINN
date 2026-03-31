# -*- coding: utf-8 -*-
"""
Split baseline runner: HardBC-Spectral PINN
Runs only one baseline on the same 8 cases used by v19.
No v19 retraining here.
"""
from __future__ import annotations

import argparse
import math
import importlib.util
import json
import time
from pathlib import Path


def load_compare_module():
    here = Path(__file__).resolve().parent
    path = here / "rcm_pinn_8cases_compare_v22.py"
    if not path.exists():
        raise FileNotFoundError(f"Need {path.name} in the same folder as this file.")
    spec = importlib.util.spec_from_file_location("rcm_compare_v22_split_hardbc_baseline_v23", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load rcm_pinn_8cases_compare_v22.py")
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = load_compare_module()
base = cmp.base
TARGET_MODEL = "hardbc_spectral_pinn"


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = cmp.build_argparser()
    p.description = "Split runner for HardBC-Spectral PINN only"
    p.set_defaults(outdir="runs_rcm_pinn_split_baselines")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    base.set_seed(args.seed)
    base.set_global_speed_flags()
    device = base.torch.device(args.device if (args.device != "cuda" or base.torch.cuda.is_available()) else "cpu")
    phys = base.PhysConfig()
    specs = cmp.model_specs()
    spec_map = {s.name: s for s in specs}
    if TARGET_MODEL not in spec_map:
        raise RuntimeError(f"Unknown model: {TARGET_MODEL}")
    spec = spec_map[TARGET_MODEL]
    chosen_cases = base.CASE_ORDER if args.cases.strip().lower() == "all" else [c.strip() for c in args.cases.split(",") if c.strip()]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.outdir) / f"hardbc_baseline_v23_{timestamp}"
    base.ensure_dir(root)

    save_json(root / "case_catalog.json", base.CASE_DESCRIPTIONS)
    save_json(root / "run_args.json", vars(args))
    save_json(root / "physics_config.json", base.asdict(phys))
    save_json(root / "model_catalog.json", {
        spec.name: {
            "display_name": spec.display_name,
            "description": spec.description,
            "train_overrides": spec.train_overrides,
            "field_weights": spec.field_weights,
        }
    })

    print("=" * 92)
    print("Split baseline runner")
    print(f"Model  : {spec.display_name} ({spec.name})")
    print(f"Output : {root}")
    print(f"Device : {device} | {base.device_name(device)}")
    print(f"Cases  : {chosen_cases}")
    print("=" * 92)

    summaries = []
    total_t0 = time.time()
    for case_name in chosen_cases:
        elapsed_total = time.time() - total_t0
        if math.isfinite(args.max_total_seconds) and elapsed_total >= args.max_total_seconds:
            print(f"Total time cap reached before {case_name}, stopping.")
            break
        print("-" * 92)
        print(f"Running {spec.display_name} | {case_name}: {base.CASE_DESCRIPTIONS[case_name]}")
        summary = cmp.train_one_case_for_model(spec, case_name, args, phys, root, device)
        summaries.append(summary)

    if not summaries:
        print("No case finished.")
        return

    flat_rows = cmp.flatten_results(summaries)
    cmp.write_csv(root / "all_results.csv", flat_rows)
    leaderboard = cmp.write_leaderboard(root, flat_rows, [spec])
    cmp.make_manifest(root, args, phys, device, [spec], flat_rows)
    save_json(root / "single_model_leaderboard.json", leaderboard)

    back_zip = base.create_back_zip(root, "back.zip")
    print("=" * 92)
    print("Finished. Send back only this file:", back_zip)
    print("=" * 92)


if __name__ == "__main__":
    main()
