# -*- coding: utf-8 -*-
"""
Aggregate 4-model comparison from one finished v19 source + three finished split baseline runs.
No retraining.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def load_compare_module():
    here = Path(__file__).resolve().parent
    path = here / "rcm_pinn_8cases_compare_v22.py"
    if not path.exists():
        raise FileNotFoundError(f"Need {{path.name}} in the same folder as this file.")
    spec = importlib.util.spec_from_file_location("rcm_compare_v22_aggregate_v23", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load rcm_pinn_8cases_compare_v22.py")
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = load_compare_module()
base = cmp.base


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def import_existing_model_results(source_path: Path, root: Path, spec: Any, chosen_cases: List[str]) -> List[Dict[str, Any]]:
    imported: List[Dict[str, Any]] = []
    source_path = source_path.resolve()
    model_root = root / spec.name
    base.ensure_dir(model_root)

    source_manifest = None
    try:
        source_manifest = cmp._read_json_from_source(source_path, "run_manifest.json")
    except Exception:
        source_manifest = None

    reuse_root = root / "reused_sources" / spec.name
    base.ensure_dir(reuse_root)
    for rel in [
        "all_results.csv",
        "run_manifest.json",
        "run_args.json",
        "physics_config.json",
        "case_catalog.json",
        "system_info.json",
        "model_catalog.json",
        "single_model_leaderboard.json",
    ]:
        try:
            cmp._copy_from_source(source_path, rel, reuse_root / Path(rel).name)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    for case_name in chosen_cases:
        case_summary = cmp._read_json_from_source(source_path, f"{spec.name}/{case_name}/case_summary.json")
        case_dir = model_root / case_name
        base.ensure_dir(case_dir)
        for rel in cmp._list_case_files_in_source(source_path, f"{spec.name}/{case_name}"):
            try:
                cmp._copy_from_source(source_path, rel, case_dir / Path(rel).name)
            except Exception:
                pass
        if not list(case_dir.glob("*")):
            try:
                case_summary = cmp._read_json_from_source(source_path, f"{case_name}/case_summary.json")
                for rel in cmp._list_case_files_in_source(source_path, case_name):
                    try:
                        cmp._copy_from_source(source_path, rel, case_dir / Path(rel).name)
                    except Exception:
                        pass
            except Exception:
                pass

        final_metrics = dict(case_summary.get("final_metrics", {}))
        if "mean_rel_l2" not in final_metrics:
            vals = [final_metrics.get(f"rel_l2_{f}", float("nan")) for f in base.FIELD_NAMES]
            vals = [float(v) for v in vals if np.isfinite(v)]
            final_metrics["mean_rel_l2"] = float(np.mean(vals)) if vals else float("nan")

        summary = {
            "model_name": spec.name,
            "display_name": spec.display_name,
            "model_description": spec.description + "（直接复用既有结果）",
            "case_name": case_name,
            "case_description": base.CASE_DESCRIPTIONS.get(case_name, case_summary.get("description", case_name)),
            "best_mean_rel_l2": float(case_summary.get("best_mean_rel_l2", final_metrics.get("mean_rel_l2", float("nan")))),
            "best_epoch": int(case_summary.get("best_epoch", -1)),
            "final_metrics": final_metrics,
            "param_count": int(case_summary.get("param_count", 0)),
            "train_seconds": float(case_summary.get("train_seconds", 0.0)),
            "stop_reason": str(case_summary.get("stop_reason", "reused_existing_result")),
            "rollback_count": int(case_summary.get("rollback_count", 0)),
            "effective_n_supervised": int(case_summary.get("effective_n_supervised", 0)),
            "effective_n_residual": int(case_summary.get("effective_n_residual", 0)),
            "effective_n_boundary_per_side": int(case_summary.get("effective_n_boundary_per_side", 0)),
            "effective_residual_chunk_size": int(case_summary.get("effective_residual_chunk_size", 0)),
            "hard_bank_size": int(case_summary.get("hard_bank_size", 0)),
            "best_model": str(case_summary.get("best_model", "best_model.pt")),
            "train_overrides": spec.train_overrides,
            "field_weights": spec.field_weights,
            "reused_from": str(source_path),
        }
        save_json(case_dir / "case_summary_reused.json", summary)
        imported.append(summary)

    save_json(root / "reused_sources" / f"{spec.name}_source.json", {
        "source_path": str(source_path),
        "source_manifest": source_manifest,
        "chosen_cases": chosen_cases,
        "spec_name": spec.name,
    })
    return imported


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate v19 + 3 split baselines into one final 4-model comparison")
    p.add_argument("--reuse_v19_source", type=str, required=True, help="Finished v19 run folder or back.zip")
    p.add_argument("--streamlite_source", type=str, required=True, help="Finished StreamLite baseline run folder or back.zip")
    p.add_argument("--hardbc_source", type=str, required=True, help="Finished HardBC baseline run folder or back.zip")
    p.add_argument("--ffmlp_source", type=str, required=True, help="Finished FFMLP baseline run folder or back.zip")
    p.add_argument("--cases", type=str, default="all", help="all or comma-separated case names")
    p.add_argument("--outdir", type=str, default="runs_rcm_pinn_compare_aggregate", help="output root")
    p.add_argument("--device", type=str, default="cuda", help="device label stored in manifest only")
    p.add_argument("--seed", type=int, default=20260322, help="seed stored in manifest only")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    base.set_seed(args.seed)
    chosen_cases = base.CASE_ORDER if args.cases.strip().lower() == "all" else [c.strip() for c in args.cases.split(",") if c.strip()]
    specs = cmp.model_specs()
    spec_map = {s.name: s for s in specs}

    v19_src = cmp.normalize_cli_path(args.reuse_v19_source)
    stream_src = cmp.normalize_cli_path(args.streamlite_source)
    hardbc_src = cmp.normalize_cli_path(args.hardbc_source)
    ffmlp_src = cmp.normalize_cli_path(args.ffmlp_source)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.outdir) / f"rcm_compare_split_v23_{timestamp}"
    base.ensure_dir(root)

    phys = base.PhysConfig()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    save_json(root / "case_catalog.json", base.CASE_DESCRIPTIONS)
    save_json(root / "run_args.json", vars(args))
    save_json(root / "physics_config.json", base.asdict(phys))
    save_json(root / "model_catalog.json", {
        s.name: {
            "display_name": s.display_name,
            "description": s.description,
            "train_overrides": s.train_overrides,
            "field_weights": s.field_weights,
        } for s in specs if s.name in {"rcm_pinn_v19", "streamlite_pinn", "hardbc_spectral_pinn", "ffmlp_pinn"}
    })

    print("=" * 92)
    print("Aggregate 4-model comparison from finished runs")
    print(f"Output : {root}")
    print(f"Cases  : {chosen_cases}")
    print(f"v19    : {v19_src}")
    print(f"stream : {stream_src}")
    print(f"hardbc : {hardbc_src}")
    print(f"ffmlp  : {ffmlp_src}")
    print("=" * 92)

    all_summaries = []
    all_summaries.extend(cmp.import_existing_v19_results(v19_src, root, spec_map["rcm_pinn_v19"], chosen_cases))
    all_summaries.extend(import_existing_model_results(stream_src, root, spec_map["streamlite_pinn"], chosen_cases))
    all_summaries.extend(import_existing_model_results(hardbc_src, root, spec_map["hardbc_spectral_pinn"], chosen_cases))
    all_summaries.extend(import_existing_model_results(ffmlp_src, root, spec_map["ffmlp_pinn"], chosen_cases))

    flat_rows = cmp.flatten_results(all_summaries)
    cmp.write_csv(root / "all_results.csv", flat_rows)
    ranking_pack = cmp.build_rankings(flat_rows)
    cmp.write_csv(root / "ranking_by_case.csv", ranking_pack["ranking_rows"])
    save_json(root / "ranking_by_case.json", ranking_pack["ranking_by_case"])

    selected_specs = [spec_map["rcm_pinn_v19"], spec_map["streamlite_pinn"], spec_map["hardbc_spectral_pinn"], spec_map["ffmlp_pinn"]]
    leaderboard = cmp.write_leaderboard(root, flat_rows, selected_specs)
    cmp.plot_case_comparisons(root, flat_rows, selected_specs)
    cmp.plot_global_comparisons(root, flat_rows, leaderboard, selected_specs)
    cmp.make_manifest(root, args, phys, device, selected_specs, flat_rows)
    save_json(root / "aggregate_sources.json", {
        "reuse_v19_source": str(v19_src),
        "streamlite_source": str(stream_src),
        "hardbc_source": str(hardbc_src),
        "ffmlp_source": str(ffmlp_src),
    })

    back_zip = base.create_back_zip(root, "back.zip")
    print("=" * 92)
    print("Finished. Send back only this file:", back_zip)
    print("=" * 92)


if __name__ == "__main__":
    main()
