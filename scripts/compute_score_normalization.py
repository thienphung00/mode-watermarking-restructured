#!/usr/bin/env python3
"""
This script computes family-level normalization statistics (mean, std) of log_odds
over clean samples for use by evaluation and threshold calibration; identity transform is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils import setup_logging

# Minimum clean samples for normalization; below this we warn (non-fatal).
MIN_CLEAN_SAMPLES_WARN = 10


def load_per_image_scores_by_family(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all result JSONs from results_dir; group per_image entries by family_id.
    Each per_image entry must have label, log_odds; transform defaults to "identity" when missing.

    Accepts:
    - Canonical format: family_id + per_image (from run_ablation_detection or detect_bayesian_test {family_id}.json).
    - detect_bayesian_test detailed_results.json: detailed_results list with label, log_odds, transform;
      family_id taken from data or "default".
    """
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for p in results_dir.glob("*.json"):
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        family_id = data.get("family_id")
        per_image = data.get("per_image")
        # Accept detect_bayesian_test.py format: detailed_results list with label, log_odds, transform
        if not per_image and "detailed_results" in data:
            detailed = data["detailed_results"]
            if isinstance(detailed, list) and detailed:
                family_id = family_id or "default"
                per_image = [
                    {
                        "label": r.get("label", 0),
                        "log_odds": r.get("log_odds", 0.0),
                        "transform": r.get("transform", "identity"),
                    }
                    for r in detailed
                ]
        if not family_id or not per_image:
            continue
        if family_id not in by_family:
            by_family[family_id] = []
        by_family[family_id].extend(per_image)
    return by_family


def _per_transform_counts(per_image: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Count total, clean, watermarked per transform; missing transform defaults to identity."""
    by_t: Dict[str, Dict[str, int]] = {}
    for e in per_image:
        t = e.get("transform", "identity")
        if t not in by_t:
            by_t[t] = {"total": 0, "clean": 0, "watermarked": 0}
        by_t[t]["total"] += 1
        if e.get("label", 1) == 0:
            by_t[t]["clean"] += 1
        else:
            by_t[t]["watermarked"] += 1
    return by_t


def compute_normalization_for_family(
    per_image: List[Dict[str, Any]],
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Compute mean and std of log_odds over clean (label==0) samples.
    Uses all transforms (no per-transform stats; family-level only). Normalization
    is applied to log_odds only; S is not used.
    """
    # Score consistency: we normalize log_odds only; do not use S
    if per_image and any(e.get("S") is not None for e in per_image):
        if logger:
            logger.warning("Normalization uses log_odds only; S is present in data but not normalized.")

    clean_scores = [
        float(e["log_odds"])
        for e in per_image
        if e.get("label", 1) == 0
    ]
    n_clean = len(clean_scores)
    if n_clean < MIN_CLEAN_SAMPLES_WARN and logger:
        logger.warning(
            "n_clean=%d is below recommended minimum %d for normalization stability.",
            n_clean, MIN_CLEAN_SAMPLES_WARN,
        )
    if not clean_scores:
        return {"mean": 0.0, "std": 1.0, "n_clean": 0, "per_transform_counts": _per_transform_counts(per_image)}

    arr = np.array(clean_scores)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0:
        std = 1.0

    by_t = _per_transform_counts(per_image)
    # Identity explicit: warn if identity has only wm or only clean
    if "identity" in by_t:
        id_clean = by_t["identity"]["clean"]
        id_wm = by_t["identity"]["watermarked"]
        if id_clean == 0 or id_wm == 0:
            if logger:
                logger.warning(
                    "Transform 'identity' has only %s samples (clean=%d, watermarked=%d).",
                    "watermarked" if id_clean == 0 else "clean", id_clean, id_wm,
                )

    return {
        "mean": mean,
        "std": std,
        "n_clean": n_clean,
        "per_transform_counts": by_t,
    }


def normalize_score(score: float, mean: float, std: float) -> float:
    """Normalize score using family-level mean and std."""
    return (score - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute score normalization per detector family (Phase 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory of detection result JSONs (from detect_bayesian_test.py or run_ablation_detection; per_image or detailed_results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path: normalization params per family (mean, std, n_clean)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    logger = setup_logging(level=args.log_level)

    by_family = load_per_image_scores_by_family(args.results_dir)
    if not by_family:
        raise FileNotFoundError(
            f"No result JSONs with per_image (or detailed_results from detect_bayesian_test) and family_id found in {args.results_dir}"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for family_id, per_image in by_family.items():
        out[family_id] = compute_normalization_for_family(per_image, logger=logger)
        logger.info(
            "family %s: mean=%.4f std=%.4f n_clean=%d",
            family_id,
            out[family_id]["mean"],
            out[family_id]["std"],
            out[family_id]["n_clean"],
        )
        counts = out[family_id].get("per_transform_counts", {})
        if counts:
            logger.info("  per_transform_counts: %s", json.dumps(counts))
            for t in sorted(counts.keys()):
                c = counts[t]
                logger.info("    %s: total=%d clean=%d watermarked=%d", t, c["total"], c["clean"], c["watermarked"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote normalization params to %s", args.output)


if __name__ == "__main__":
    main()
