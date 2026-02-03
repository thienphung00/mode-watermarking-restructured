#!/usr/bin/env python3
"""
Phase 4: Apply image transformations as views (no new samples).

Transforms are parameterized and applied programmatically. Outputs are grouped
by transform name. Image identity (sample_id, label, key_id, seed) is preserved
across transforms. Uses PIL for image I/O; no YAML explosion.

Input: manifest from generate_training_images (image_path, label, key_id, seed)
Output: transformed images under output_dir/transforms/<transform_name>/ and
        manifest entries with transform field for downstream precompute/detection.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils import setup_logging


# -----------------------------------------------------------------------------
# Parameterized transform definitions (reusable, no YAML)
# -----------------------------------------------------------------------------

def transform_identity(image: Image.Image, **kwargs: Any) -> Image.Image:
    """Return image unchanged (canonical view)."""
    return image.copy()


def transform_resize(image: Image.Image, size: int = 256, **kwargs: Any) -> Image.Image:
    """Resize to size x size then back to 512x512 (view simulation)."""
    out = image.resize((size, size), Image.LANCZOS)
    return out.resize((512, 512), Image.LANCZOS)


def transform_jpeg(image: Image.Image, quality: int = 90, **kwargs: Any) -> Image.Image:
    """JPEG compress-decompress (quality 1–100)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def transform_gaussian_blur(image: Image.Image, radius: float = 1.0, **kwargs: Any) -> Image.Image:
    """Gaussian blur (radius in pixels)."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


# Registry: transform_name -> (callable, default_kwargs)
TRANSFORM_REGISTRY: Dict[str, Tuple[Callable[..., Image.Image], Dict[str, Any]]] = {
    "identity": (transform_identity, {}),
    "resize_256": (transform_resize, {"size": 256}),
    "jpeg_90": (transform_jpeg, {"quality": 90}),
    "jpeg_75": (transform_jpeg, {"quality": 75}),
    "gaussian_blur_1": (transform_gaussian_blur, {"radius": 1.0}),
    "gaussian_blur_2": (transform_gaussian_blur, {"radius": 2.0}),
}


def load_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load manifest (JSONL or JSON list) and return list of entries with image_path."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    entries: List[Dict[str, Any]] = []
    if manifest_path.suffix == ".jsonl":
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    else:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and ("watermarked" in data or "clean" in data):
            entries = data.get("watermarked", []) + data.get("clean", [])
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path}")
    for i, e in enumerate(entries):
        if "image_path" not in e:
            raise ValueError(f"Entry {i} missing 'image_path'")
    return entries


def apply_transforms(
    input_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    transform_names: List[str],
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Apply each registered transform to every image in the manifest; save under
    output_dir/transforms/<name>/ preserving directory structure; write
    manifest with transform field.
    """
    if logger is None:
        logger = setup_logging()
    entries = load_manifest_entries(manifest_path)
    base_dir = input_dir.resolve()

    out_manifest_lines: List[Dict[str, Any]] = []
    for entry in entries:
        rel_path = entry["image_path"]
        src_path = base_dir / rel_path
        if not src_path.exists():
            # Fallback: relative to manifest parent (e.g. same dir as manifest)
            src_path = manifest_path.parent / rel_path
        if not src_path.exists():
            logger.warning("Image not found %s (base %s), skipping", rel_path, base_dir)
            continue
        try:
            image = Image.open(src_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open %s: %s", src_path, e)
            continue

        for name in transform_names:
            if name not in TRANSFORM_REGISTRY:
                logger.warning("Unknown transform %s, skipping", name)
                continue
            func, defaults = TRANSFORM_REGISTRY[name]
            out_image = func(image, **defaults)
            # Output: output_dir/transforms/<name>/<same rel_path>
            out_sub = output_dir / "transforms" / name
            out_path = out_sub / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_image.save(out_path)
            # Manifest entry: same identity + transform
            new_entry = {k: v for k, v in entry.items() if k != "prompt"}
            new_entry["image_path"] = str(Path("transforms") / name / rel_path)
            new_entry["transform"] = name
            out_manifest_lines.append(new_entry)

    out_manifest_path = output_dir / "transforms_manifest.jsonl"
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest_path, "w") as f:
        for line in out_manifest_lines:
            f.write(json.dumps(line, default=str) + "\n")
    logger.info("Wrote %d entries to %s", len(out_manifest_lines), out_manifest_path)
    return out_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply image transforms as views (Phase 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Base directory for manifest image_path (e.g. outputs/train)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.jsonl from generate_training_images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for transforms/ and transforms_manifest.jsonl",
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["identity", "resize_256", "jpeg_90", "gaussian_blur_1"],
        help="Transform names from registry: identity, resize_256, jpeg_90, jpeg_75, gaussian_blur_1, gaussian_blur_2",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    logger = setup_logging(level=args.log_level)
    apply_transforms(
        input_dir=args.input_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        transform_names=args.transforms,
        logger=logger,
    )


if __name__ == "__main__":
    main()
