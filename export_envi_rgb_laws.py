#!/usr/bin/env python3
"""
Export ENVI hyperspectral cube to multiple RGB PNG sets using different band laws.

Supports ENVI BSQ cubes with little-endian uint16 (data type 12).
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


def parse_hdr_text(text: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*([^=]+?)\s*=\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        key = m.group(1).strip().lower()
        value_lines = [m.group(2)]
        depth = value_lines[0].count("{") - value_lines[0].count("}")
        j = i + 1
        while j < len(lines) and depth > 0:
            value_lines.append(lines[j])
            depth += lines[j].count("{") - lines[j].count("}")
            j += 1
        entries[key] = "\n".join(value_lines).strip()
        i = j
    return entries


def strip_braces(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    return value


def parse_int_field(hdr: Dict[str, str], key: str) -> int:
    if key not in hdr:
        raise ValueError(f"Missing '{key}' in HDR")
    return int(strip_braces(hdr[key]))


def resolve_data_path(hdr_path: Path, explicit_data: Path | None) -> Path:
    if explicit_data is not None:
        return explicit_data
    stem = hdr_path.with_suffix("")
    candidates = [stem, stem.with_suffix(".img"), stem.with_suffix(".raw"), stem.with_suffix(".dat")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not infer data file. Pass --data explicitly (expected same name as .hdr, or .img/.raw/.dat)."
    )


def load_cube(hdr_path: Path, data_path: Path | None) -> np.ndarray:
    hdr = parse_hdr_text(hdr_path.read_text(encoding="utf-8", errors="replace"))
    samples = parse_int_field(hdr, "samples")
    lines = parse_int_field(hdr, "lines")
    bands = parse_int_field(hdr, "bands")
    interleave = strip_braces(hdr.get("interleave", "")).lower()
    data_type = parse_int_field(hdr, "data type")
    byte_order = parse_int_field(hdr, "byte order")
    header_offset = int(strip_braces(hdr.get("header offset", "0")))

    if interleave != "bsq":
        raise ValueError(f"Only bsq is supported in this script (got '{interleave}')")
    if data_type != 12:
        raise ValueError(f"Only ENVI data type 12 (uint16) is supported (got {data_type})")
    if byte_order not in (0, 1):
        raise ValueError(f"Unsupported byte order value: {byte_order}")

    dtype = np.dtype("<u2" if byte_order == 0 else ">u2")
    actual_data = resolve_data_path(hdr_path, data_path)
    raw = np.fromfile(actual_data, dtype=dtype, offset=header_offset)
    expected = bands * lines * samples
    if raw.size != expected:
        raise ValueError(
            f"Data size mismatch. Expected {expected} elements, got {raw.size}. "
            f"Check hdr dimensions and selected data file."
        )
    cube = raw.reshape((bands, lines, samples))
    return cube


def normalize_to_u8(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    lo = np.percentile(band, p_low)
    hi = np.percentile(band, p_high)
    if hi <= lo:
        out = np.zeros_like(band, dtype=np.uint8)
        return out
    scaled = (band.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def rgb_from_bands(cube: np.ndarray, b_r: int, b_g: int, b_b: int) -> np.ndarray:
    # band indices are 1-based in UI/filenames; convert to 0-based for array access.
    r = normalize_to_u8(cube[b_r - 1])
    g = normalize_to_u8(cube[b_g - 1])
    b = normalize_to_u8(cube[b_b - 1])
    return np.stack([r, g, b], axis=-1)


def chunks_triplets(bands: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for i in range(1, bands + 1, 3):
        a = i
        b = min(i + 1, bands)
        c = min(i + 2, bands)
        out.append((a, b, c))
    return out


def reverse_triplets(bands: int) -> List[Tuple[int, int, int]]:
    base = chunks_triplets(bands)
    return [(c, b, a) for (a, b, c) in base]


def anchored_triplets(bands: int, g_anchor: int = 150, b_anchor: int = 224) -> List[Tuple[int, int, int]]:
    g_anchor = max(1, min(g_anchor, bands))
    b_anchor = max(1, min(b_anchor, bands))
    out: List[Tuple[int, int, int]] = []
    for i in range(1, bands + 1, 3):
        out.append((i, g_anchor, b_anchor))
    return out


def interleaved_stride_triplets(bands: int) -> List[Tuple[int, int, int]]:
    # 1, 76, 151 then 2, 77, 152 ... gives another "cover all range" style.
    part = math.ceil(bands / 3)
    out: List[Tuple[int, int, int]] = []
    for i in range(1, part + 1):
        a = i
        b = min(i + part, bands)
        c = min(i + 2 * part, bands)
        out.append((a, b, c))
    return out


def rolling_window_triplets(bands: int, step: int = 1) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for i in range(1, bands - 1, max(step, 1)):
        out.append((i, i + 1, i + 2))
    return out


def edge_symmetric_triplets(bands: int) -> List[Tuple[int, int, int]]:
    # Pair early and late spectrum bands with center as green.
    center = max(1, bands // 2)
    out: List[Tuple[int, int, int]] = []
    for i in range(1, bands + 1, 3):
        mirror = bands - i + 1
        mirror = max(1, min(mirror, bands))
        out.append((i, center, mirror))
    return out


def random_triplets(bands: int, seed: int = 42, count: int | None = None) -> List[Tuple[int, int, int]]:
    rng = random.Random(seed)
    pool = list(range(1, bands + 1))
    if count is None:
        count = math.ceil(bands / 3)
    out: List[Tuple[int, int, int]] = []
    for _ in range(max(1, count)):
        # sample() ensures three different bands.
        r, g, b = rng.sample(pool, k=min(3, len(pool)))
        if len(pool) < 3:
            # Fallback for tiny cubes.
            g = pool[0]
            b = pool[-1]
        out.append((r, g, b))
    return out


def unique_triplets(seq: Iterable[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    seen = set()
    out = []
    for t in seq:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def build_law_catalog() -> Dict[str, Tuple[str, Callable[[int], List[Tuple[int, int, int]]]]]:
    catalog = {
        "law1_anchor_150_last": (
            "R=i(step3), G=150 (clamped), B=last band. Generalized from 224-specific idea.",
            lambda bands: unique_triplets(anchored_triplets(bands, g_anchor=150, b_anchor=bands)),
        ),
        "law2_seq_123_456": (
            "Sequential triplets: (1,2,3), (4,5,6), ...",
            lambda bands: unique_triplets(chunks_triplets(bands)),
        ),
        "law3_reverse": (
            "Reverse each sequential triplet: (3,2,1), (6,5,4), ...",
            lambda bands: unique_triplets(reverse_triplets(bands)),
        ),
        "law4_interleaved": (
            "Interleaved thirds across spectrum: (1,1+T,1+2T), ...",
            lambda bands: unique_triplets(interleaved_stride_triplets(bands)),
        ),
        "law5_anchor_mid_last": (
            "R=i(step3), G=middle band, B=last band.",
            lambda bands: unique_triplets(anchored_triplets(bands, g_anchor=bands // 2, b_anchor=bands)),
        ),
        "law6_rolling_step3": (
            "Rolling window with step=3: (1,2,3), (4,5,6), ...",
            lambda bands: unique_triplets(rolling_window_triplets(bands, step=3)),
        ),
        "law7_rolling_step1": (
            "Dense rolling window with step=1: (1,2,3), (2,3,4), ...",
            lambda bands: unique_triplets(rolling_window_triplets(bands, step=1)),
        ),
        "law8_edge_symmetric": (
            "Edge/center symmetry: (i, center, bands-i+1), step=3.",
            lambda bands: unique_triplets(edge_symmetric_triplets(bands)),
        ),
        "law9_random_seed42": (
            "Reproducible random triplets (seed=42), count≈bands/3.",
            lambda bands: unique_triplets(random_triplets(bands, seed=42, count=math.ceil(bands / 3))),
        ),
    }
    # Backward-compatible alias for previously used name.
    catalog["law1_anchor_150_224"] = catalog["law1_anchor_150_last"]
    return catalog


def export_law(cube: np.ndarray, triplets: Sequence[Tuple[int, int, int]], law_dir: Path, law_name: str) -> int:
    law_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, (r, g, b) in enumerate(triplets, start=1):
        rgb = rgb_from_bands(cube, r, g, b)
        out_name = f"{idx:03d}_R{r:03d}_G{g:03d}_B{b:03d}.png"
        Image.fromarray(rgb).save(law_dir / out_name)
        count += 1
    print(f"[{law_name}] saved {count} pngs -> {law_dir}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ENVI BSQ cube into RGB PNG sets by band laws.")
    parser.add_argument("--hdr", type=Path, help="Path to .hdr file")
    parser.add_argument("--data", type=Path, default=None, help="Path to data file (.img/.raw/etc). Optional")
    parser.add_argument("--out-dir", type=Path, help="Output root directory")
    parser.add_argument("--list-laws", action="store_true", help="Print available laws and exit")
    parser.add_argument(
        "--laws",
        nargs="*",
        default=[
            "law1_anchor_150_last",
            "law2_seq_123_456",
            "law3_reverse",
            "law4_interleaved",
            "law5_anchor_mid_last",
            "law6_rolling_step3",
            "law7_rolling_step1",
            "law8_edge_symmetric",
            "law9_random_seed42",
        ],
        help=(
            "Subset of laws to export. Available: law1_anchor_150_last, law2_seq_123_456, "
            "law3_reverse, law4_interleaved, law5_anchor_mid_last, law6_rolling_step3, "
            "law7_rolling_step1, law8_edge_symmetric, law9_random_seed42"
        ),
    )
    args = parser.parse_args()

    catalog = build_law_catalog()
    if args.list_laws:
        print("Available laws:")
        for name, (desc, _) in catalog.items():
            print(f"  - {name}: {desc}")
        return

    if args.hdr is None:
        parser.error("--hdr is required unless --list-laws is used")
    if args.out_dir is None:
        parser.error("--out-dir is required unless --list-laws is used")

    cube = load_cube(args.hdr, args.data)
    bands = cube.shape[0]

    all_laws = {name: fn(bands) for name, (_, fn) in catalog.items()}

    requested = []
    for name in args.laws:
        if name not in all_laws:
            raise ValueError(f"Unknown law '{name}'. Available: {', '.join(all_laws.keys())}")
        requested.append(name)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for law_name in requested:
        total += export_law(cube, all_laws[law_name], args.out_dir / law_name, law_name)

    print(f"Done. Total PNG saved: {total}")


if __name__ == "__main__":
    main()
