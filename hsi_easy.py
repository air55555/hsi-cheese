from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

_backend = str(matplotlib.get_backend() or "").lower()
if "interagg" in _backend:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from main4 import (
    calculate_statistics,
    display_images,
    generate_rgb,
    load_config,
    load_hsi,
    save_config,
    select_roi,
)

ROI = Tuple[int, int, int, int]  # (x_start, x_end, y_start, y_end)


def run_roi_gui(
    hdr_path: str | Path,
    rgb_bands: Tuple[int, int, int] = (10, 20, 30),
    config_path: str = "config.json",
) -> Optional[ROI]:
    """
    Case 1: run ROI GUI and save ROI to config.json.
    """
    hdr_path = Path(hdr_path).expanduser().resolve()
    if not hdr_path.exists():
        raise FileNotFoundError(f"HDR file not found: {hdr_path}")

    hsi_cube = load_hsi(str(hdr_path))
    rgb = generate_rgb(hsi_cube, *rgb_bands)
    roi = select_roi(rgb)

    if roi:
        save_config(str(hdr_path), roi, config_path=config_path)

    return roi


def compute_rgb_and_stats(
    hdr_path: str | Path,
    roi: Optional[ROI] = None,
    rgb_bands: Tuple[int, int, int] = (10, 20, 30),
    config_path: str = "config.json",
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[ROI]]:
    """
    Load cube, apply ROI (argument > config), compute RGB and stats maps.
    """
    hdr_path = Path(hdr_path).expanduser().resolve()
    if not hdr_path.exists():
        raise FileNotFoundError(f"HDR file not found: {hdr_path}")

    cfg_file, cfg_roi = load_config(config_path=config_path)
    if roi is None and cfg_file and Path(cfg_file).resolve() == hdr_path and cfg_roi:
        roi = tuple(cfg_roi)  # type: ignore[assignment]

    hsi_cube = load_hsi(str(hdr_path))
    if roi:
        x_start, x_end, y_start, y_end = roi
        hsi_cube = hsi_cube[y_start:y_end, x_start:x_end, :]

    rgb = generate_rgb(hsi_cube, *rgb_bands)
    stats = calculate_statistics(hsi_cube)
    stats = {k: v for k, v in stats.items() if v is not None}  # drop failed calculations
    return rgb, stats, roi


def display_all_images(
    rgb: np.ndarray,
    stats: Dict[str, np.ndarray],
    max_stats_per_figure: int = 31,
) -> None:
    """
    Case 2: display RGB + all statistic maps (paged across figures if needed).

    Uses the existing `main4.display_images` for each page.
    """
    items = list(stats.items())
    if not items:
        display_images(rgb)
        return

    for i in range(0, len(items), max_stats_per_figure):
        chunk = dict(items[i : i + max_stats_per_figure])
        display_images(rgb, **chunk)


def save_images_to_files(
    output_dir: str | Path,
    rgb: np.ndarray,
    stats: Dict[str, np.ndarray],
    prefix: str = "",
    cmap: str = "viridis",
) -> Path:
    """
    Case 3: save RGB and all statistic maps to image files.

    Produces:
      - <prefix>rgb.png
      - <prefix><stat_name_sanitized>.png  (with colorbar)
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def sanitize(name: str) -> str:
        keep = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            elif ch.isspace():
                keep.append("_")
        out = "".join(keep).strip("_")
        return out or "stat"

    rgb_path = output_dir / f"{prefix}rgb.png"
    plt.imsave(rgb_path, rgb)

    for name, stat_map in stats.items():
        stat = np.asarray(stat_map, dtype=float)
        finite_mask = np.isfinite(stat)
        if not np.any(finite_mask):
            continue  # skip completely invalid maps

        vmin = stat[finite_mask].min()
        vmax = stat[finite_mask].max()
        if vmax > vmin:
            stat = (stat - vmin) / (vmax - vmin)
        else:
            stat = np.zeros_like(stat)

        out_path = output_dir / f"{prefix}{sanitize(name)}.png"
        plt.imsave(out_path, stat, cmap=cmap)

    return output_dir


def save_cubes_to_file(
    output_path: str | Path,
    rgb: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> Path:
    """
    Save raw arrays only (no text/plots).

    Writes a single compressed NumPy archive (.npz) containing:
      - rgb: (H, W, 3) uint8 (or whatever you pass in)
      - stats/<name>: each statistic map as a 2D array
    """
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")

    payload: Dict[str, np.ndarray] = {"rgb": np.asarray(rgb)}
    for k, v in stats.items():
        payload[f"stats/{k}"] = np.asarray(v)

    np.savez_compressed(output_path, **payload)
    return output_path
 
