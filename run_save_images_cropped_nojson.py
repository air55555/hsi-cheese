from __future__ import annotations

from pathlib import Path

from hsi_easy import save_images_to_files
from main4 import calculate_statistics, generate_rgb
from utils import load_hsi


def main() -> None:
    # Cropped cube (no config.json / ROI needed)
    hdr_path = Path(
        r"C:\Users\1\Downloads\cube_26_02_12_43_33\cube_26_02_12_43_33_(230,319,224)_cropped.hdr"
    )
    if not hdr_path.exists():
        raise SystemExit(f"HDR file not found: {hdr_path}")

    hsi_cube = load_hsi(str(hdr_path))
    rgb = generate_rgb(hsi_cube, 10, 20, 30)
    stats = calculate_statistics(hsi_cube)
    stats = {k: v for k, v in stats.items() if v is not None}

    output_dir = hdr_path.parent / "saved_images_cropped"
    prefix = f"{hdr_path.stem}_"
    out = save_images_to_files(output_dir, rgb, stats, prefix=prefix)
    print(f"Saved {1 + len(stats)} images to: {out}")


if __name__ == "__main__":
    main()

