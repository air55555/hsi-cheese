from __future__ import annotations

from pathlib import Path

from hsi_easy import save_images_to_files
from main4 import calculate_statistics, generate_rgb
from utils import load_hsi


CAPTURE_ROOT = Path(r"D:\hsm_capture")
OUTPUT_ROOT = Path(r"D:\statistic_images")


def process_hdr(hdr_path: Path) -> None:
    hsi_cube = load_hsi(str(hdr_path))
    rgb = generate_rgb(hsi_cube, 10, 20, 30)
    stats = calculate_statistics(hsi_cube)
    stats = {k: v for k, v in stats.items() if v is not None}

    cube_dir_name = hdr_path.parent.name
    out_dir = OUTPUT_ROOT / cube_dir_name
    prefix = f"{hdr_path.stem}_"

    out = save_images_to_files(out_dir, rgb, stats, prefix=prefix)
    print(f"[OK] {hdr_path} -> {out} ({1 + len(stats)} images)")


def main() -> None:
    if not CAPTURE_ROOT.exists():
        raise SystemExit(f"Capture root not found: {CAPTURE_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # One HDR per cube directory under CAPTURE_ROOT
    for cube_dir in sorted(p for p in CAPTURE_ROOT.iterdir() if p.is_dir()):
        hdr_files = sorted(cube_dir.glob("*.hdr"))
        if not hdr_files:
            print(f"[SKIP] No HDR in {cube_dir}")
            continue

        hdr_path = hdr_files[0]

        try:
            process_hdr(hdr_path)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Failed for {hdr_path}: {e}")


if __name__ == "__main__":
    main()

