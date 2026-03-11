from __future__ import annotations

from pathlib import Path

from hsi_easy import compute_rgb_and_stats, save_images_to_files


def main() -> None:
    # Input HDR (as requested)
    hdr_path = Path(r"C:\Users\1\Downloads\cube_26_02_12_43_33\cube_26_02_12_43_33_(230,319,224)_cropped.hdr")

    # Use existing config.json in repo root (default in hsi_easy)
    rgb, stats, roi = compute_rgb_and_stats(hdr_path)

    # Save next to the HDR file
    output_dir = hdr_path.parent / "saved_images"
    prefix = f"{hdr_path.stem}_"
    out = save_images_to_files(output_dir, rgb, stats, prefix=prefix)

    print(f"Saved {1 + len(stats)} images to: {out}")
    if roi:
        print(f"Used ROI from config: {roi}")
    else:
        print("No ROI used (no matching ROI in config.json).")


if __name__ == "__main__":
    main()
 
