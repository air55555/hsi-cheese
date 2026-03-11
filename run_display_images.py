 from __future__ import annotations
 
 from pathlib import Path
 
 from hsi_easy import compute_rgb_and_stats, display_all_images
 
 
 def main() -> None:
     hdr_path = Path(r"C:\Users\1\Downloads\cube_26_02_12_43_33\cube_26_02_12_43_33.hdr")
 
     # Uses existing config.json (and ROI) by default
     rgb, stats, roi = compute_rgb_and_stats(hdr_path)
     if roi:
         print(f"Using ROI from config: {roi}")
     else:
         print("No ROI used (no matching ROI in config.json).")
 
     display_all_images(rgb, stats)
 
 
 if __name__ == "__main__":
     main()
 
