 import argparse
 from pathlib import Path
 
 from main4 import load_hsi, generate_rgb, select_roi, save_config
 
 
 def main() -> None:
     """
     Run only the GUI for ROI selection and store the result in the config.
 
     Usage:
         hsi-statistic-roi path/to/file.hdr
     """
     parser = argparse.ArgumentParser(
         prog="hsi-statistic-roi",
         description="Open ROI selection GUI for an ENVI HDR cube and save the ROI.",
     )
     parser.add_argument(
         "hdr_path",
         type=str,
         help="Path to the ENVI .hdr file.",
     )
 
     args = parser.parse_args()
 
     hdr_path = Path(args.hdr_path).expanduser().resolve()
     if not hdr_path.exists():
         raise SystemExit(f"HDR file not found: {hdr_path}")
 
     # Load hyperspectral cube and show ROI GUI
     hsi_cube = load_hsi(str(hdr_path))
     rgb_image = generate_rgb(hsi_cube, 10, 20, 30)
 
     roi = select_roi(rgb_image)
     if roi:
         save_config(str(hdr_path), roi)
         print(f"ROI {roi} saved for file: {hdr_path}")
     else:
         print("No ROI selected; configuration not updated.")

