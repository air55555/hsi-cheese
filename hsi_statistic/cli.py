 import argparse
 import os
 from pathlib import Path
 
 from main4 import save_config, main as _interactive_main
 
 
 def main() -> None:
     """
     Console entry point.
 
     Parameters
     ----------
     hdr_path : str (positional)
         Path to the ENVI .hdr file. This is the only required parameter.
     """
     parser = argparse.ArgumentParser(
         prog="hsi-statistic",
         description="Run hyperspectral preprocessing and statistics on an ENVI HDR cube.",
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
 
     # Save configuration so the existing workflow can pick it up.
     save_config(str(hdr_path), roi=None)
 
     # Delegate to the existing interactive workflow (ROI selection, processing, etc.).
     _interactive_main()
 
