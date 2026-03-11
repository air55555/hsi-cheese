 # hsi_statistic
 
 Interactive toolkit for preprocessing ENVI hyperspectral cubes and computing a broad set of per-pixel statistical metrics, with an optional GUI for ROI selection and visualization.
 
## Usage from Python (recommended)

This repo is now easiest to use directly from Python via `hsi_easy.py`.

### Case 1: run ROI GUI (and save ROI)

```python
from hsi_easy import run_roi_gui

roi = run_roi_gui("path/to/your_file.hdr")
print("ROI:", roi)
```

This writes `config.json` with the HDR path and ROI.

### Case 2: display RGB + all statistic maps

```python
from hsi_easy import compute_rgb_and_stats, display_all_images

rgb, stats, roi = compute_rgb_and_stats("path/to/your_file.hdr")
display_all_images(rgb, stats)
```

### Case 3: save RGB + all statistic maps to files

```python
from hsi_easy import compute_rgb_and_stats, save_images_to_files

rgb, stats, roi = compute_rgb_and_stats("path/to/your_file.hdr")
save_images_to_files("outputs", rgb, stats, prefix="")
```

## Using from Python

You can also call the workflow directly from Python instead of using the CLI.

### 1. Run ROI GUI from Python

```python
from main4 import load_hsi, generate_rgb, select_roi, save_config

hdr_path = "path/to/your_file.hdr"

hsi_cube = load_hsi(hdr_path)
rgb_image = generate_rgb(hsi_cube, 10, 20, 30)

roi = select_roi(rgb_image)
if roi:
    save_config(hdr_path, roi)
```

### 2. Run processing and statistics from Python

```python
from main4 import main as run_pipeline
from main4 import save_config

hdr_path = "path/to/your_file.hdr"

# Optional: ensure the HDR path (and ROI, if known) are stored in config.json
save_config(hdr_path, roi=None)  # or your previously selected ROI

# Run the full pipeline (preprocessing, saving, statistics, visualization)
run_pipeline()
```

## Preparing for PyPI upload

1. **Install build and twine** (once per environment):

   ```bash
   python -m pip install --upgrade build twine
   ```

2. **Build the source and wheel distributions** from the project root (where `pyproject.toml` lives):

   ```bash
   python -m build
   ```

   This will create a `dist/` folder with files like:

   - `hsi_statistic-0.1.0.tar.gz`
   - `hsi_statistic-0.1.0-py3-none-any.whl`

3. **Create a PyPI account and API token**:

   - Go to the PyPI website and create an account.
   - In your account settings, create an **API token** with upload permissions.
   - Save the token somewhere secure (you will use it as a password).

4. **Upload to TestPyPI (recommended first)**:

   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

   When prompted for credentials:

   - **Username**: `__token__`
   - **Password**: your API token (starting with `pypi-...` or `pats_...`)

5. **Verify the package on TestPyPI**:

   - Visit the TestPyPI website and search for `hsi_statistic`.
   - Optionally install from TestPyPI in a clean environment to test:

     ```bash
     python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple hsi_statistic
     ```

6. **Upload to the real PyPI** (once everything looks good):

   ```bash
   python -m twine upload dist/*
   ```

   Use the same `__token__` username and API token password (but the token must be created on the real PyPI, not TestPyPI).

 
