# Biofilm Coverage Analyzer

ImageJ-compatible biofilm fluorescence microscopy analysis utility (CLI + interactive web UI) for **SYTO 9 (green)** and **OYP (orange)** imaging, including **morphotype differentiation** between **large rod-shaped bacteria** and **smaller coccobacilli**.

## Features

- **Single-channel & dual-channel detection:**
  - Automatically detects whether an image contains **SYTO 9 (green)** alone, **OYP (orange)** alone, or **both SYTO 9 + OYP**.
  - Reports **Green (SYTO 9) %**, **Orange (OYP) %**, and **Total Coverage %** (with spectral unmixing for dual-channel micrographs so orange emission does not bleed into the green channel).
- **Morphotype differentiation (Rods vs. Coccobacilli):**
  - Splits foreground objects via Euclidean distance-transform Voronoi seeding and covariance major/minor-axis geometry into **Large Rods**, **Small Coccobacilli**, and optional **Dense Clusters** (or strict 2-way split with `cluster_mode=off`), for both single- and dual-channel images.
- **ImageJ-compatible preprocessing & thresholding:**
  - 65th-percentile-capped rolling-ball background subtraction (prevents eroding large dense microcolonies).
  - 8 thresholding methods: `otsu`, `li`, `isodata`, `triangle`, `moments`, `yen`, `huang`, and `manual`.
  - Automatic bottom-right scale-bar exclusion and optional rectangular ROI cropping.
- **Batch statistics & publication export:**
  - Batch upload, per-group **Mean ± SD** (`ddof=1`), SEM, Median, and IQR.
  - Export **CSV** tables and **300 DPI PNG + vector PDF** summary figures.

## Quick Start (Web UI)

```bash
git clone https://github.com/phageforlifeai/Biofilm-Coverage-Analyzer-.git
cd Biofilm-Coverage-Analyzer-
./run.sh
```

Then open **http://127.0.0.1:8000/** in your browser.

Or install manually in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m biofilm.server --host 0.0.0.0 --port 8000
```

## CLI Usage

```bash
python -m biofilm path/to/images/*.jpg \
  --group auto \
  --method li \
  --bg-radius 35 \
  --morphotypes \
  --csv coverage.csv \
  --plot coverage_summary.png
```

Run `python -m biofilm --help` for all options (`--channel`, `--method`, `--rod-min-area`, `--rod-min-length`, `--cluster-mode`, `--no-unmix`, `--no-exclude-scalebar`, etc.).

## Running Tests

```bash
python tests/test_biofilm.py
```

## License

MIT
