# Kaplan's Preference Matrix in Game Interiors: Computational Pipeline

Code accompanying the manuscript *Genre-Shaped Virtual Space: A Computational Analysis of Kaplan's Preference Matrix in Digital Game Interior Environments*.

The pipeline samples official Steam screenshots of four game genres (Horror, Life Sim, Exploration, Dungeon Crawler), keeps only interior scenes, computes the four dimensions of Kaplan and Kaplan's Preference Matrix (Coherence, Complexity, Legibility, Mystery) from image statistics and monocular depth estimates, and tests whether genre shapes them. No human participants are involved.

Everything is in one file, `kaplan_game_pipeline.py`. Citation details will be added after publication.

## Pipeline

| Stage | What it does |
|---|---|
| `sample` | For each genre, scans the SteamSpy tag list and accepts games whose target tag ranks among their five most-voted tags (and is the best-ranked of the four target tags). Up to 10 gallery thumbnails per accepted game are passed through a two-stage scene gate (Places365 indoor probability and CLIP zero-shot interior classification); the first four passing screenshots are downloaded at full resolution. |
| `metrics` | Estimates depth with MiDaS_small, computes 13 sub-measures, and averages their z-scores into the four composites. |
| `savoias` | Correlates the Complexity composite with the human-derived ranking of the SAVOIAS Interior Design set. |
| `analyze` | Intraclass correlations and mixed models (game as random intercept), MANOVA, permutation test, ANOVA with Tukey-Kramer or Games-Howell comparisons, Welch's ANOVA, exploratory factor analyses, bootstrap test of the predicted axis pairing, clustering, tag-prominence sensitivity analysis; writes figures, tables and an Excel workbook. |
| `crosscheck` | Optional. Compares this script's statistical routines with `statsmodels`, `pingouin` and `factor_analyzer`. |

The statistical routines (MANOVA, random-intercept mixed model, Welch and Games-Howell tests, minimum-residual factor analysis with varimax rotation, KMO and Bartlett's test) are implemented directly with NumPy and SciPy, so the three libraries above are not required for a normal run.

## Requirements

- Python 3.10 or newer (the study was run with Python 3.10)
- An NVIDIA GPU is strongly recommended; the script also runs on CPU but much more slowly
- Internet access (Steam, SteamSpy, model downloads)

## Installation

Create an environment and install PyTorch first, using the command that matches your CUDA version from <https://pytorch.org/get-started/locally/>. On Windows, installing `torch` from PyPI without that command usually gives a CPU-only build. Check the result:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then install the remaining dependencies:

```
pip install -r requirements.txt
```

For the optional cross-check:

```
pip install statsmodels pingouin factor_analyzer
```

## Usage

Run the script from the folder where you want the outputs to be written (all outputs go next to the script).

```
python kaplan_game_pipeline.py --quick
python kaplan_game_pipeline.py
```

`--quick` is a small test run (4 games per genre, reduced resampling) that writes to a `quick_test/` subfolder. Use it first to check the installation, GPU and network access. The full run is the second command.

Other options:

```
python kaplan_game_pipeline.py --stage {all,sample,metrics,savoias,analyze,crosscheck}
python kaplan_game_pipeline.py --cross-check
python kaplan_game_pipeline.py --gate-test path/to/example_images
python kaplan_game_pipeline.py --n-games 100 --max-candidates 1000
python kaplan_game_pipeline.py --force-sample
python kaplan_game_pipeline.py --skip-savoias
```

- `--gate-test FOLDER` runs the images in a folder through the scene gate, prints the decisions, and writes `results/00m_gate_test.csv`. Use it to check that collages, text screens and outdoor or space scenes are rejected.
- `--cross-check` runs the library comparison after the analysis (requires the optional packages).
- `--force-sample` discards the sampling checkpoint and starts sampling again.

Defaults (editable in the `CFG` dictionary at the top of the script): 100 accepted games per genre, up to 1,000 candidates per genre, target tag within the five most-voted tags, up to 10 thumbnails scanned and 4 interior images kept per game, 2,000 permutations, 5,000 bootstrap resamples, random seed 42.

## Outputs

```
images/                 downloaded screenshots, one subfolder per genre
rejected_thumbnails/    thumbnails rejected by the scene gate
figures/                300 dpi PNG figures
results/                CSV tables
cache/                  checkpoints, SteamSpy cache, model weights, SAVOIAS copy
kaplan_game_interiors_all_results_v3.xlsx   all result tables in one workbook
analyzed_images.zip     the images that entered the analysis
```

The CSV files are semicolon-separated with a decimal comma and UTF-8 encoding with a byte-order mark, so that they open directly in Excel with Turkish or other European regional settings. Use `sep=";"` and `decimal=","` when reading them with pandas.

## Runtime and resuming

The sampling stage is limited by SteamSpy's rate limit (about one request per second) and can take an hour or more with the default settings. Progress is checkpointed (`cache/sampling_state.pkl`, `cache/steamspy_appdetails_cache.json`, `cache/metrics_partial.csv`), so an interrupted run continues when you run the same command again. If a SteamSpy or Steam request is throttled, the script waits with exponential back-off and prints a message. In a Windows console, clicking inside the window can pause the program; press Enter to resume.

The first run downloads the Places365 ResNet-18 weights, CLIP ViT-B/32, MiDaS_small (through `torch.hub`) and the SAVOIAS repository.

To rerun the validation, delete `results/02_savoias_validation.csv`.

## Reproducibility

Steam and SteamSpy return live data, so running the sampling stage at a later date may produce a different set of games and screenshots. The corpus analyzed in the paper is documented in `results/00_corpus_index.csv` (sheet `corpus_index` of the workbook). To rerun the measurement and analysis stages on a fixed image set, keep `images/` and `results/00_corpus_index.csv` and run `--stage metrics` followed by `--stage analyze`. All random procedures use a fixed seed.

## Data and copyright

The screenshots are promotional material owned by their publishers. The script downloads them only for local analysis and they should not be redistributed; for the same reason, `images/`, `rejected_thumbnails/` and `analyzed_images.zip` are excluded by `.gitignore`, as are the audit contact sheets (`figures/fig15_*`, `figures/fig16_*`), which contain screenshots. Please comply with the terms of service of Steam and SteamSpy and keep the request delays unchanged.

## Known limitations

- The Legibility sub-measure `legibility_openness_ratio` (share of depth-map pixels above the median) is close to 0.5 for every image by construction, so its standardized score is mostly noise. The manuscript reports a sensitivity analysis without it.
- Item-level measures of sampling adequacy for the 13 sub-measures are computed here from the correlation matrix. `factor_analyzer` returns different item-level values for the same data (the overall KMO of the four composites, test statistics, variance components and loadings agree with the libraries to within 2 x 10^-5).
- Coherence, Legibility and Mystery are not validated against human judgments, and the Complexity composite did not converge with the SAVOIAS ranking.
