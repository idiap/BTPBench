# Standalone Plots

These plotting commands generate standalone visualizations that are independent from the experiment-specific plots (DIR, DET, irreversibility curves). Each experiment's own plots are documented in their respective pages.

## Distribution Plot (T-SNE)

Runs feature extraction and generates several views of unprotected templates, plus protected-template views when `btps.algs` is configured:

- `dist_<database>_<baseline>.png`: element-value scatter plot for reference templates from `irreversibility/for_distribution.csv`.
- `norm_<database>_<baseline>.png`: norm of each of those unprotected reference templates.
- `tsne_<database>_<baseline>.png`: t-SNE subject clustering for all verification samples belonging to a random selection of exactly 50 subjects.
- `dist_<database>_<baseline>_<btp-name>.png` and `tsne_<database>_<baseline>_<btp-name>.png`: protected-template plots for each entry in `btps.algs`. No protected norm plot is produced.

Files are written to the experiment's `output_dir`, or to the directory supplied with `-o`.

```bash
uv run btpbench plots distribution \
  -s config/system_config.yaml \
  -e config/experiment_config.yaml \
  [-o ./output_dir]
```

| Argument | Description |
|---|---|
| `-s` | Path to the system configuration file (required). |
| `-e` | Path to the experiment configuration file (required). |
| `-o` | Output directory (optional; overrides the value from the experiment config). |

The distribution pipeline uses the [common fields](experiment_config.md#common-fields) and, when present, `btps.algs`. See [experiment_config.md](experiment_config.md) for configuration details.

Before running it, check the following implementation constraints:

- The database must provide both `irreversibility/for_distribution.csv` and `irreversibility/for_irreversibility.csv`, even though only the distribution references are plotted.
- Its `verification_samples` CSV must contain at least 50 distinct subjects. After missing media files are skipped in compliant mode, more than 30 templates must remain for t-SNE's default perplexity.
- Plot-title mappings currently support only the `icarb`, `soteria`, and `multipie` databases and the `iresnet100`, `iresnet50`, `edgeface`, `edgefacexs`, and `facenet` baselines. Other registered names, including `random`, fail during title construction.
- Protected plot titles are currently hard-coded as PolyProtected with overlap 3. For a different BTP configuration, use the experiment configuration and `<btp-name>` in the filename as the authoritative description.
- All supported BTPs produce protected templates that this plotting command can align with its subject labels.

## Histogram Plot

Generates histogram plots comparing unprotected score distributions (genuine/impostor) with inverted score distributions. Useful for visually assessing irreversibility.

```bash
uv run btpbench plots histogram \
  -u unprotected_scores.csv \
  [-i inversion1.csv -i inversion2.csv ...] \
  [-l "Label1" -l "Label2" ...] \
  -o histogram.png \
  [-t "My Title"] [-f 0.01 -f 0.001] \
  [-c COLOR] [-H HATCH] [-v LEVEL]
```

| Argument | Description |
|---|---|
| `-u` | Path to unprotected score file (required). |
| `-i` | Path to inversion score file(s) (repeatable, optional). |
| `-l` | Custom label(s) for inversion scores (must match number of `-i`). |
| `-o` | Output PNG file (required). |
| `-t` | Plot title (default: `Score Distributions: Unprotected vs Inverted`). |
| `-f` | FMR value(s) to display threshold lines (repeatable, optional). |
| `-c` | Color(s) for inversion distributions (repeatable, optional). When supplied, the number must match the number of `-i` files. |
| `-H` | Hatch pattern(s) for inversion distributions (repeatable, optional). When supplied, the number must match the number of `-i` files. |
| `-v` | Verbosity level 0–4 (default: 3/INFO). |

## PolyProtect Coefficient Plot

Visualizes polynomial coefficients and exponents reconstructed from a key explorer JSON file. The input must contain a non-empty `keys` mapping from threshold names to mappings of entries and integer-compatible PolyProtect keys. Every threshold bucket currently needs at least two keys, and `--nb-coef` must be at least 3 for the correlation-matrix subplot. The `--nb-coef` and `--coef-range` values must match those used to generate the keys. `--overlap` is accepted to construct the PolyProtect object but does not affect coefficient/exponent reconstruction.

The output contains one row per threshold: a coefficient-versus-exponent count heatmap and a Spearman correlation matrix across coefficient positions. There is no combined "all thresholds" subplot; data from all thresholds is used only to establish shared heatmap bins and a shared color scale. The optional JSON output records coefficient value/probability distributions for each threshold and exponent.

```bash
uv run btpbench plots polyprotect_coeff \
  -i key_explorer.json \
  -o coeff_plot.png \
  [--output-json distributions.json] \
  [--overlap N] [--nb-coef N] [--coef-range N] \
  [-t "My Title"] [-v LEVEL]
```

| Argument | Description |
|---|---|
| `-i` | Key explorer JSON file (required). |
| `-o` | Output PNG file (required). |
| `--output-json` | Output JSON with per-threshold, per-exponent coefficient value/probability distributions (optional). |
| `--overlap` | PolyProtect overlap parameter (default: 3). |
| `--nb-coef` | Number of coefficients per segment (default: 5). |
| `--coef-range` | Coefficient range parameter (default: 50). |
| `-t` | Plot title (default: `"Good" Key C/E Distribution`). |
| `-v` | Verbosity level 0–4 (default: 3/INFO). |
