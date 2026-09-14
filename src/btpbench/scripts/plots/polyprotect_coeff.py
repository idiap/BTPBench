# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import json
import logging

from collections import Counter
from pathlib import Path

import click
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import spearmanr

from btpbench.btps.polyprotect import PolyProtect
from btpbench.scripts import pipeline_utils

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-file",
    "input_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
    help="Specify JSON file containing key explorer results.",
)
@click.option(
    "-o",
    "--output-file",
    "output_file",
    type=click.Path(dir_okay=False, file_okay=True, exists=False, path_type=Path),
    required=True,
    help="Specify the output file to store plot.",
)
@click.option(
    "--overlap",
    "overlap",
    type=int,
    default=3,
    help="PolyProtect overlap parameter (default: 3).",
)
@click.option(
    "--nb-coef",
    "nb_coef",
    type=int,
    default=5,
    help="PolyProtect number of coefficients parameter (default: 5).",
)
@click.option(
    "--coef-range",
    "coef_range",
    type=int,
    default=50,
    help="PolyProtect coefficient range parameter (default: 50).",
)
@click.option(
    "-t",
    "--title",
    "title",
    type=str,
    default='"Good" Key C/E Distribution',
    required=False,
    help="Specify the plot's title (default: '\"Good\" Key C/E Distribution').",
)
@click.option(
    "--output-json",
    "output_json",
    type=click.Path(dir_okay=False, file_okay=True, exists=False, path_type=Path),
    required=False,
    default=None,
    help="Specify an output JSON file to store per-exponent coefficient distributions.",
)
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    type=click.IntRange(0, 4),
    default=3,
    help=(
        "Set verbosity level: 0=CRITICAL (silent), 1=ERROR, 2=WARNING, 3=INFO (default), 4=DEBUG."
    ),
)
def main(
    input_file: Path,
    output_file: Path,
    overlap: int,
    nb_coef: int,
    coef_range: int,
    title: str,
    output_json: Path | None,
    verbosity: int,
) -> None:
    """Plot coefficient/exponent scatter plot from PolyProtect keys."""

    pipeline_utils.setup_logger_from_verbosity(verbosity)

    # Create a PolyProtect object with the given parameters
    config = {
        "overlap": overlap,
        "nb_coef": nb_coef,
        "coef_range": coef_range,
    }
    polyprotect = PolyProtect(
        work_dir=Path(),
        save=False,
        config=config,
    )

    logger.info(
        f"PolyProtect parameters: overlap={overlap}, nb_coef={nb_coef}, coef_range={coef_range}"
    )

    # Load JSON file
    with input_file.open() as f:
        data = json.load(f)

    thresholds_keys = data["keys"]
    thresholds = list(thresholds_keys.keys())

    logger.info(f"Loaded {len(thresholds)} thresholds from {input_file}")

    # Collect coefficient/exponent data per threshold
    threshold_data = {}
    for threshold in thresholds:
        key_data = thresholds_keys[threshold]
        all_coefficients = []
        all_exponents = []

        logger.info(f"Processing threshold {threshold} with {len(key_data)} keys")

        for _, key in key_data.items():
            secret = polyprotect.get_secret(int(key))
            # Sort by exponent (ascending) and reorder coefficients accordingly
            sort_idx = np.argsort(secret["exponents"])
            all_coefficients.extend(np.array(secret["coefficients"])[sort_idx])
            all_exponents.extend(np.array(secret["exponents"])[sort_idx])

        threshold_data[threshold] = (
            np.array(all_coefficients),
            np.array(all_exponents),
        )

    # Create subplots: two per threshold (heatmap + correlation matrix)
    n_thresholds = len(thresholds)
    ncols = 2
    nrows = n_thresholds
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.96)

    # Combine all data for shared color scale
    global_coefficients = np.concatenate([c for c, _ in threshold_data.values()])
    global_exponents = np.concatenate([e for _, e in threshold_data.values()])

    # Build bin edges aligned to integer exponents and evenly-spaced coefficients
    unique_exponents = sorted(set(global_exponents))
    exp_edges = [e - 0.5 for e in unique_exponents] + [unique_exponents[-1] + 0.5]
    coef_bins = 30
    coef_min = global_coefficients.min()
    coef_max = global_coefficients.max()
    coef_edges = np.linspace(coef_min - 0.5, coef_max + 0.5, coef_bins + 1)
    bins = [coef_edges, exp_edges]

    # Compute shared color scale (log) from the combined data
    counts, _, _ = np.histogram2d(global_coefficients, global_exponents, bins=bins)
    vmax = counts.max()
    norm = mcolors.LogNorm(vmin=1, vmax=max(vmax, 2))

    def _plot_heatmap(ax, coefficients, exponents, subplot_title):
        h = ax.hist2d(
            coefficients,
            exponents,
            bins=bins,
            cmap="viridis",
            norm=norm,
            cmin=1,
        )
        ax.set_xlabel("Coefficient (C)")
        ax.set_ylabel("Exponent (E)")
        ax.set_yticks(unique_exponents)
        for edge in exp_edges:
            ax.axhline(y=edge, color="white", linewidth=0.8, alpha=0.7)
        ax.set_title(subplot_title)
        ax.grid(True, alpha=0.3)
        return h

    for idx, threshold in enumerate(thresholds):
        coefficients, exponents = threshold_data[threshold]
        n_keys = len(thresholds_keys[threshold])

        # Heatmap subplot
        ax_heatmap = axes[idx][0]
        _ = _plot_heatmap(
            ax_heatmap,
            coefficients,
            exponents,
            f"Threshold: {threshold} ({n_keys} keys)",
        )

        # Correlation matrix subplot
        ax_corr = axes[idx][1]
        coeffs_matrix = coefficients.reshape(-1, nb_coef)
        corr_matrix, _ = spearmanr(coeffs_matrix)
        im = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax_corr.set_title(f"Spearman Correlation (threshold: {threshold})")
        ax_corr.set_xlabel("Coefficient (C) index")
        ax_corr.set_ylabel("Coefficient (C) index")
        ax_corr.set_xticks(np.arange(nb_coef))
        ax_corr.set_yticks(np.arange(nb_coef))
        for i in range(nb_coef):
            for j in range(nb_coef):
                ax_corr.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                )
        fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    # Shared colorbar for heatmap column
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(
        mappable, ax=[axes[r][0] for r in range(nrows)], label="Count", shrink=0.8
    )

    # Save the plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    logger.info(f"Plot saved to {output_file}")

    # Write per-exponent coefficient distributions to JSON
    if output_json is not None:

        def _compute_distribution(coefficients, exponents):
            result = {}
            for exp_val in unique_exponents:
                mask = exponents == exp_val
                coef_values = coefficients[mask]
                counts = Counter(int(v) for v in coef_values)
                total = sum(counts.values())
                sorted_values = sorted(counts.keys())
                result[int(exp_val)] = {
                    "values": sorted_values,
                    "probabilities": [counts[v] / total for v in sorted_values],
                }
            return result

        distributions = {}
        for threshold in thresholds:
            coefficients, exponents = threshold_data[threshold]
            distributions[threshold] = _compute_distribution(coefficients, exponents)

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(distributions, f, indent=2)
        logger.info(f"Distributions saved to {output_json}")


if __name__ == "__main__":
    main()
