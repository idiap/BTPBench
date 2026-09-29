# MOBIO User-Specific Quickstart

This recipe evaluates normalized, user-specific PolyProtect on
MOBIO with EdgeFace. Run every command from the repository root. Generated
keys, templates, scores, metrics, and plots are stored in
`quickstart/output/`.

## 0. Prepare

1. Obtain MOBIO under its own terms.
2. Replace `/absolute/path/to/mobio` in
   [`configs/system_config.yaml`](configs/system_config.yaml) with its absolute
   local path.
3. Run `uv sync`.

EdgeFace downloads and caches its upstream weights on first use. That first run
requires network access and may write to the user-level Torch Hub cache.
The configs use compliant mode, so unusable acquisitions appear in FTA and log
warnings. Pipelines skip existing score outputs; add `--override` only when
deliberately replacing one stage's outputs.

## 1. Measure baseline verification

```bash
uv run btpbench verification pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/01_baseline.yaml

uv run btpbench verification metrics \
  -s quickstart/output/verification-edgeface.csv \
  -l "EdgeFace" \
  -s quickstart/output/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "PolyProtect: ordinary user keys" \
  -f 0.1 \
  -o quickstart/output/01-verification-metrics.csv

uv run btpbench verification plots \
  -f quickstart/output/verification-edgeface.csv \
  -l "EdgeFace" \
  -f quickstart/output/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "PolyProtect: ordinary user keys" \
  -t "MOBIO verification: ordinary user keys" \
  -o quickstart/output/01-verification-det.png
```

Creates:

```text
quickstart/output/verification-edgeface.csv
quickstart/output/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv
quickstart/output/01-verification-metrics.csv
quickstart/output/01-verification-det.png
```

Inspect `fnmr_1000.0` in the metrics CSV and the DET curves. Lower FNMR at the
same FMR is better. See the [verification guide](../docs/verification.md) for
details.

## 2. Measure baseline irreversibility

```bash
uv run btpbench irreversibility pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/01_baseline.yaml

uv run btpbench irreversibility metrics \
  -i quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Ordinary user keys" \
  -s quickstart/output/verification-edgeface.csv \
  -f 0.1 \
  -o quickstart/output/02-baseline-irreversibility-metrics.csv

uv run btpbench irreversibility plots \
  -i quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Ordinary user keys" \
  -s quickstart/output/verification-edgeface.csv \
  -t "MOBIO irreversibility: ordinary user keys" \
  -o quickstart/output/02-baseline-isr.png

uv run btpbench plots histogram \
  -u quickstart/output/verification-edgeface.csv \
  -i quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Ordinary user keys" \
  -f 0.1 \
  -t "MOBIO inversion scores: ordinary user keys" \
  -o quickstart/output/02-baseline-irreversibility-histogram.png
```

Creates:

```text
quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv
quickstart/output/02-baseline-irreversibility-metrics.csv
quickstart/output/02-baseline-isr.png
quickstart/output/02-baseline-irreversibility-histogram.png
```

Inspect `success_rate_far0.1000`, the ISR curve, and the inversion histogram.
Lower inversion success is better. See the
[irreversibility guide](../docs/irreversibility.md) for details.

## 3. Select one key per user

```bash
uv run btpbench keyselection pipeline_user \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/01_baseline.yaml \
  -v quickstart/output/verification-edgeface.csv \
  -f 0.1
```

Creates:

```text
quickstart/output/keys_select_fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.json
quickstart/output/ref_keys_select_fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv
```

The JSON maps each MOBIO subject ID to one selected integer key. The shipped
MOBIO selection templates overlap the irreversibility attack templates, so the
next comparison is not an independent security estimate. See the
[key-selection guide](../docs/key_selection.md) for details.

## 4. Repeat irreversibility with selected keys

```bash
uv run btpbench irreversibility pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/02_selected_keys.yaml

uv run btpbench irreversibility metrics \
  -i quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Ordinary user keys" \
  -i quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/irreversibility-dictionary-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Selected user keys" \
  -s quickstart/output/verification-edgeface.csv \
  -f 0.1 \
  -o quickstart/output/04-irreversibility-comparison-metrics.csv

uv run btpbench irreversibility plots \
  -i quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Ordinary user keys" \
  -i quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/irreversibility-dictionary-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Selected user keys" \
  -s quickstart/output/verification-edgeface.csv \
  -t "MOBIO irreversibility: ordinary vs selected keys" \
  -o quickstart/output/04-isr-comparison.png

uv run btpbench plots histogram \
  -u quickstart/output/verification-edgeface.csv \
  -i quickstart/output/irreversibility-subject-id-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Ordinary user keys" \
  -i quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/irreversibility-dictionary-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "Selected user keys" \
  -f 0.1 \
  -t "MOBIO inversion scores: ordinary vs selected keys" \
  -o quickstart/output/04-irreversibility-comparison-histogram.png
```

Creates:

```text
quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/irreversibility-dictionary-systems1-trials10-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface.csv
quickstart/output/04-irreversibility-comparison-metrics.csv
quickstart/output/04-isr-comparison.png
quickstart/output/04-irreversibility-comparison-histogram.png
```

Verify that selected keys reduce match rate, inversion-success rate, and the ISR
curve at the chosen operating point. See the
[irreversibility guide](../docs/irreversibility.md) for details.

## 5. Repeat verification with selected keys

```bash
uv run btpbench verification pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/02_selected_keys.yaml

uv run btpbench verification metrics \
  -s quickstart/output/verification-edgeface.csv \
  -l "EdgeFace" \
  -s quickstart/output/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "PolyProtect: ordinary user keys" \
  -s quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "PolyProtect: selected user keys" \
  -f 0.1 \
  -o quickstart/output/05-verification-comparison-metrics.csv

uv run btpbench verification plots \
  -f quickstart/output/verification-edgeface.csv \
  -l "EdgeFace" \
  -f quickstart/output/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "PolyProtect: ordinary user keys" \
  -f quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv \
  -l "PolyProtect: selected user keys" \
  -t "MOBIO verification: ordinary vs selected keys" \
  -o quickstart/output/05-verification-comparison-det.png
```

Creates:

```text
quickstart/output/fmr1000-legacy_None-minimize_cos-3-100-mobio-normalized_polyprotect_usr_3_5_50-edgeface/verification-normalized_polyprotect_usr_3_5_50-edgeface.csv
quickstart/output/05-verification-comparison-metrics.csv
quickstart/output/05-verification-comparison-det.png
```

Compare `fnmr_1000.0` for ordinary and selected user keys. The increase is the
recognition-accuracy cost of the selected keys. See the
[verification guide](../docs/verification.md) for details.

## 6. Generate random and selected key pools

```bash
uv run python quickstart/make_random_key_pool.py

uv run btpbench keyselection key_explorer \
  -e quickstart/configs/03_key_explorer.yaml \
  -t -0.8 \
  -t -0.9 \
  -t -1.0 \
  --time-limit 172200 \
  --batch-size 30 \
  --vector-dim 512 \
  --n-dist-samples 1000
```

Creates:

```text
quickstart/output/random-key-pool.json
quickstart/output/key_explorer_-0.8_-0.9_-1.0-legacy_None-minimize_cos-3-100-normalized_polyprotect_usr_3_5_50.json
```

The `random` bucket contains deterministic, unselected integer seeds 0 through
59. In the explorer JSON, verify that
`metadata.keys_per_threshold["-1.0"]` is at least 60 before continuing. The
172,200-second search lasts 47 hours 50 minutes. The explorer has no exposed
random seed, so separate runs need not select identical keys. See the
[key-selection guide](../docs/key_selection.md) for details.

## 7. Compare unlinkability

```bash
uv run btpbench unlinkability pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/04_unlinkability_diversity.yaml \
  -k quickstart/output/random-key-pool.json \
  -b random \
  --samples-per-subject 60 \
  --non-mated-samples-per-subject 10 \
  --seed 42

uv run btpbench unlinkability pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/04_unlinkability_diversity.yaml \
  -k quickstart/output/key_explorer_-0.8_-0.9_-1.0-legacy_None-minimize_cos-3-100-normalized_polyprotect_usr_3_5_50.json \
  --keys-bucket=-1.0 \
  --samples-per-subject 60 \
  --non-mated-samples-per-subject 10 \
  --seed 42

uv run btpbench unlinkability plots \
  -m quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-mated.csv \
  -n quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-non-mated-10.csv \
  -t "MOBIO unlinkability: random control keys" \
  --metric-bins 100 \
  --omega 1.0 \
  --d-local-threshold 0.5 \
  --x-min -2 \
  --x-max 0 \
  -o quickstart/output/07-unlinkability-random.png

uv run btpbench unlinkability plots \
  -m quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-mated.csv \
  -n quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-non-mated-10.csv \
  -t "MOBIO unlinkability: selected keys" \
  --metric-bins 100 \
  --omega 1.0 \
  --d-local-threshold 0.5 \
  --x-min -2 \
  --x-max 0 \
  -o quickstart/output/07-unlinkability-selected.png
```

Creates:

```text
quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-mated.csv
quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-non-mated-10.csv
quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-mated.csv
quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-non-mated-10.csv
quickstart/output/07-unlinkability-random.png
quickstart/output/07-unlinkability-selected.png
```

Each plot contains the mated and non-mated densities, local `D(s)`, and `Dsys`.
Lower `Dsys` means better unlinkability. See the
[unlinkability guide](../docs/unlinkability.md) for details.

## 8. Compare diversity

The diversity pipelines perform exact clique searches and may be expensive.

```bash
uv run btpbench diversity pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/04_unlinkability_diversity.yaml \
  -k quickstart/output/random-key-pool.json \
  -b random \
  --unlinkability-mated-score-file quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-mated.csv \
  --unlinkability-non-mated-score-file quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-non-mated-10.csv \
  --metric-bins 100 \
  --omega 1.0 \
  --d-local-threshold 0.5 \
  --x-min -2 \
  --x-max 0 \
  --seed 42 \
  --real-sample-set unlinkability

uv run btpbench diversity pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/04_unlinkability_diversity.yaml \
  -k quickstart/output/key_explorer_-0.8_-0.9_-1.0-legacy_None-minimize_cos-3-100-normalized_polyprotect_usr_3_5_50.json \
  --keys-bucket=-1.0 \
  --unlinkability-mated-score-file quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-mated.csv \
  --unlinkability-non-mated-score-file quickstart/output/unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-non-mated-10.csv \
  --metric-bins 100 \
  --omega 1.0 \
  --d-local-threshold 0.5 \
  --x-min -2 \
  --x-max 0 \
  --seed 42 \
  --real-sample-set unlinkability

uv run btpbench diversity plots \
  -f quickstart/output/diversity-mobio-unlinkability-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-threshold0d5.csv \
  -t "MOBIO diversity: random control keys" \
  -o quickstart/output/08-diversity-random.png

uv run btpbench diversity plots \
  -f quickstart/output/diversity-mobio-unlinkability-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-threshold0d5.csv \
  -t "MOBIO diversity: selected keys" \
  -o quickstart/output/08-diversity-selected.png

uv run btpbench diversity unlinkability-summary \
  -i quickstart/output \
  -o quickstart/output/08-diversity-summary.csv
```

Creates:

```text
quickstart/output/diversity-mobio-unlinkability-normalized_polyprotect_usr_3_5_50-edgeface-keysrandom-60-threshold0d5.csv
quickstart/output/diversity-mobio-unlinkability-normalized_polyprotect_usr_3_5_50-edgeface-keysm1d0-60-threshold0d5.csv
quickstart/output/08-diversity-random.png
quickstart/output/08-diversity-selected.png
quickstart/output/08-diversity-summary-biohash.csv
quickstart/output/08-diversity-summary-polyprotect.csv
quickstart/output/08-diversity-summary-combined-polyprotect.csv
```

Inspect `08-diversity-summary-polyprotect.csv` and both violin plots. Larger
mean maximum-clique size means more mutually non-matching protected templates
and better diversity. See the [diversity guide](../docs/diversity.md) for
details. The generic summary command also writes empty BioHash and combined
PolyProtect family files in this PolyProtect-only recipe.

## 9. Measure user key-selection cost

This measures the time spent searching for each subject's key and the number
of distinct candidate keys tested, at 5%, 10%, and 20% FMR:

```bash
uv run btpbench keyselection pipeline_user_cost \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/01_baseline.yaml \
  -v quickstart/output/verification-edgeface.csv \
  -f 0.05 -f 0.1 -f 0.2
```

Read `elapsed_seconds` and `n_trials` in the subject and final `mean` rows of
the three `key_selection_cost_fmr...csv` files. Feature extraction is excluded
from the timings. Record hardware and worker count when comparing runs. This
command performs fresh selection and does not replace the JSON from step 3.
See [key-selection cost](../docs/key_selection.md#key-selection-cost).

## 10. Compare online user-specific unlinkability

This evaluates ten distinct keys per subject, with each selected key tested
against its own sample. It uses the first ten samples in `mobio_unlink.csv` and
does not require the shared pools from step 6.

```bash
uv run btpbench unlinkability online_pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/01_baseline.yaml \
  --random \
  --samples-per-subject 10 --non-mated-samples-per-subject 10 \
  --seed 42 \
  -o quickstart/output/online-random

uv run btpbench unlinkability online_pipeline \
  -s quickstart/configs/system_config.yaml \
  -e quickstart/configs/01_baseline.yaml \
  -v quickstart/output/verification-edgeface.csv \
  -f 0.1 \
  --samples-per-subject 10 --non-mated-samples-per-subject 10 \
  --seed 42 \
  -o quickstart/output/online-selected

uv run btpbench unlinkability plots \
  -m quickstart/output/online-selected/online-unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-selected-ksfmr0d1-poolrandomspace-keyseed42-minimize_cos-3-100-n10-nm10-seed42-mated.csv \
  -n quickstart/output/online-selected/online-unlinkability-mobio-normalized_polyprotect_usr_3_5_50-edgeface-selected-ksfmr0d1-poolrandomspace-keyseed42-minimize_cos-3-100-n10-nm10-seed42-non-mated.csv \
  --metric-bins 100 --omega 1 --x-min -2 --x-max 0 \
  -t "MOBIO unlinkability: online selected keys" \
  -o quickstart/output/online-selected/unlinkability.png
```

Repeat the plot with the CSVs in `online-random/` (replace
`selected-ksfmr0d1` with `random` in their names). Compare `Dsys`; lower is
better. Each run also writes a JSON audit with accepted keys, sample IDs,
candidate counts, and timing. `--random` skips inversion filtering; the selected
run uses 10% FMR (`0.1`). The configuration's `key_sampling_seed` controls key
assignment, while `--seed` controls non-mated sample sampling. See
[online unlinkability](../docs/unlinkability.md#online-user-specific-key-selection).

## 11. Decide

- Selected keys should reduce ISR.
- Selected keys may increase verification FNMR.
- Lower `Dsys` is better.
- Larger diversity cliques are better.
- Key-selection time and candidate counts quantify the enrollment cost.

Keep the selected keys only if their observed security gain justifies their
observed verification cost.
