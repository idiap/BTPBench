# PolyProtect irreversibility: experiment guide

**A Deeper Dive into the Irreversibility of PolyProtect: Making Protected Face
Templates Harder to Invert**

Vedrana Krivokuća Hahn, Jérémy Maceiras, Sébastien Marcel.

This paper is under review at
[IEEE Transactions on Information Forensics and Security (TIFS)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10206)
and is available as a preprint on [arXiv:2605.03857](https://arxiv.org/abs/2605.03857).

To cite the preprint:

```bibtex
@misc{krivokucahahn2026deeperdive,
  title         = {A Deeper Dive into the Irreversibility of {PolyProtect}: Making Protected Face Templates Harder to Invert},
  author        = {Krivoku{\'c}a Hahn, Vedrana and Maceiras, J{\'e}r{\'e}my and Marcel, S{\'e}bastien},
  year          = {2026},
  eprint        = {2605.03857},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2605.03857},
  url           = {https://arxiv.org/abs/2605.03857},
  note          = {Under review at IEEE Transactions on Information Forensics and Security (TIFS)}
}
```

This guide starts with unprotected accuracy, element-range plots, and t-SNE
plots. The main verification and inversion example then uses **SOTERIA,
iResNet100, and normalized PolyProtect with overlap 3**, selects user-specific
keys, and repeats the measurements. Section 10 maps the other paper experiments
to their configurations. Paper references below use the manuscript's section, figure,
and table numbers. **Random keys (R)** and **selected keys (KS)** correspond to
the two conditions in Tables III-V.

## 1. Prepare

1. Follow the [installation instructions](../../README.md#installation).
2. Obtain SOTERIA and set its `dataset_dir` in
   [system_config.yaml](configs/system_config.yaml).
3. Run the commands below from the repository root.

The configurations use MediaPipe. The inversion configurations use cosine
minimization with L-BFGS-B (`minimize_cos`), up to five solver guesses per
inversion attempt, ten attack trials, and seed 42. Results are saved under
`paper/polyprotect-irreversibility/output/soteria-iresnet100/`.
The guide uses the original protocol CSVs in the repository's
[protocols/](../../protocols/) directory.

## 2. Measure unprotected accuracy

**Paper:** Section II-B, Fig. 3 (verification accuracy of the five face
recognition models before applying PolyProtect).

Use the provided [unprotected.yaml](configs/unprotected.yaml), configured for
SOTERIA and iResNet100. It omits `btps` and shares its output directory with
the later protected experiment so that the baseline scores can be reused:

```bash
uv run btpbench verification pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/unprotected.yaml
```

This creates `verification-iresnet100.csv` in
`paper/polyprotect-irreversibility/output/soteria-iresnet100/`. Compute the
unprotected FNMR and plot its DET curve:

```bash
uv run btpbench verification metrics \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -l "Unprotected: iResNet100" \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/unprotected-metrics.csv

uv run btpbench verification plots \
  -f paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -l "Unprotected: iResNet100" \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/unprotected-det.png
```

Read `fnmr_10.0` for 0.1% FMR and `fnmr_1.0` for 0.01% FMR.
Multiply these rates by 100 to report percentages. All CLI FMR values are
fractions: `0.001` means 0.1%.

For Fig. 3, repeat for all five `bio_alg` values (`iresnet50`, `iresnet100`,
`edgeface`, `edgefacexs`, and `facenet`) on all three datasets (`multipie`,
`soteria`, and `icarb`). Use a separate `output_dir` for each dataset/model
pair. For each dataset, pass its five verification CSVs to `verification plots`
with one `-f` and corresponding `-l` per model to compare their DET curves.
The main experiments below use the selected iResNet100 and EdgeFace models.

## 3. Plot template element ranges and t-SNE clustering

**Paper:** Section III, Fig. 5 (element ranges) and Fig. 6 (class/identity
separation), before and after PolyProtect using SOTERIA, iResNet100 and
EdgeFace, and overlap 3.

Use the provided [distribution.yaml](configs/distribution.yaml), configured
for SOTERIA, iResNet100, overlap 3, and `normalize_input: false`. It writes to
`paper/polyprotect-irreversibility/output/soteria-iresnet100-unnormalized/`.
Both figures use embeddings **before input normalization**. One command
generates both the range and t-SNE plots:

```bash
uv run btpbench plots distribution \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/distribution.yaml
```

For Fig. 5, compare these files in the configured output directory:

- `dist_soteria_iresnet100.png`: unprotected element ranges.
- `dist_soteria_iresnet100_unnormalized_polyprotect_usr_3_5_50.png`:
  protected element ranges with random keys.

For Fig. 6, compare the t-SNE files from the same run:

- `tsne_soteria_iresnet100.png`: unprotected identity clustering.
- `tsne_soteria_iresnet100_unnormalized_polyprotect_usr_3_5_50.png`:
  protected identity clustering with random keys and overlap 3.

Repeat with `bio_alg: edgeface` and an output directory ending in
`soteria-edgeface-unnormalized` to obtain the other model's plots for both
figures. Their filenames use `edgeface` in place of `iresnet100`. The command
also generates an unprotected template norm plot (`norm_*.png`).

The t-SNE plots show how PolyProtect changes the spread and separation of identities.
The command selects 50 subjects from `verification_samples` for t-SNE; keep
that full sample pool available. See the
[distribution plot reference](../../docs/plots.md#distribution-plot-t-sne)
for sampling details and output descriptions.

Continue with the original, normalized `experiment.yaml` for the remaining
experiments.

## 4. Measure verification with random keys

**Paper:** Section III (accuracy), especially Fig. 7 and Table I for normalized
inputs; Fig. 4 and Table I cover the unnormalized comparison. The unprotected
baseline is evaluated in Section II-B, Fig. 3.

```bash
uv run btpbench verification pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml
```

This reuses the unprotected scores from step 2 and creates protected
verification scores. Compare their FNMR at 0.1% and 0.01% FMR:

```bash
uv run btpbench verification metrics \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -l "Unprotected" \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "PolyProtect: random keys" \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-metrics.csv
```

To plot the DET curves:

```bash
uv run btpbench verification plots \
  -f paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -l "Unprotected" \
  -f paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "PolyProtect: random keys" \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-det.png
```

## 5. Measure inversion with random keys

**Paper:** Section IV (irreversibility), Figs. 8-9; these scores also provide
the random-key comparison in Section V, Fig. 10 and Table III.

The configuration's `method` values map to the solver labels in Fig. 9 as
follows. Keep the configuration identifiers in YAML and filenames; use the
paper labels when presenting or plotting results.

| Configuration `method` | Paper label (Fig. 9) | Numerical solver |
|---|---|---|
| `root_l2` | Inverted (Euclidean - root) | `scipy.optimize.root`, `lm` |
| `minimize_l2` | Inverted (Euclidean - minimize) | `scipy.optimize.minimize`, `L-BFGS-B`, squared Euclidean objective |
| `minimize_cos` | Inverted (Cosine - minimize) | `scipy.optimize.minimize`, `L-BFGS-B`, cosine objective |

This example uses `minimize_cos`, the cosine-based attacker used for key
selection and its evaluation in Section V. Fig. 8 compares `root_l2` with
`minimize_cos`, labelled **Inverted (Euclidean)** and **Inverted (Cosine)** in
that figure. Fig. 9 adds `minimize_l2` to isolate the effect of the distance
function. All three attacks are evaluated against the original unprotected
embedding using cosine comparison scores, regardless of the solver objective.

```bash
uv run btpbench irreversibility pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml
```

## 6. Select keys and repeat the measurements

**Paper:** Section V (key selection algorithm), Fig. 10 and Table III;
Section V-A evaluates the resulting accuracy in Fig. 11 and Table IV.

Select one key per user at the paper's **20% FMR** threshold:

```bash
uv run btpbench keyselection pipeline_user \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml \
  -v paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -f 0.2
```

The command creates a subject-to-key JSON file.
[selected_keys.yaml](configs/selected_keys.yaml) already points to that file.
Use this configuration to evaluate the selected keys:

```bash
uv run btpbench irreversibility pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/selected_keys.yaml

uv run btpbench verification pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/selected_keys.yaml
```

Selected-key scores are stored in a separate `fmr2000-...` subdirectory under
`output_dir`. The `fmr2000` prefix comes from the key-selection option `-f 0.2`:
filenames encode the FMR fraction as `int(fmr * 10000)`, so
`int(0.2 * 10000) = 2000` represents **20% FMR**. This is the threshold used
to select keys; the evaluation thresholds in the next section are set separately.

The full directory name comes from `key_dictionary_file` in
[selected_keys.yaml](configs/selected_keys.yaml): the pipeline removes the
`keys_select_` prefix and `.json` extension from the selected-key filename.
For example, `keys_select_fmr2000-<settings>.json` produces the score directory
`<output_dir>/fmr2000-<settings>/`.

The remaining suffix records the key-selection settings (`legacy_None` means
`ks_method: legacy` with `ks_n_elements` unset), inversion settings
(`minimize_cos-3-5`: method, precision, and maximum number
of guesses), dataset (`soteria`), PolyProtect configuration
(`normalized_polyprotect_usr_3_5_50`), and face model (`iresnet100`).

## 7. Compare random and selected keys

**Paper:** Section V, Fig. 10 and Table III (ISR); Section V-A, Fig. 11 and
Table IV (verification accuracy). Tables report both iResNet100 and EdgeFace;
Figs. 10-11 show iResNet100.

Compute inversion success rates (ISR) using the same unprotected verification
scores to determine the match thresholds:

```bash
uv run btpbench irreversibility metrics \
  -i paper/polyprotect-irreversibility/output/soteria-iresnet100/irreversibility-subject-id-systems1-trials10-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Random keys" \
  -i paper/polyprotect-irreversibility/output/soteria-iresnet100/fmr2000-legacy_None-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100/irreversibility-dictionary-systems1-trials10-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Selected keys" \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/isr-comparison.csv
```

The ISR column names encode each `-f` value as a fraction formatted to four
decimal places, using `success_rate_far{f:.4f}`. The `far` label comes from
the command's `--far` option (false accept rate); with the unprotected
verification scores used here, it corresponds to the paper's FMR operating point:

| CLI option | Evaluation FMR | ISR column |
|---|---|---|
| `-f 0.001` | 0.1% | `success_rate_far0.0010` |
| `-f 0.0001` | 0.01% | `success_rate_far0.0001` |

Each cell stores successful inversion attempts divided by **all attempts**,
including unsolved attempts in the denominator. The values are fractions;
multiply by 100 for the ISR percentages in Table III. These evaluation FMRs
are independent of the 20% FMR used to select keys in section 6.

Compare verification accuracy for Table IV:

```bash
uv run btpbench verification metrics \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Random keys" \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/fmr2000-legacy_None-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Selected keys" \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/fnmr-comparison.csv
```

For Fig. 11, use the same two verification files with `verification plots`,
as in step 4. For Fig. 10, pass the two inversion files to
[`plots histogram`](../../docs/plots.md#histogram-plot), together with the
unprotected verification file. Lower ISR indicates better inversion resistance;
lower FNMR indicates better recognition accuracy.

## 8. Compare online unlinkability

**Paper:** Section V-B, Table V (random versus selected keys); Section V-C,
Table VI repeats the comparison at different key-selection thresholds.

Use the random-key configuration: this command selects a new key for each
sample during protection. It keeps ten distinct keys within each subject and
records the sample-to-key assignments in a JSON audit file.

```bash
uv run btpbench unlinkability online_pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml \
  --random \
  --samples-per-subject 10 --non-mated-samples-per-subject 10 \
  --seed 42 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/online-random

uv run btpbench unlinkability online_pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml \
  -v paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -f 0.2 \
  --samples-per-subject 10 --non-mated-samples-per-subject 10 \
  --seed 42 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected
```

`--random` assigns keys without inversion filtering. The second command tests
candidate keys against each sample at 20% FMR.

Plot the selected-key scores and read `Dsys` from the figure title or log:

```bash
uv run btpbench unlinkability plots \
  -m paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected/online-unlinkability-soteria-normalized_polyprotect_usr_3_5_50-iresnet100-selected-ksfmr0d2-poolrandomspace-keyseed42-minimize_cos-3-5-n10-nm10-seed42-mated.csv \
  -n paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected/online-unlinkability-soteria-normalized_polyprotect_usr_3_5_50-iresnet100-selected-ksfmr0d2-poolrandomspace-keyseed42-minimize_cos-3-5-n10-nm10-seed42-non-mated.csv \
  --metric-bins 10 --omega 1 \
  -t "SOTERIA: selected user keys" \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected/unlinkability.png
```

Repeat with the two CSVs in `online-random/` to measure the random-key
condition. These use `random` instead of `selected-ksfmr0d2` in their names.
`Dsys` is the paper's global unlinkability measure, $D_{\leftrightarrow}^{sys}$:
0 means full unlinkability and 1 means full linkability. This example uses the
current metric defaults; the manuscript does not specify the bin count or
score range, so matching its numerical values requires those original settings.
An optional `-k` restricts selection to a supplied candidate pool; see
the [online unlinkability reference](../../docs/unlinkability.md#online-user-specific-key-selection).

## 9. Measure key-selection cost

**Paper:** Section V-D, Table VII (average search time) and Table VIII (average
number of failed keys); the thresholds are introduced in Section V-C.

Measure actual search time and candidate counts for each subject at the three
selection thresholds:

```bash
uv run btpbench keyselection pipeline_user_cost \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml \
  -v paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -f 0.05 -f 0.1 -f 0.2
```

Each `key_selection_cost_fmr...csv` contains per-subject `elapsed_seconds` and
`n_trials`, followed by a `row_type: mean` row. A trial is a distinct candidate
key, including the accepted one. Use the mean row's `elapsed_seconds` for
Table VII. Table VIII counts **failed keys**, so use `n_trials - 1` for each
subject with `status: selected`, or subtract 1 from the mean row's `n_trials`.
The mean includes only successful selections; check `n_selected` against
`n_subjects` when reporting results.

Timing excludes feature extraction. Record hardware and `num_processes`
alongside the CSVs. For Tables VII-VIII, change to Multi-PIE and evaluate both
iResNet100 and EdgeFace at overlaps 0 and 3. Candidate keys are tested
sequentially within each subject's search; `num_processes` controls parallel
subject searches.

## 10. Run the other paper experiments

**Paper:** Sections II-B, III, IV and V, Figs. 3-11 and Tables I, III-VIII.
The result map below gives the section and configuration for each experiment;
Fig. 2 and Table II describe the transform dimensions.

Edit [experiment.yaml](configs/experiment.yaml) and, when evaluating selected
keys, make the same changes in [selected_keys.yaml](configs/selected_keys.yaml).
Update the paths in the commands to match the new outputs.

| Change | What to edit |
|---|---|
| Dataset | Set `database` to `multipie` or `icarb`; fill in that dataset's `dataset_dir` in the system configuration. |
| Face model | Set `bio_alg` to `edgeface`. Fig. 3 also uses `iresnet50`, `edgefacexs`, and `facenet`. |
| Overlap | Change `overlap` to 0-4 for random keys, or 0-3 for selected keys. Section V excludes overlap 4 because key selection did not find suitable keys for every template on Multi-PIE and iCarB-Face. |
| Normalization | Set `normalize_input: false` for unnormalized experiments; filenames then start with `unnormalized_polyprotect`. |
| Solver | Set `method` to `root_l2`, `minimize_l2`, or `minimize_cos`; see the [solver-to-paper label mapping](#5-measure-inversion-with-random-keys). |
| Selection threshold | Set `-f 0.05` for **5% FMR**, `-f 0.1` for **10% FMR**, or `-f 0.2` for **20% FMR**. Key filenames encode this fraction as `int(fmr * 10000)`: `0.05 * 10000 = 500`, `0.1 * 10000 = 1000`, and `0.2 * 10000 = 2000`, giving the tags `fmr500`, `fmr1000`, and `fmr2000`, respectively. See [section 6](#6-select-keys-and-repeat-the-measurements) for how the JSON filename determines the score subdirectory. |

Use a separate `output_dir` for each dataset, model, and normalization setting.
After selecting keys for a different setting, update `key_dictionary_file` in
`selected_keys.yaml` to the new JSON file. Keep normalized inputs and
`method: minimize_cos` for key selection. Use a new output directory when
changing settings. Matching completed runs are skipped; use `--override`
when deliberately repeating them.

| Paper section | Paper result | Experiments to run |
|---|---|---|
| II-B | Fig. 3 | Repeat [step 2](#2-measure-unprotected-accuracy) on all three datasets and all five models. |
| III | Figs. 4 and 7; Table I | Verification with iResNet100 and EdgeFace, all datasets, overlaps 0-4, without and with normalization; Table I reports FNMR at 0.1% FMR. |
| III | Figs. 5 and 6 | Follow [step 3](#3-plot-template-element-ranges-and-t-sne-clustering) for range and t-SNE plots: SOTERIA, iResNet100 and EdgeFace, unnormalized inputs, overlap 3. |
| IV | Fig. 8 | Normalized iResNet100, all datasets, overlaps 0-4; compare `root_l2` and `minimize_cos` inversion scores using histograms. |
| IV | Fig. 9 | Add `minimize_l2` to the normalized iResNet100 solver comparison on all datasets, for overlaps 0 and 3. |
| V and V-A | Figs. 10 and 11; Tables III and IV | Repeat steps 4-7 for all datasets, both main models, and overlaps 0-3. The figures show iResNet100; the tables also report EdgeFace. |
| V-B | Table V | Repeat step 8 for all datasets, both main models, and overlaps 0-3. |
| V-C | Table VI | Multi-PIE, both main models, overlap 3; select at 5%, 10%, and 20% FMR, evaluate ISR/FNMR at 0.01% FMR, and repeat online unlinkability at each selection threshold. |
| V-D | Tables VII and VIII | Repeat step 9 on Multi-PIE, both main models, overlaps 0 and 3, at 5%, 10%, and 20% FMR; report mean search time and mean failed-key count separately. |

Fig. 1 (Section I) illustrates the study's focus, Fig. 2 (Section II-A)
illustrates the transform, and Table II (Section IV) gives its dimensions;
they require no dataset experiment.
