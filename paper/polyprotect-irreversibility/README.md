# PolyProtect irreversibility: experiment guide

**A Deeper Dive into the Irreversibility of PolyProtect: Making Protected Face
Templates Harder to Invert**

Vedrana Krivokuća Hahn, Jérémy Maceiras, Sébastien Marcel. Forthcoming.

This example uses **SOTERIA, iResNet100, and normalized PolyProtect with overlap
3**. It measures verification and inversion, selects user-specific keys, then
repeats the measurements. The last section explains how to run the other paper
experiments.

## 1. Prepare

1. Follow the [installation instructions](../../README.md#installation).
2. Obtain SOTERIA and set its `dataset_dir` in
   [system_config.yaml](configs/system_config.yaml).
3. Run the commands below from the repository root.

The configurations use MediaPipe, cosine minimization with L-BFGS-B, five
solver guesses per inversion attempt, ten attack trials, and seed 42. Results are saved under
`paper/polyprotect-irreversibility/output/soteria-iresnet100/`.
The guide uses the original protocol CSVs in the repository's
[protocols/](../../protocols/) directory.

## 2. Measure verification with ordinary keys

```bash
uv run btpbench verification pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml
```

This creates unprotected and protected verification scores. Compute FNMR at
0.1% and 0.01% FMR:

```bash
uv run btpbench verification metrics \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -l "Unprotected" \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "PolyProtect: ordinary keys" \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-metrics.csv
```

Read `fnmr_10.0` for 0.1% FMR and `fnmr_1.0` for 0.01% FMR.
Multiply these rates by 100 to report percentages. All CLI FMR values are
fractions: `0.001` means 0.1%.

To plot the DET curves:

```bash
uv run btpbench verification plots \
  -f paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -l "Unprotected" \
  -f paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "PolyProtect: ordinary keys" \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-det.png
```

## 3. Measure inversion with ordinary keys

```bash
uv run btpbench irreversibility pipeline \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml
```

The SOTERIA protocol uses 350 reference samples from 35 identities to estimate
the inversion distribution and attacks 350 samples from the other 35 identities.
With all target samples available and ten trials, the score CSV contains
3,500 attempts.

## 4. Select keys and repeat the measurements

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

Selected-key scores are stored in a separate `fmr2000-...` subdirectory.

## 5. Compare ordinary and selected keys

Compute inversion success rates (ISR) using the same unprotected verification
scores to determine the match thresholds:

```bash
uv run btpbench irreversibility metrics \
  -i paper/polyprotect-irreversibility/output/soteria-iresnet100/irreversibility-subject-id-systems1-trials10-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Ordinary keys" \
  -i paper/polyprotect-irreversibility/output/soteria-iresnet100/fmr2000-legacy_None-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100/irreversibility-dictionary-systems1-trials10-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Selected keys" \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-iresnet100.csv \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/isr-comparison.csv
```

Read `success_rate_far0.0010` and `success_rate_far0.0001`, including unsolved
attempts in the denominator. Multiply by 100 for the ISR percentages in Table III.

Compare verification accuracy for Table IV:

```bash
uv run btpbench verification metrics \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Ordinary keys" \
  -s paper/polyprotect-irreversibility/output/soteria-iresnet100/fmr2000-legacy_None-minimize_cos-3-5-soteria-normalized_polyprotect_usr_3_5_50-iresnet100/verification-normalized_polyprotect_usr_3_5_50-iresnet100.csv \
  -l "Selected keys" \
  -f 0.001 -f 0.0001 \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/fnmr-comparison.csv
```

For Fig. 11, use the same two verification files with `verification plots`,
as in step 2. For Fig. 10, pass the two inversion files to
[`plots histogram`](../../docs/plots.md#histogram-plot), together with the
unprotected verification file. Lower ISR indicates better inversion resistance;
lower FNMR indicates better recognition accuracy.

## 6. Compare online unlinkability

Use the ordinary-key configuration: this command selects a new key for each
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
candidate keys against each sample at 20% FMR. Both use the same ten samples
per subject from `unlink_samples`. With 70 usable SOTERIA identities, each run
writes 3,150 mated and 241,500 non-mated scores.

Plot the selected-key scores and read `Dsys` from the figure title or log:

```bash
uv run btpbench unlinkability plots \
  -m paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected/online-unlinkability-soteria-normalized_polyprotect_usr_3_5_50-iresnet100-selected-ksfmr0d2-poolrandomspace-keyseed42-minimize_cos-3-5-n10-nm10-seed42-mated.csv \
  -n paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected/online-unlinkability-soteria-normalized_polyprotect_usr_3_5_50-iresnet100-selected-ksfmr0d2-poolrandomspace-keyseed42-minimize_cos-3-5-n10-nm10-seed42-non-mated.csv \
  --metric-bins 10 --omega 1 \
  -t "SOTERIA: selected user keys" \
  -o paper/polyprotect-irreversibility/output/soteria-iresnet100/online-selected/unlinkability.png
```

Repeat with the two CSVs in `online-random/` to measure the ordinary-key
condition. These use `random` instead of `selected-ksfmr0d2` in their names.
Lower `Dsys` means better unlinkability. This example uses the current metric
defaults; retain the original bin count and score range when matching paper
values. An optional `-k` restricts selection to a supplied candidate pool; see
the [online unlinkability reference](../../docs/unlinkability.md#online-user-specific-key-selection).

## 7. Measure key-selection cost

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
key, including the accepted one. Timing excludes feature extraction. Record
hardware and `num_processes` alongside the CSVs. For Table VII, change to
Multi-PIE and evaluate both models at overlaps 0 and 3.

## 8. Run the other paper experiments

Edit [experiment.yaml](configs/experiment.yaml) and, when evaluating selected
keys, make the same changes in [selected_keys.yaml](configs/selected_keys.yaml).
Update the paths in the commands to match the new outputs.

| Change | What to edit |
|---|---|
| Dataset | Set `database` to `multipie` or `icarb`; fill in that dataset's `dataset_dir` in the system configuration. |
| Face model | Set `bio_alg` to `edgeface`. Fig. 3 also uses `iresnet50`, `edgefacexs`, and `facenet`. |
| Overlap | Change `overlap` to 0–4 for ordinary keys, or 0–3 for selected keys. |
| Normalization | Set `normalize_input: false` for unnormalized experiments; filenames then start with `unnormalized_polyprotect`. |
| Solver | Set `method` to `root_l2`, `minimize_l2`, or `minimize_cos`. |
| Selection threshold | Change the selection command to `-f 0.05`, `-f 0.1`, or `-f 0.2`. These produce `fmr500`, `fmr1000`, or `fmr2000` key files. |

Use a separate `output_dir` for each dataset, model, and normalization setting.
After selecting keys for a different setting, update `key_dictionary_file` in
`selected_keys.yaml` to the new JSON file. Keep normalized inputs and
`method: minimize_cos` for key selection. Use a new output directory when
changing settings. Matching completed runs are skipped; use `--override`
when deliberately repeating them.

| Paper result | Experiments to run |
|---|---|
| Fig. 3 | Verification on all three datasets and all five models; remove `btps` from a configuration copy for unprotected-only evaluation. |
| Figs. 4 and 7; Table I | Verification with iResNet100 and EdgeFace, all datasets, overlaps 0–4, without and with normalization. |
| Figs. 5 and 6 | SOTERIA, iResNet100 and EdgeFace, unnormalized inputs, overlap 3; use the distribution command below. |
| Fig. 8 | Normalized iResNet100, all datasets, overlaps 0–4; compare `root_l2` and `minimize_cos` inversion scores using histograms. |
| Fig. 9 | Add `minimize_l2` to the solver comparison for overlaps 0 and 3. |
| Figs. 10 and 11; Tables III and IV | Repeat steps 2–5 for all datasets, both main models, and overlaps 0–3. |
| Table V | Repeat step 6 for all datasets, both main models, and overlaps 0–3. |
| Table VI | Multi-PIE, both main models, overlap 3; select at 5%, 10%, and 20% FMR, evaluate ISR/FNMR at 0.01% FMR, and repeat online unlinkability at each selection threshold. |
| Table VII | Repeat step 7 on Multi-PIE, both main models, overlaps 0 and 3. |

For Figs. 5 and 6, set `normalize_input: false` and a separate `output_dir`, then run:

```bash
uv run btpbench plots distribution \
  -s paper/polyprotect-irreversibility/configs/system_config.yaml \
  -e paper/polyprotect-irreversibility/configs/experiment.yaml
```

The command writes `dist_*.png` element-range plots and `tsne_*.png` identity
clustering plots. Figures 1–2 and Table II describe the method and its dimensions;
they require no dataset experiment.

## Reproduction notes

Keep the configurations, protocol CSVs, selected-key JSONs, repository revision,
and execution logs with your results.
