# Protocol Format and Custom Databases

A protocol tells BTPBench which media belong to which subjects and
which samples take part in each evaluation. The raw images or videos remain in
their original database directory; the protocol is a small, versionable bundle
of CSV files that points to them.

Use this guide when adapting a new database. The example assumes this raw image
layout:

```text
/data/acme-face/
  subjects/1/enroll.jpg
  subjects/1/probe.jpg
  subjects/2/enroll.jpg
  subjects/2/probe.jpg
```

Do not copy restricted database media into the protocol directory. Commit only
the CSV definitions when the database's terms allow those definitions to be
redistributed.

## Protocol bundle layout

A complete bundle can support four kinds of evaluation:

```text
protocols/acme-face/
  acme-face-verification.csv
  acme-face-unlinkability.csv
  grandtest/
    dev/
      for_enrolling.csv
      for_probing.csv
    eval/
      for_enrolling.csv
      for_probing.csv
  irreversibility/
    for_distribution.csv
    for_irreversibility.csv
```

| File or directory | Used by | Purpose |
|---|---|---|
| Verification sample CSV | Verification, verification-based key selection, and the default diversity flow | Defines one pool whose every unique sample pair is compared. Its filename is configured by `verification_samples`. |
| Unlinkability sample CSV | Unlinkability and unlinkability-based diversity | Defines the per-subject sample pool protected under different keys. Its filename is configured by `unlink_samples`. |
| `<protocol>/<split>/for_enrolling.csv` | Identification and identification-based key validation | Defines gallery/reference samples. |
| `<protocol>/<split>/for_probing.csv` | Identification and identification-based key validation | Defines probe/query samples. |
| `irreversibility/for_distribution.csv` | Irreversibility and distribution plots | Defines reference samples used to estimate the unprotected-template distribution. |
| `irreversibility/for_irreversibility.csv` | Irreversibility | Defines samples whose protected templates are inverted and evaluated. |

The two sample-list filenames are arbitrary because the system configuration
points to them explicitly. The four `for_*.csv` filenames are fixed.

Identification protocols are discovered recursively. The directory immediately
above a split is the protocol name, and the split directory supplies the split
name. In the example, BTPBench discovers protocol `grandtest` with `dev` and
`eval` splits. A `for_enrolling.csv` without a sibling `for_probing.csv` is
reported as incomplete and is not loaded.

Only create the identification and irreversibility files for experiments you
intend to run. Missing irreversibility files produce a warning and make those
workflows unavailable. Both irreversibility files are required as a pair, even
for the distribution plot, which reads samples only from
`for_distribution.csv`. Both configured sample-list CSVs must always exist,
however, because a `Dataset` reads them both during initialization. If you do
not need a distinct unlinkability pool, `unlink_samples` may point to the
verification CSV.

## Common CSV schema

Every protocol CSV is a comma-separated file with one header row. Column names
are case-sensitive.

| Column | Required | Meaning |
|---|---|---|
| `path` | Yes | Media path relative to the configured `dataset_dir`. The configured `extension` is appended verbatim. |
| `subject_id` | Yes | Identity label. Rows with equal values represent the same person. |
| `template_id` | Yes | Stable identifier for one logical sample. Together with `subject_id`, it identifies cached features and score rows. |
| `flip` | No | Boolean requesting a 180-degree rotation after decoding. Use `true` or `false` in every row when this column is present. |
| `frame_idx` | No | Video position expressed as a percentage from `0` to `100`; `0` selects the first frame and `100` the last frame. |

All other columns become sample metadata. Examples include `session`, `device`,
`camera`, `illumination`, and `accessory`. Metadata is copied into score files
as `bio_ref_<name>` and `probe_<name>` columns.

Use these rules throughout a bundle:

- Make each `(subject_id, template_id)` pair unique for a logical sample. In
  particular, repeated rows for different frames of one video need different
  template IDs, such as `1-session2-f0` and `1-session2-f30`.
- Keep subject IDs consistent across every file. Standard user-specific BTP
  protection without a `key_dictionary_file` derives its key by converting
  `subject_id` to an integer. For that mode, use integer IDs, configure a key
  dictionary, or choose system-specific protection. Workflows that supply keys
  explicitly are not subject to this conversion.
- Avoid IDs that rely on leading zeroes. CSV type inference can turn `001` into
  `1`. A text prefix such as `subject-001` preserves the formatting, subject to
  the BTP key constraint above.
- Do not leave `flip` or `frame_idx` cells empty. Empty CSV cells are inferred as
  missing values and can lead to unintended loader behavior. Either omit the
  entire optional column or supply a valid value in every row.
- Do not add an exported DataFrame index column such as `Unnamed: 0`; it would
  be treated as metadata.
- Avoid a metadata column named `key`; key-selection workflows use that name
  for the evaluated BTP key.

### Metadata header consistency

The verification and unlinkability sample lists each define the metadata
headers for their own scores. Identification derives its score metadata from
each enrollment/probe pair, and irreversibility derives it from its
distribution/irreversibility pair. The two files in each pair must define the
same metadata columns, although their column order may differ.

Different workflows may use different metadata schemas. Using one consistent
schema throughout a database bundle is still the simplest convention when the
same attributes apply everywhere.

## Paths, extensions, images, and videos

For every row, the loader constructs the media location as:

```text
<dataset_dir>/<path><extension>
```

With `dataset_dir: /data/acme-face`, `path: subjects/1/enroll`, and
`extension: .jpg`, the resulting file is
`/data/acme-face/subjects/1/enroll.jpg`. If CSV paths already contain `.jpg`,
set `extension: ""` to avoid creating `enroll.jpg.jpg`.

Protocol paths should use `/` separators, should not start with `/`, and should
not contain shell wildcards. Paths in the system and experiment YAML files are
interpreted relative to the process's current working directory when they are
not absolute.

Set `video: false` for image files and `video: true` for videos. A video row
without `frame_idx`, or with `frame_idx: 0`, reads the first frame. Positive
values seek to that percentage of the video's duration; they are not absolute
frame numbers. When the same video supplies several samples, repeat its path
with a distinct `frame_idx` and a distinct template ID.

One `video` setting selects one loader for the entire database entry. A database
that mixes images and videos needs separate system-configuration entries, or a
custom `Dataset` loader when using the Python API.

With experiment option `compliant: true`, missing paths are warned about and
skipped. With `compliant: false`, the first missing path raises an error. Media
decoding is lazy, so constructing a `Dataset` alone does not prove that every
image or video can be decoded.

## Define each evaluation set

### Verification

Verification compares every unique pair of rows. Same-subject pairs are genuine
comparisons and different-subject pairs are impostor comparisons. A useful set
therefore needs at least two samples for some subjects and at least two
subjects. With `N` rows, the pipeline performs `N * (N - 1) / 2` comparisons,
so keep the quadratic cost in mind for large databases.

For the example database:

```csv
path,subject_id,template_id,session
subjects/1/enroll,1,1-enroll,1
subjects/1/probe,1,1-probe,2
subjects/2/enroll,2,2-enroll,1
subjects/2/probe,2,2-probe,2
```

### Unlinkability

The unlinkability pipeline groups rows by subject while preserving CSV order,
then keeps the first `--samples-per-subject` rows for each subject. Every
subject must have at least that many readable samples. The default is 60, so a
real unlinkability protocol commonly needs many frames or sessions per subject.
See the [unlinkability guide](unlinkability.md#protocol-requirements) for the
key and sample-count requirements.

You may initially copy the verification rows into a separate unlinkability CSV,
then expand it with additional sessions or video-frame positions. Do not reuse
one template ID for several frame positions.

### Identification

Each enrollment row becomes a gallery reference and each probe row is compared
with every reference. `template_id` does not aggregate multiple rows into one
template; multiple enrollment rows for a subject remain separate references.

Minimal `grandtest/eval/for_enrolling.csv`:

```csv
path,subject_id,template_id,session
subjects/1/enroll,1,1-enroll,1
subjects/2/enroll,2,2-enroll,1
```

Minimal `grandtest/eval/for_probing.csv`:

```csv
path,subject_id,template_id,session
subjects/1/probe,1,1-probe,2
subjects/2/probe,2,2-probe,2
```

Create additional protocols when you want to evaluate different gallery/probe
conditions, and additional splits when the evaluation design defines separate
development and evaluation partitions. Preserve an official database protocol
when one exists; otherwise document how subjects and samples were selected so
the experiment can be reproduced.

### Irreversibility

`for_distribution.csv` supplies templates used to estimate the input-feature
distribution for inversion. `for_irreversibility.csv` supplies the protected
templates that the pipeline attempts to invert. Both use the common schema.
Unless the official evaluation says otherwise, keep the two sample sets
disjoint and consider using disjoint subjects to avoid leakage. Document any
overlap; the loader does not enforce subject or sample separation.

When evaluating a user-specific `key_dictionary_file`, every attack-target
subject must have an entry in that dictionary. The background-distribution set
does not need to contain the same subjects. When keys are selected from an
enrollment capture for each user, use different captures of those users as the
attack targets so that key selection and irreversibility evaluation are not
performed on the same template.

## Register the database

Add an entry to the `databases` mapping in a system configuration file:

```yaml
databases:
  acme-face:
    proto_dir: ./protocols/acme-face
    verification_samples: ./protocols/acme-face/acme-face-verification.csv
    unlink_samples: ./protocols/acme-face/acme-face-unlinkability.csv
    dataset_dir: /data/acme-face
    extension: .jpg
    video: false
```

Then select it in the experiment configuration:

```yaml
database: acme-face
protocols: grandtest
splits: eval
```

`protocols` and `splits` are used by identification workflows. Verification and
unlinkability use their configured sample lists, while irreversibility uses its
two fixed files. See the [system configuration](system_config.md) and
[experiment configuration](experiment_config.md) references for the remaining
fields.

## Creation workflow

For a new database:

1. Read the database's license and official evaluation specification. Decide
   which protocol definitions may be redistributed.
2. Inventory the available media and assign stable subject and template IDs.
3. Choose one metadata schema and use it consistently across the bundle.
4. Build the verification sample list and, when needed, a sufficiently large
   unlinkability sample list.
5. Create each identification protocol and split as paired enrollment and probe
   files.
6. Add the two irreversibility files when those experiments are required.
7. Register all paths, the common suffix, and the media type in the system YAML.
8. Validate paths and decoding with `compliant: false` before a long run, then
   inspect the discovered protocol and split names.

If the database is arranged as `<subject>/<image>.jpg`, a small generator can
walk its subject directories, use the first sorted image for enrollment, and
use the remaining images for probing. Treat that only as a starting point:
published database partitions, subject-disjoint splits, session constraints,
and demographic balance must take precedence over a filename-based split.

## Validation checklist

Before running an evaluation, check that:

- every CSV has `path`, `subject_id`, and `template_id` exactly once;
- every constructed media path exists and has the configured media type;
- all images or selected video frames decode to a three-channel RGB array;
- `(subject_id, template_id)` pairs are unique where rows represent different
  logical samples;
- verification contains both genuine and impostor pairs;
- every unlinkability subject has the requested number of readable samples;
- every enrollment file has a sibling probe file;
- protocol and split names match the experiment YAML exactly;
- both files in every enrollment/probe or irreversibility pair use the same
  metadata columns; and
- no subject or sample crosses partitions contrary to the evaluation design.

This short smoke test exercises the same loaders used by the pipelines. Adjust
the paths and loader for the database being added:

```python
from pathlib import Path

from btpbench.dataloader import Dataset, cv_loader

dataset = Dataset(
    protocols_dir=Path("./protocols/acme-face"),
    dataset_dir=Path("/data/acme-face"),
    verification_samples_file=Path(
        "./protocols/acme-face/acme-face-verification.csv"
    ),
    unlinkability_samples_file=Path(
        "./protocols/acme-face/acme-face-unlinkability.csv"
    ),
    extension=".jpg",
    compliant=False,
    load_f=cv_loader,
)

sample_sets = {
    "verification": dataset.samples(),
    "unlinkability": dataset.samples(verification=False),
}
for protocol in dataset.protocols():
    for split in dataset.protocol_splits(protocol):
        loader = dataset.load_protocol(protocol, split)
        sample_sets[f"{protocol}/{split}/enrollment"] = list(loader.references())
        sample_sets[f"{protocol}/{split}/probe"] = list(loader.probes())

try:
    irreversibility = dataset.load_irreversibility()
except RuntimeError:
    pass
else:
    sample_sets["irreversibility/distribution"] = list(
        irreversibility.references()
    )
    sample_sets["irreversibility/inversion"] = list(irreversibility.probes())

for name, samples in sample_sets.items():
    for sample in samples:
        image = sample.data()
        assert image.ndim == 3 and image.shape[2] == 3, sample.path
    print(f"{name}: {len(samples)} readable sample(s)")
```

For videos, import `cv_loader_video`, pass it as `load_f`, use the database's
real suffix, and set the system configuration's `video` field to `true`.
