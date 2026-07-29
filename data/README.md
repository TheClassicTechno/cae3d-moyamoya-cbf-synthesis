# Data

The Moyamoya ASL-CBF dataset used in this work is **not included in this
repository**. It consists of clinical perfusion MRI (pre- and
post-acetazolamide CBF maps) and derived imaging from an IRB-approved
Stanford cohort, and cannot be redistributed publicly.

## Requesting access

Researchers affiliated with an institution and covered by an appropriate
data-use agreement / IRB protocol may request access to the dataset by
contacting the corresponding author of the associated paper (see
`CITATION.cff`). Please include:

- your institutional affiliation,
- the IRB/ethics approval covering your use of the data, and
- a brief description of the intended use.

## Expected layout

Code in this repository expects a data directory (path supplied via
`--data-dir` or the `MOYAMOYA_CVR_DATA_DIR` environment variable) containing
subject-level pre/post ASL CBF volumes registered to MNI152 space, plus a
train/val/test split manifest in the format documented in
`src/moyamoya_cvr/data/`. See `configs/` for an example (de-identified)
split manifest once one is published.

## Synthetic / toy data

No synthetic example data is currently bundled. If you need a minimal
example to test the pipeline end-to-end without real patient data, please
open an issue.
