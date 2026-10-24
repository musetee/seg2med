# Anatomy label conventions

`params_ct.csv` uses the TotalSegmentator `total` label IDs. `params_mr.csv`
uses `total_mr` IDs 1-50, verified against TotalSegmentator 2.13.0 and its
[official class map](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py).
Row 0 is the body-contour prior; 201-203 are the existing merged tissue labels.
Semantic-prior values add the binary body contour to these raw labels, so inside
the body an anatomical label N is normally exported as N+1.

`params_mr_legacy.csv` supports the 56-label dictionary embedded in older masks
such as `1PC000/mr_seg.nii.gz`. For example, legacy label 39 is right gluteus
maximus, while current label 39 is right hip. Legacy label 56 is brain, not a
kidney cyst. Shared tissue names use identical parameters across both CSVs;
brain uses the approximate cortex proxy described below in both tables.

## Input detection and model convention

Both `prepare_case.py` and `create_image_priors` accept these top-level YAML
settings (also overridable per case for `prepare_case.py`):

```yaml
segmentation_label_scheme: auto
prior_label_scheme: legacy
```

- `segmentation_label_scheme: auto` reads the raw ID-to-name dictionary embedded
  in each organ NIfTI. It detects current/legacy conventions from that dictionary,
  not from the maximum label or the acquisition sequence. Each input is resolved
  independently, so old and new masks may appear in one run.
- Metadata-free masks require an explicit `current` or `legacy` input setting.
  An explicit setting cannot override a contradictory embedded dictionary.
  `legacy` is the MR 56-label preset; no metadata-free legacy CT preset is supplied.
  Custom CT/MR dictionaries can be mapped by embedded anatomical names.
- `prior_label_scheme: legacy` selects the old MR output/model convention,
  regardless of whether the input anatomy came from current MR, legacy MR, or CT.
  `current` selects the 50-label MR output convention. CT output retains its
  canonical CT label convention.
- `prior_label_scheme: auto` (the library default) preserves a recognized MR
  input convention; CT-to-MR uses current. Custom or ambiguous MR dictionaries
  require an explicit output choice. For a model trained on legacy priors, choose
  `legacy` explicitly so CT and MR inputs produce a consistent condition.

Input IDs are first resolved to anatomy names; names are then mapped to the
selected output table before BOTH semantic and intensity-prior generation.
Thus legacy/current images of the same anatomy can produce the same model-facing
label values and simulated signals. This applies to same-modality generation too.
The example workflow YAMLs select legacy MR output for older models.

Optional parameter overrides are `parameters.ct`, `parameters.mr` (current), and
`parameters.mr_legacy` (legacy). In the Gradio app, select the input label scheme
in the loading tab and the output MR/checkpoint convention in the simulation tab.
Named-mask ZIPs already identify anatomy by name; selecting legacy MR input uses
legacy IDs for those names. Low-level array APIs cannot inspect NIfTI metadata:
pass an appropriate source-name table / parameter table when using them directly.

The MR table previously contained a different label convention from ID 32 onward.
It is now aligned with `total_mr`; existing relaxation parameters are retained by
tissue name, not by the incorrect row ID. Newly listed scapulae and clavicles use
the existing bone parameter proxy. Whole-brain label 50 uses the cerebral-cortex
proxy (1350/95/86) also used for brain T1 priors; it is not a
validated whole-brain quantitative measurement. Adjust these approximate signal
parameters for your application and acquisition protocol.

Cross-modality conversion uses exact anatomical names and explicit many-to-one
CT-lung-lobe / CT-vertebral-level aggregations into the broader MR labels. There
is no fuzzy matching. MR lungs/vertebrae cannot be split into CT lobes/levels from
a label alone; unsupported labels warn and become unlabeled body. Discs are not
mapped to spinal cord, and thigh muscles are not mapped to gluteal muscles.

This change does not rewrite masks, regenerate old priors, or alter CT label IDs.

After updating, regenerate affected MR and cross-modality priors:

```bash
python prepare_case.py --config configs/prepare_case.yaml --no-overwrite-seg --overwrite-prior --prior-label-scheme legacy
```

This reuses existing masks. Missing masks will still be generated. Original input
images are preserved. Changing label schemes/parameters does not update cached
prior files unless overwrite is enabled; reuse emits a warning. For catalog-level
generation, set `overwrite: true` when regenerating.

Selecting legacy restores the intended legacy anatomical convention, not a
historical parameter bug. Exact checkpoint compatibility also depends on which
parameter values actually generated its training priors. Keep/inspect those
training priors and parameter tables before replacing them; the earlier table's
brain/kidney-cyst confusion is not intentionally reproduced.
