# Spine stack extent and equivalent slice count (0.6–0.7 mm)

Source: `output/dicom_distribution_per_study.csv`
Studies used (valid `z_spacing_mm` and `n_slices` > 0): **2019**

## Definitions

- **Per-study spine length (mm)** is taken as `z_spacing_mm * n_slices` (total extent along the stack if spacing is uniform).
- Your written step `(z_spacing * n_slices) / n_slices` equals **`z_spacing_mm` only**; this report uses the **product** `z_spacing * n_slices` as stack length.
- **Equivalent slice count** at a reference through-plane spacing `z_ref` mm: `spine_length_mm / z_ref`.

## Step 1 — Spine length per study (summary)

| Metric | mm |
|--------|-----|
| Mean `spine_length_mm` | 198.2066 |
| Std (sample) | 28.9247 |
| Min | 49.0000 |
| Max | 335.2000 |

Mean observed **n_slices** (actual): **352.45**

## Step 2 — Range of average equivalent slice count

Using **mean** spine length **198.2066 mm** and reference spacing **0.6 mm** to **0.7 mm**:

- `mean_spine / 0.7` = **283.15** slices (at 0.7 mm spacing)
- `mean_spine / 0.6` = **330.34** slices (at 0.6 mm spacing)

**Reported range (equivalent average slice count): [283.15, 330.34]**

## Per-study output

Full table with `equiv_slices_at_0.7mm` and `equiv_slices_at_0.6mm`: `spine_slice_per_study.csv`.
