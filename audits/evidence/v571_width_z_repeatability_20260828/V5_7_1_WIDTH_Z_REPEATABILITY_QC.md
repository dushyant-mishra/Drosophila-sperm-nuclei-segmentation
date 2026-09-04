# Saturn v5.7.1 Width Z-Repeatability QC

## Primary conclusion

In this retained replay, largest-area representative-plane widths were usually repeatable against adjacent observed planes; material QC differences occurred in 4.7% of evaluable tracks.

## Compact results

| Specimen | Eligible tracks | With adjacent plane | Median width CV | Median range (um) | Material adjacent difference |
|---|---:|---:|---:|---:|---:|
| KJ-01 | 2813 | 2738 | 0.096 | 0.365 | 4.6% |
| WT-01 | 1682 | 1642 | 0.096 | 0.405 | 4.9% |

## Scope and formulas

- Scope: **repeatability QC**, not biological truth or group inference.
- Population: technical_valid tracks with at least 3 distinct observed planes having finite positive body_width_um.
- Per-track CV: sample standard deviation of observed-plane `body_width_um` divided by its mean.
- Per-track range: maximum minus minimum observed-plane `body_width_um`.
- Adjacent comparison: same track at representative_z - 1 or + 1.
- Material QC flag: absolute difference >= 0.500 um **and** relative difference >= 20% of representative width.
- The flag is descriptive only. It does not invalidate a track, remove unusual morphology, or establish physical diameter accuracy.
