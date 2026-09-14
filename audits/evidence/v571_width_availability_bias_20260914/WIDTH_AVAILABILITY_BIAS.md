# Signal-width availability bias (technical QC)

Technical quality control on signal-width availability. This is not a biological morphology comparison and must not be used to tune any parameter toward a genotype outcome.

- Specimens: 35
- Planes per specimen: 3
- Detections: 22381
- Overall unavailable fraction: 0.2135
- Availability vs crowding (Spearman): -0.2596638655462185
- Availability vs density (Spearman): -0.3633053221288516

## Group contrast of the unavailable fraction

```json
{
  "column": "width_unavailable_fraction",
  "KJ": {
    "n": 18,
    "median": 0.21343482222381477,
    "mean": 0.21213312242026142,
    "sd": 0.02415704442732467
  },
  "WT": {
    "n": 17,
    "median": 0.22146739130434778,
    "mean": 0.221112050856557,
    "sd": 0.028671786781941044
  },
  "difference_of_means": 0.00897892843629558,
  "welch_p_value": 0.3254221555133919
}
```

## Group contrast of the mask-width selection bias

```json
{
  "column": "mask_width_selection_bias_um",
  "KJ": {
    "n": 18,
    "median": -0.541790466308594,
    "mean": -0.5395777723524307,
    "sd": 0.04790662962809475
  },
  "WT": {
    "n": 17,
    "median": -0.5456976318359374,
    "mean": -0.5633196662454045,
    "sd": 0.08116690336845453
  },
  "difference_of_means": -0.023741893892973875,
  "welch_p_value": 0.3052487830370014
}
```

A positive selection bias means the dropped objects had wider masks than
the measured ones, so the measured subset under-represents wide objects.

The signal-width contrast is included only so a reviewer can judge whether
an availability difference is large enough to matter relative to it. It is
a technical readout on sampled planes, not a biological result.
