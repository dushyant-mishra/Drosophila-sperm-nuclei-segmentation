# Saturn v5.7.1 Intensity-Width Hardening Plan

## Goal

Repair the unaccepted intensity-width work without changing frozen Saturn v5.7
behavior or using morphology to reject unusual WT or mutant nuclei.

## Measurement contract

- The one biological width candidate is the background-corrected raw-signal
  FWHM sampled perpendicular to the centerline. It is an apparent comparative
  signal width, not an absolute nucleus diameter.
- The filled-mask contour chord and EDT width remain explicitly named technical
  diagnostics. They do not drive the primary biological width figure.
- The width-to-length ratio uses the FWHM width and centerline length from the
  same representative observed Z plane.
- Integrated profile signal is object-scoped, reported in arbitrary units, and
  remains technical/exploratory QC. With no independent staining reference it
  is not a primary biological endpoint and is never an acceptance gate.
- Filled-mask slab volume and signal-profile footprint proxy are separate
  estimands. Neither silently falls back to the other, and neither is described
  as anatomical seminal-vesicle volume.
- Width, intensity, length, and shape may generate non-destructive annotations;
  they do not veto technically valid mutant morphology.

## Implementation order

1. Add adversarial tests for neighbouring signal, calibration, unavailable
   profiles, explicit area/volume provenance, representative-plane pairing, and
   report routing.
2. Restrict FWHM crossings and integrated signal to the current instance's
   owned profile support. Reject boundary-clipped profiles as unavailable rather
   than using neighbouring signal.
3. Preserve explicit mask-derived legacy area and volume fields. Add separate
   profile-footprint proxy fields with no mixed fallback.
4. Propagate representative FWHM width and its same-plane ratio through tracks,
   specimen summaries, study outputs, and the biological comparison generator.
5. Keep integrated signal and alternate width methods in technical QC only.
6. Reconcile the design ledger and register a new high-risk intensity-width
   claim as implemented but not accepted.
7. Generate current synthetic/adversarial evidence and run compile, focused,
   full-suite, report-contract, and diff checks.
8. Run an independent append-only audit against a clean candidate commit before
   any cohort rerun, release, merge, tag, or push.

## Release boundary

Passing tests is necessary but not sufficient. A full biological rerun remains
blocked until the required independent audit roles accept the final, clean,
hash-bound candidate.
