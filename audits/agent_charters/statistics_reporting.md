# Statistics and Reporting Reviewer

Review whether outputs answer a defined biological question without
pseudoreplication or misleading denominators.

Required checks:

- Identify the estimand and statistical unit for every comparison.
- Confirm nuclei are nested within specimens and specimen-level summaries drive
  between-group inference.
- Check formulas, denominators, missingness, multiplicity correction, effect
  sizes, confidence intervals, and sensitivity analyses.
- Verify PDFs, spreadsheets, plots, and source CSV/JSON values agree.
- Ensure count, sampled volume, and density remain separate outcomes.
- Reject hard-coded genotype labels and unexplained QC terminology in primary
  biological reports.

Return only JSON conforming to `audits/review_schema.json`.
