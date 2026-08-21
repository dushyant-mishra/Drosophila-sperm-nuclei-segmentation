# Saturn v5.7.1 Candidate Validation Receipt

- Validated commit: `fd1e1d0dfc427f2196cd6ec5566309e814dfacc9`
- Branch: `feature/v5.7.1-annotation-tolerant-unet`
- Validation date: `2026-08-21`
- Platform: Windows PowerShell
- Interpreter: `.venv\Scripts\python.exe`

## Commands and results

1. `python -m py_compile sperm_segmentation_saturnv5.7.1.py utils/tune_parameters_Saturnv5_7_1.py utils/saturn_unet25d_bridge.py`
   - Result: passed.
2. `python -m pytest -q`
   - Result: `205 passed in 24.41s`.
3. `python utils/tune_parameters_Saturnv5_7_1.py --self-check`
   - Result: passed all listed checks and ended with `Saturn v5.7.1 tuner self-check passed`.
4. `git diff --check`
   - Result: passed; Git reported only expected LF-to-CRLF working-copy warnings.

The commands were run immediately before candidate commit `fd1e1d0`. The
commit contains the exact staged implementation, tests, documentation, and
curated evidence that were validated; the only remaining local modifications
were confirmed line-ending-only changes in three older v5.7 files and were not
included in the candidate commit.
