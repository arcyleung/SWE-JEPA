# Closed PR Patch Recovery: python_js_ts_rust_closed_prs

- Rows targeted: `9296`
- Rows updated: `9296`
- Patch fetch success: `9057`
- File-list fetch success: `9189`

## Split quality

- `has_fix_patch`: `8727`
- `has_test_patch`: `1758`
- `is_splittable`: `1453`

## Notes

- Patch splitting follows the legacy `explore_agent_simple.py` test-file heuristic.
- The current implementation stores both `prs_copy`-style JSON patch fields (`file_patches`, `test_file_patches`) and split patch fields (`non_test_patch`, `*_patch_files`, split flags).
- `repo_language` / `extracted_language` now default from each row's `primary_language`, so mixed-language closed tables can be backfilled with one run.
