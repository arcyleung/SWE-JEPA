# Experiment 4.7.3 — go_prs_closed Patch Recovery

- Rows targeted: `5`
- Rows updated: `5`
- Patch fetch success: `5`
- File-list fetch success: `5`

## Split quality

- `has_fix_patch`: `5`
- `has_test_patch`: `1`
- `is_splittable`: `1`

## Notes

- Patch splitting follows the legacy `explore_agent_simple.py` test-file heuristic.
- The current implementation stores both old `prs_copy`-style fields (`file_patches`, `test_file_patches`) and new Go-style split fields (`non_test_patch`, `*_patch_files`, split flags).
