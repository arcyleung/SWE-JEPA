# Experiment 4.7.3 — go_prs_closed Patch Recovery

- Rows targeted: `18717`
- Rows updated: `18717`
- Patch fetch success: `18282`
- File-list fetch success: `18527`

## Split quality

- `has_fix_patch`: `17949`
- `has_test_patch`: `2881`
- `is_splittable`: `2572`

## Notes

- Patch splitting follows the legacy `explore_agent_simple.py` test-file heuristic.
- The current implementation stores both old `prs_copy`-style fields (`file_patches`, `test_file_patches`) and new Go-style split fields (`non_test_patch`, `*_patch_files`, split flags).
