# Experiment 4.7.3 — Closed PR Ingestion into prs_copy_closed

This report measures how much of the `prs_copy` schema can be recovered from the
closed/unmerged JSONL corpus in `/shared_workspace_mfs/akki/scratch_mfs/arthur-task/enriched-all-unmerged`.

- Source PR rows: `18722`
- Repo files: `415`
- Skipped merged/non-closed tail: `2714`
- GraphQL enrichment success: `18721` / `18722`
- Overflowed GitHub PR ids coerced to NULL: `4662`
- Commit payload partial rows: `91`
- Comment payload partial rows: `3`
- Review-thread partial rows: `2`

## Recoverability classes

- `direct_or_derived`: present locally or derivable from the JSONL
- `fetchable`: missing locally but recoverable from GitHub REST/patch endpoints
- `unavailable`: not present in this corpus and not targeted for recovery in 4.7.3

## Column coverage

| Column | Class | Non-null rows | Coverage |
|---|---|---:|---:|
| `id` | `direct_or_derived` | 14060 | 75.1% |
| `crawl_time` | `direct_or_derived` | 18722 | 100.0% |
| `instance_id` | `direct_or_derived` | 18722 | 100.0% |
| `repo` | `direct_or_derived` | 18722 | 100.0% |
| `pull_number` | `direct_or_derived` | 18722 | 100.0% |
| `issue_numbers` | `direct_or_derived` | 7463 | 39.9% |
| `base_commit` | `direct_or_derived` | 18722 | 100.0% |
| `patch` | `fetchable` | 0 | 0.0% |
| `file_patches` | `fetchable` | 0 | 0.0% |
| `test_patch` | `fetchable` | 0 | 0.0% |
| `test_file_patches` | `fetchable` | 0 | 0.0% |
| `problem_statement` | `unavailable` | 0 | 0.0% |
| `hints_text` | `unavailable` | 0 | 0.0% |
| `pass_to_pass` | `unavailable` | 0 | 0.0% |
| `fail_to_pass` | `unavailable` | 0 | 0.0% |
| `repo_id` | `direct_or_derived` | 18722 | 100.0% |
| `stars` | `direct_or_derived` | 18722 | 100.0% |
| `forks` | `direct_or_derived` | 18722 | 100.0% |
| `primary_language` | `direct_or_derived` | 18722 | 100.0% |
| `pr_title` | `direct_or_derived` | 18722 | 100.0% |
| `pr_body` | `direct_or_derived` | 16894 | 90.2% |
| `pr_url` | `direct_or_derived` | 18722 | 100.0% |
| `pr_state` | `direct_or_derived` | 18722 | 100.0% |
| `pr_merged` | `direct_or_derived` | 18722 | 100.0% |
| `pr_is_draft` | `direct_or_derived` | 18722 | 100.0% |
| `pr_author` | `direct_or_derived` | 18722 | 100.0% |
| `pr_author_name` | `direct_or_derived` | 18587 | 99.3% |
| `pr_labels` | `direct_or_derived` | 18722 | 100.0% |
| `base_branch` | `direct_or_derived` | 18722 | 100.0% |
| `head_branch` | `direct_or_derived` | 18722 | 100.0% |
| `base_sha` | `direct_or_derived` | 18722 | 100.0% |
| `head_sha` | `direct_or_derived` | 18722 | 100.0% |
| `created_at` | `direct_or_derived` | 18722 | 100.0% |
| `updated_at` | `direct_or_derived` | 18722 | 100.0% |
| `total_commits` | `direct_or_derived` | 18722 | 100.0% |
| `commits` | `direct_or_derived` | 18511 | 98.9% |
| `total_comments` | `direct_or_derived` | 18722 | 100.0% |
| `comments` | `direct_or_derived` | 15139 | 80.9% |
| `total_review_threads` | `direct_or_derived` | 18722 | 100.0% |
| `review_threads` | `direct_or_derived` | 1447 | 7.7% |
| `requested_reviewers` | `direct_or_derived` | 4291 | 22.9% |
| `submitted_reviews` | `direct_or_derived` | 2287 | 12.2% |
| `additions` | `direct_or_derived` | 18721 | 100.0% |
| `deletions` | `direct_or_derived` | 18721 | 100.0% |
| `changed_files` | `direct_or_derived` | 18721 | 100.0% |
| `pr_category` | `direct_or_derived` | 18722 | 100.0% |
| `pr_category_confidence` | `direct_or_derived` | 18722 | 100.0% |
| `pr_category_reasoning` | `direct_or_derived` | 18722 | 100.0% |
| `linked_issues` | `direct_or_derived` | 612 | 3.3% |
| `closing_issue_id` | `direct_or_derived` | 612 | 3.3% |
| `merged_at` | `direct_or_derived` | 0 | 0.0% |
| `merged_by` | `unavailable` | 0 | 0.0% |

## Notes

- `patch` and `file_patches` are the main fetch-required artifacts for downstream patch-based experiments.
- `comments` / `commits` are mostly present locally; the crawler only needs to repair a small partial tail.
- `review_threads` are already present in almost all rows that have them; the remaining partial rows are small enough to defer.
