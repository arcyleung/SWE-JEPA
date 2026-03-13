# Experiment 4.7.3 — Go-Only Steerer

- Run tag: `phase4_7_3_go_pr_steerer_slurm_v1`
- Rows emitted: `140,981`
- Source mix: `go_prs=123,027`  `go_prs_closed=17,954`
- Acceptance rate: `87.3%`
- Refactor rate: `13.0%`

## CV metrics

- `acceptance`: mean AUROC `0.985`, mean PR-AUC `0.997`
- `refactor`: mean AUROC `0.732`, mean PR-AUC `0.274`

## Artifacts

- Features: `data/phase4_7_3_go_pr_steerer_slurm_v1/conway_patch_features_go_merged_closed.jsonl`
- Labels: `data/phase4_7_3_go_pr_steerer_slurm_v1/phase4_7_3_go_pr_labels.jsonl`
- Model: `data/phase4_7_3_go_pr_steerer_model.json`
- Metrics: `data/phase4_7_3_go_pr_steerer_metrics.json`
