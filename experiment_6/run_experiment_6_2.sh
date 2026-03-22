#!/usr/bin/env bash
# Pipeline runner for Experiment 6.2: Multi-Axis Embedding Steerer with Contrastive Projection
#
# Stages:
#   1. Contrastive projection head (InfoNCE, 2048 → 256)
#   2. Multi-axis probing heads (12 × LogReg on projected embeddings)
#   3. HDBSCAN clustering + post-hoc Conway labeling
#
# Usage:
#   bash run_experiment_6_2.sh                # full run (100 epochs, GPU recommended)
#   bash run_experiment_6_2.sh --smoke        # quick smoke test (5 epochs)
#   bash run_experiment_6_2.sh --cluster-only # skip stages 1+2, run only HDBSCAN (stage 3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EMBEDDINGS="${PROJECT_ROOT}/data/phase6_1/patch_embeddings_100k.npz"
CONWAY="${PROJECT_ROOT}/data/conway_patch_features_v2_100k.jsonl"
OUT_DIR="${PROJECT_ROOT}/data/phase6_2"

EPOCHS=100
BATCH_SIZE=512
CLUSTER_ONLY=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --smoke)
            echo "=== SMOKE TEST MODE ==="
            EPOCHS=5
            BATCH_SIZE=256
            ;;
        --cluster-only)
            CLUSTER_ONLY=true
            ;;
    esac
done

echo "============================================================"
echo "Experiment 6.2: Multi-Axis Embedding Steerer"
echo "  Embeddings: ${EMBEDDINGS}"
echo "  Conway:     ${CONWAY}"
echo "  Output:     ${OUT_DIR}"
echo "  Epochs:     ${EPOCHS}"
echo "============================================================"

mkdir -p "${OUT_DIR}"

if [[ "${CLUSTER_ONLY}" == "false" ]]; then

# ── Stage 1: Contrastive Projection Head ──────────────────────────────────
echo ""
echo ">>> Stage 1: Contrastive projection head (InfoNCE)"
python "${SCRIPT_DIR}/train_contrastive_head.py" \
    --embeddings "${EMBEDDINGS}" \
    --out-dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr 3e-4 \
    --d-hidden 512 \
    --d-out 256 \
    --tau 0.07 \
    --seed 42

# ── Stage 2: Multi-Axis Probing Heads ────────────────────────────────────
echo ""
echo ">>> Stage 2: Multi-axis probing heads (12 × LogReg)"
python "${SCRIPT_DIR}/train_multiaxis_heads.py" \
    --projected "${OUT_DIR}/projected_embeddings.npz" \
    --raw-embeddings "${EMBEDDINGS}" \
    --conway-features "${CONWAY}" \
    --out-dir "${OUT_DIR}" \
    --cv-folds 5 \
    --seed 42

else
    echo ""
    echo ">>> Skipping stages 1+2 (--cluster-only), using existing ${OUT_DIR}/projected_embeddings.npz"
fi

# ── Stage 3: HDBSCAN Clustering + Labeling ──────────────────────────────
echo ""
echo ">>> Stage 3: HDBSCAN clustering + Conway labeling"
python "${SCRIPT_DIR}/cluster_and_label.py" \
    --projected "${OUT_DIR}/projected_embeddings.npz" \
    --conway-features "${CONWAY}" \
    --out-dir "${OUT_DIR}" \
    --min-cluster-size 200 \
    --min-samples 20

echo ""
echo "============================================================"
echo "Experiment 6.2 complete. Outputs in ${OUT_DIR}:"
echo "  contrastive_head.pt          — projection weights"
echo "  projected_embeddings.npz     — h (N, 256)"
echo "  contrastive_metrics.json     — loss curves, retrieval P@10"
echo "  multiaxis_heads.json         — 12 × probing head weights"
echo "  multiaxis_metrics.json       — per-axis AUROC/PR-AUC"
echo "  clusters.json                — cluster profiles"
echo "  cluster_assignments.npz      — per-patch cluster labels"
echo "  cluster_summary.json         — clustering summary stats"
echo "============================================================"
