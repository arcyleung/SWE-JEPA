 Restart the steered run on the other node. Make sure images are pulled there first:
  docker images | grep featbench | wc -l  # should be 156
  # If not:
  cd /shared_workspace_mfs/arthur/coder/eval/FeatBench && python scripts/pull_images.py --dataset dataset/featbench_v1_0.json --concurrency 48

# V2 HDBSCAN steerer
  cd /shared_workspace_mfs/arthur/coder/eval/FeatBench && \
  PYTHONDONTWRITEBYTECODE=1 python -m docker_agent.runner.parallel_eval \
      --agents qwen35-steered --workers 4 --skip-done \
      --results-file docker_agent/evaluation_results_steered_phase6_2.json

# V1 steerer (62 features hardcoded)
  cd /shared_workspace_mfs/arthur/coder/eval/FeatBench && \
  PYTHONDONTWRITEBYTECODE=1 \
  STEERER_MODEL_PATH=/shared_workspace_mfs/arthur/coder/data/phase5_1_python_pr_corpus_slurm_v1/pr_steerer_model_python_merged_closed_allrows.json \
  python -m docker_agent.runner.parallel_eval \
      --agents qwen35-steered --workers 12 --skip-done \
      --results-file docker_agent/evaluation_results_steered_v1.json