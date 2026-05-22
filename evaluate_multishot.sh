#!/usr/bin/env bash
set -euo pipefail

# Change these two paths to your real generated videos root and manifest.
RESULT_ROOT="videos/eval_caption_multishot_t2v_100"
MANIFEST="../FAR-Dev2/assets/data/meta/vbench/Vbench_multishot_manifest.json"
OUTPUT_DIR="videos/eval_caption_multishot_t2v_100"
DEVICE="cuda"

python evaluate_multishot.py \
  --result-root "$RESULT_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --metrics overall_quality shot_structure intra_shot_quality inter_shot_quality \
  --text-alignment-metric overall_consistency \
  --overall-quality-dimensions aesthetic_quality dynamic_degree \
  --intra-shot-quality-dimensions subject_consistency background_consistency \
  --sca-detector transnetv2 \
  --sca-threshold 0.5 \
  --sca-min-gap-sec 0.35 \
  --character-frame-strategy middle \
  --continue-on-error
