#!/usr/bin/env bash
# End-to-end smoke test of every pipeline stage on CPU, fully offline:
# synthetic images, randomly initialised encoders and a tiny random Qwen3.5-architecture VLM.
# Accuracy numbers from this script are meaningless; it only proves the code runs.
#
#   bash tests/smoke.sh
set -euo pipefail

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export EUROFM_DATA=$TMP/synthetic.npz EUROFM_RESULTS=$TMP/results EUROFM_FEATURES=$TMP/features HF_HUB_OFFLINE=1

python -m eurofm.prepare_data --source synthetic --out "$EUROFM_DATA"
python -m eurofm.extract_features --encoders resnet50_in1k clip_vitb16 --no-pretrained
python -m eurofm.linear_probe --encoders resnet50_in1k clip_vitb16 --budgets 1,5 --seeds 0 1 --export-head
python -m eurofm.zero_shot --encoders clip_vitb16 --no-pretrained
python -m eurofm.finetune --mode scratch --budgets 1,5 --seeds 0 --max-steps 10
python -m eurofm.finetune --mode lora --encoders dinov3_vitb16_web --budgets 2 --seeds 0 --max-steps 2 --batch-size 4 --no-pretrained

python tests/make_tiny_vlm.py "$TMP/tiny-vlm"
python -m eurofm.vlm eval --model "$TMP/tiny-vlm" --test-per-class 2 --batch-size 4
python -m eurofm.vlm eval --model "$TMP/tiny-vlm" --shots 1 --test-per-class 1 --batch-size 2
python -m eurofm.vlm finetune --model "$TMP/tiny-vlm" --budgets 2 --test-per-class 1 --epochs 1

python -m eurofm.report
test -f "$EUROFM_RESULTS/summary.md" && test -f "$EUROFM_RESULTS/figures/learning_curves.png"
echo "SMOKE TEST PASSED ($(($(wc -l < "$EUROFM_RESULTS/runs.csv") - 1)) runs logged)"
