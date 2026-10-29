#!/usr/bin/env bash
# Sweep the research-equivalence regression over all canonical seeds and datasets.
#
# Compares the production split_no_veto pipeline against the frozen research run for
# ENA24 (single split) and LoTE (train+test concatenated), seeds {3,7,11,13,17}.
#
# Usage: regression/run_all.sh   (from the production repo root)
set -uo pipefail

PY="${PY:-/fs/ess/PAS2699/harvest_modules/miniconda3/envs/sam3/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

DATA=/fs/ess/PAS2699/object_detection_datasets/wacv_2027/detection_paper/datasets
SRC=/fs/ess/PAS2699/object_detection_datasets/training_free_paper_expts/datasets
SUITE="$DATA/final_wacv_suite_5seed_lote_train_test_iou025_20260823"
RUNROOT="$SUITE/runs/split_no_veto"
CACHE="$SUITE/density_cache"

SEEDS=(3 7 11 13 17)
FAIL=0

# Large/dense pools: a float32 kNN density matmul is not bit-for-bit reproducible across
# BLAS/hardware, so on LoTE we pin the frozen research density to reproduce tie-breaking
# exactly. ENA24 is small enough that recomputation is already bit-identical.
run_one() {  # dataset run_subdir bg fg coco split_tags density_cache seed
  local name="$1" run="$2" bg="$3" fg="$4" coco="$5" tags="$6" dcache="$7" seed="$8"
  local extra=()
  [ -n "$tags" ] && extra+=(--split-tags "$tags")
  [ -n "$dcache" ] && extra+=(--density-cache "$dcache")
  "$PY" "$HERE/check_research_equivalence.py" \
    --research-run "$run" \
    --bg-features "$bg" --fg-features "$fg" --coco "$coco" "${extra[@]}" \
    --seed "$seed" --rounds 16 --batch-size 50 --calibration-size 100 --iou-threshold 0.25
  local rc=$?
  [ $rc -ne 0 ] && FAIL=1
  return $rc
}

for seed in "${SEEDS[@]}"; do
  run_one ENA24 \
    "$RUNROOT/ENA24/density_balanced_kcenter/seed_${seed}" \
    "$DATA/ENA24/features/all/animal/dinov3/box" \
    "$DATA/ENA24/features/all/animal/bioclip/box" \
    "$SRC/ENA24/annotations/ena24_animal_bbox_instances.json" \
    "" "" "$seed"
done

for seed in "${SEEDS[@]}"; do
  run_one LoTE \
    "$RUNROOT/LoTE/density_balanced_kcenter/seed_${seed}" \
    "$DATA/LoTE/features/train/animal/dinov3/box,$DATA/LoTE/features/test/animal/dinov3/box" \
    "$DATA/LoTE/features/train/animal/bioclip/box,$DATA/LoTE/features/test/animal/bioclip/box" \
    "$SRC/lote_animal/annotations/bbox_only/wildlife_instance_train2017_sharded.json,$SRC/lote_animal/annotations/bbox_only/wildlife_instance_test2017_sharded.json" \
    "train,test" \
    "$CACHE/LoTE_all_box_foreground_dinov3_bg_bioclip_fg_density_balanced_kcenter_full_knn_k30_seed${seed}.npy" \
    "$seed"
done

echo
if [ $FAIL -eq 0 ]; then
  echo "ALL REGRESSIONS PASSED"
else
  echo "SOME REGRESSIONS FAILED"
fi
exit $FAIL
