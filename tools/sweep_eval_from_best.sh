#!/usr/bin/env bash
set -euo pipefail

# Sweep experiment folders and run direct evaluation using each folder's
# best checkpoint.
#
# Usage:
#   bash tools/sweep_eval_from_best.sh <target_dir> <glob_pattern>
#
# Example:
#   bash tools/sweep_eval_from_best.sh \
#     /netscratch/naeem/mmseg_output/eccv_results \
#     'Deeplabv3Plus_r50_phenobench_combined_b4_50_50_real_*'
#
# Required inputs:
#   1) target_dir   : parent directory that contains experiment folders
#   2) glob_pattern : shell glob used to match folder names under target_dir
#
# Defaults can be overridden via environment variables:
#   CONFIG_PATH            (default from your launch.json)
#   EVAL_SCRIPT            (default: tools/test.py)
#   PYTHON_BIN             (default: python)
#   OUTPUT_ROOT            (default: <target_dir>/sweep_eval_runs)
#   DRY_RUN=1              (print commands only, do not execute)
#   EXTRA_ARGS             (extra args appended to each command)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <target_dir> <glob_pattern>"
  exit 1
fi

TARGET_DIR="$1"
GLOB_PATTERN="$2"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: target_dir does not exist: $TARGET_DIR"
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_ohem_loss.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-tools/test.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TARGET_DIR/sweep_eval_runs}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: eval script not found: $EVAL_SCRIPT"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# Collect matching directories using shell glob under target_dir.
shopt -s nullglob
matches=("$TARGET_DIR"/$GLOB_PATTERN)
shopt -u nullglob

if [[ ${#matches[@]} -eq 0 ]]; then
  echo "No folders matched: $TARGET_DIR/$GLOB_PATTERN"
  exit 0
fi

find_best_ckpt() {
  local exp_dir="$1"

  # Priority 1: explicit best_mIoU_iter_*.pth files.
  local best_miou_iter_list=()
  while IFS= read -r p; do
    best_miou_iter_list+=("$p")
  done < <(find "$exp_dir" -maxdepth 1 -type f -name 'best_mIoU_iter_*.pth' | sort -V)

  if [[ ${#best_miou_iter_list[@]} -gt 0 ]]; then
    # Usually there is a single file. If multiple exist, pick highest iteration.
    printf '%s\n' "${best_miou_iter_list[-1]}"
    return 0
  fi

  # Priority 2: any best_*.pth fallback.
  local best_list=()
  while IFS= read -r p; do
    best_list+=("$p")
  done < <(find "$exp_dir" -maxdepth 1 -type f -name 'best_*.pth' | sort -V)

  if [[ ${#best_list[@]} -gt 0 ]]; then
    printf '%s\n' "${best_list[-1]}"
    return 0
  fi

  # Priority 3: latest.pth
  if [[ -f "$exp_dir/latest.pth" ]]; then
    printf '%s\n' "$exp_dir/latest.pth"
    return 0
  fi

  # Priority 4: highest iter_*.pth
  local iter_list=()
  while IFS= read -r p; do
    iter_list+=("$p")
  done < <(find "$exp_dir" -maxdepth 1 -type f -name 'iter_*.pth' | sort -V)

  if [[ ${#iter_list[@]} -gt 0 ]]; then
    printf '%s\n' "${iter_list[-1]}"
    return 0
  fi

  return 1
}

echo "Target dir   : $TARGET_DIR"
echo "Glob pattern : $GLOB_PATTERN"
echo "Config       : $CONFIG_PATH"
echo "Eval script  : $EVAL_SCRIPT"
echo "Output root  : $OUTPUT_ROOT"
echo "Dry run      : $DRY_RUN"
echo

run_count=0
skip_count=0

for path in "${matches[@]}"; do
  [[ -d "$path" ]] || continue

  exp_name="$(basename "$path")"
  ckpt=""
  if ! ckpt="$(find_best_ckpt "$path")"; then
    echo "[SKIP] $exp_name -> no checkpoint found"
    ((skip_count+=1))
    continue
  fi

  eval_work_dir="$OUTPUT_ROOT/${exp_name}_eval"
  mkdir -p "$eval_work_dir"

  cmd=(
    "$PYTHON_BIN" "$EVAL_SCRIPT"
    "$CONFIG_PATH"
    "$ckpt"
    --work-dir "$eval_work_dir"
  )

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_arr=($EXTRA_ARGS)
    cmd+=("${extra_arr[@]}")
  fi

  echo "[RUN ] $exp_name"
  echo "       ckpt: $ckpt"
  echo "       work: $eval_work_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '       cmd : '
    printf '%q ' "${cmd[@]}"
    echo
    echo
  else
    "${cmd[@]}"
    echo
  fi

  ((run_count+=1))
done

echo "Sweep finished. launched=$run_count skipped=$skip_count"
