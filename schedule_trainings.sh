#!/bin/bash
set -euo pipefail

# Usage:
#   bash schedule_trainings.sh                   # submit jobs
#   bash schedule_trainings.sh --dry-run         # show generated paths/commands
#   bash schedule_trainings.sh --print-trainings # print "config;work_dir" lines

# ---------------------------------------------------------------------------
# User-edit section
# ---------------------------------------------------------------------------
CONFIG_DIR="configs/wacv"
OUTPUT_ROOT="/netscratch/naeem/mmseg_output/wacv_results"

# Provide only config filenames (or relative paths if you want to override).
declare -a CONFIG_FILES=(
    "deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench.py"
    "deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_ohem_loss.py"
    "deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic2026.py"
    "deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic2026_ohem_loss.py"
    "deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic2026_and_phenobench.py"
    "fcn_hr18_1xb4-30k_1024x1024_phenobench"
    "fcn_hr18_1xb4-30k_1024x1024_phenobench_ohem_loss.py"
    "fcn_hr18_1xb4-30k_1024x1024_sugarbeetsynthetic2026.py"
    "fcn_hr18_1xb4-30k_1024x1024_sugarbeetsynthetic2026_and_phenobench.py"
    "segformer_mit-b2_1xb2-30k_1024x1024_phenobench_vanilla.py"
    "segformer_mit-b2_1xb2-30k_1024x1024_phenobench_ohem_loss.py"
    "segformer_mit-b2_1xb2-30k_1024x1024_sugarbeetsynthetic2026_vanilla.py"
    "mask2former_r50_1024x1024_phenobench.py"
    "mask2former_r50_1024x1024_sugarbeetsynthetic2026.py"
    "mask2former_r50_1024x1024_sugarbeetsynthetic2026_and_phenobench.py"
    "mask2former_swin-t_1024x1024_phenobench.py"
    "mask2former_swin-t_1024x1024_sugarbeetsynthetic2026.py"
    "mask2former_swin-t_1024x1024_sugarbeetsynthetic2026_and_phenobench.py"
)

MODE="submit"
if [[ "${1:-}" == "--dry-run" ]]; then
    MODE="dry-run"
elif [[ "${1:-}" == "--print-trainings" ]]; then
    MODE="print-trainings"
fi

model_display_name() {
    local model_key="$1"
    case "$model_key" in
        deeplabv3plus) echo "Deeplabv3Plus" ;;
        mask2former) echo "Mask2Former" ;;
        segformer) echo "Segformer" ;;
        fcn) echo "FCN" ;;
        pspnet) echo "PSPNet" ;;
        hrnet) echo "HRNet" ;;
        upernet) echo "UPerNet" ;;
        *) echo "${model_key^}" ;;
    esac
}

normalize_config_name() {
    local config_file="$1"

    # Backward-compatible aliasing: transparently map legacy syclops names
    # to the canonical sugarbeetsynthetic2026 naming.
    config_file="${config_file//_syclops_and_phenobench/_sugarbeetsynthetic2026_and_phenobench}"
    config_file="${config_file//_syclops_ohem_loss/_sugarbeetsynthetic2026_ohem_loss}"
    config_file="${config_file//_syclops/_sugarbeetsynthetic2026}"

    echo "$config_file"
}

resolve_config_path() {
    local config_file="$1"
    config_file="$(normalize_config_name "$config_file")"

    if [[ "$config_file" != *.py ]]; then
        config_file="${config_file}.py"
    fi

    if [[ "$config_file" == */* ]]; then
        echo "$config_file"
    else
        echo "${CONFIG_DIR%/}/$config_file"
    fi
}

build_work_dir_name() {
    local config_file="$1"
    config_file="$(normalize_config_name "$config_file")"

    local filename="${config_file##*/}"
    local stem="${filename%.py}"

    IFS='_' read -r -a parts <<< "$stem"

    local model_key="${parts[0]:-unknown_model}"
    local backbone="${parts[1]:-unknown_backbone}"
    local model_name
    local dataset="phenobench"
    local variant
    local -a metadata_tokens=()
    local -a variant_tokens=()

    model_name="$(model_display_name "$model_key")"

    # Keep backbones like mit-b5/swin-t, but simplify r50-d8 -> r50, r101-d16 -> r101.
    backbone="${backbone%-d[0-9]*}"

    local part
    for part in "${parts[@]:2}"; do
        # Ignore schedule markers like 8xb1-30k.
        if [[ "$part" =~ ^[0-9]+xb[0-9]+-[0-9]+k$ ]]; then
            continue
        fi

        # Ignore pure resolution markers like 1024x1024.
        if [[ "$part" =~ ^[0-9]+x[0-9]+$ ]]; then
            continue
        fi

        metadata_tokens+=("$part")
    done

    # Dataset token conventions used in this project:
    # - syclops / sugarbeetsynthetic2026
    # - phenobench / phenobench3
    # - syclops_and_phenobench or sugarbeetsynthetic2026_and_phenobench
    #   (rendered as sugarbeetsynthetic2026_2phenobench)
    local idx=0
    if [[ ${#metadata_tokens[@]} -gt 0 ]]; then
        if [[ "${metadata_tokens[0]}" == "syclops" || "${metadata_tokens[0]}" == "sugarbeetsynthetic2026" ]]; then
            if [[ ${#metadata_tokens[@]} -ge 3 && "${metadata_tokens[1]}" == "and" && "${metadata_tokens[2]}" == "phenobench" ]]; then
                dataset="sugarbeetsynthetic2026_2phenobench"
                idx=3
            else
                dataset="sugarbeetsynthetic2026"
                idx=1
            fi
        elif [[ "${metadata_tokens[0]}" == "phenobench" || "${metadata_tokens[0]}" == "phenobench3" ]]; then
            dataset="phenobench"
            idx=1
        fi
    fi

    for ((; idx < ${#metadata_tokens[@]}; idx++)); do
        variant_tokens+=("${metadata_tokens[$idx]}")
    done

    # Avoid producing ..._baseline_baseline if baseline is already in config name.
    if [[ ${#variant_tokens[@]} -gt 0 ]]; then
        local last_index=$(( ${#variant_tokens[@]} - 1 ))
        if [[ "${variant_tokens[$last_index]}" == "baseline" ]]; then
            unset "variant_tokens[$last_index]"
        fi
    fi

    if [[ ${#variant_tokens[@]} -eq 0 ]]; then
        variant="vanilla"
    else
        local IFS="_"
        variant="${variant_tokens[*]}"
    fi

    # Baseline is only for pure phenobench runs with default loss.
    if [[ "$dataset" == "phenobench" && "$variant" == "vanilla" ]]; then
        echo "${model_name}_${backbone}_${dataset}_baseline"
    elif [[ "$variant" == "vanilla" ]]; then
        echo "${model_name}_${backbone}_${dataset}"
    else
        echo "${model_name}_${backbone}_${dataset}_${variant}"
    fi
}

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "No configs found in CONFIG_FILES." >&2
    exit 1
fi

scheduled_count=0
skipped_count=0
for config_item in "${CONFIG_FILES[@]}"; do
    config_path="$(resolve_config_path "$config_item")"
    work_dir_name="$(build_work_dir_name "$config_item")"
    work_dir="${OUTPUT_ROOT%/}/${work_dir_name}/"

    if [[ ! -f "$config_path" ]]; then
        echo "[WARN] Config does not exist, skipping: $config_path" >&2
        skipped_count=$((skipped_count + 1))
        continue
    fi

    if [[ "$MODE" == "print-trainings" ]]; then
        echo "${config_path};${work_dir}"
        scheduled_count=$((scheduled_count + 1))
        continue
    fi

    echo "Scheduling training for config: $config_path"
    echo "Output directory: $work_dir"

    if [[ "$MODE" == "dry-run" ]]; then
        echo "[DRY-RUN] sbatch sbatch_train.sh \"$config_path\" \"$work_dir\""
    else
        sbatch sbatch_train.sh "$config_path" "$work_dir"
    fi

    scheduled_count=$((scheduled_count + 1))
    echo "--------------------------------------------------"
done

if [[ "$MODE" == "print-trainings" ]]; then
    echo "Printed ${scheduled_count} training entries. Skipped: ${skipped_count}"
elif [[ "$MODE" == "dry-run" ]]; then
    echo "Dry run complete. Jobs prepared: ${scheduled_count}, Skipped: ${skipped_count}"
else
    echo "All training jobs have been scheduled. Total: ${scheduled_count}, Skipped: ${skipped_count}"
fi
