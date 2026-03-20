#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2.yaml}"
DOCBLOCK_CONFIG_PATH="${FLASHDEBERTA_DOCBLOCK_CONFIG_PATH:-configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUT_DIR="${ROOT_DIR}/local-scratch/benchmarks/flashdeberta/flashdeberta_bench_${STAMP}"
OUT_DIR="${FLASHDEBERTA_BENCH_OUT_DIR:-${DEFAULT_OUT_DIR}}"

MICRO_WARMUP="${FLASHDEBERTA_MICRO_WARMUP:-10}"
MICRO_STEPS="${FLASHDEBERTA_MICRO_STEPS:-30}"
PACKED_MAX_STEPS="${FLASHDEBERTA_PACKED_MAX_STEPS:-100}"
UNPACKED_MAX_STEPS="${FLASHDEBERTA_UNPACKED_MAX_STEPS:-100}"
DOCBLOCK_MAX_STEPS="${FLASHDEBERTA_DOCBLOCK_MAX_STEPS:-100}"
LOGGING_STEPS="${FLASHDEBERTA_LOGGING_STEPS:-10}"
AVG_FROM_STEP="${FLASHDEBERTA_AVG_FROM_STEP:-20}"
INCLUDE_DOCBLOCK="${FLASHDEBERTA_INCLUDE_DOCBLOCK:-0}"

mkdir -p "${OUT_DIR}"

run_case() {
    local name="$1"
    shift

    local log_path="${OUT_DIR}/${name}.log"
    local meta_path="${OUT_DIR}/${name}.meta"
    local start_ts end_ts elapsed_s

    echo "==> ${name}"
    echo "    log: ${log_path}"

    start_ts="$(date +%s)"
    (
        cd "${ROOT_DIR}"
        "$@"
    ) >"${log_path}" 2>&1
    end_ts="$(date +%s)"
    elapsed_s="$((end_ts - start_ts))"

    {
        printf 'name=%s\n' "${name}"
        printf 'elapsed_s=%s\n' "${elapsed_s}"
        printf 'log=%s\n' "${log_path}"
    } >"${meta_path}"
}

micro_case() {
    local name="$1"
    shift

    run_case \
        "${name}" \
        conda run --name neobert --no-capture-output python tools/flashdeberta_microbench.py \
        --warmup "${MICRO_WARMUP}" \
        --steps "${MICRO_STEPS}" \
        "$@"
}

train_eager_case() {
    local name="$1"
    local steps="$2"
    local config_path="$3"
    shift 3

    local output_dir="${OUT_DIR}/${name}"
    mkdir -p "${output_dir}"

    run_case \
        "${name}" \
        env HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 TOKENIZERS_PARALLELISM=false \
        conda run --name neobert --no-capture-output deberta train "${config_path}" \
        --train.max_steps "${steps}" \
        --logging.logging_steps "${LOGGING_STEPS}" \
        --train.checkpoint.output_dir "${output_dir}" \
        --logging.output_dir "${output_dir}" \
        --logging.backend none \
        --logging.wandb.enabled false \
        --train.checkpoint.export_hf_final false \
        --train.checkpoint.save_steps 1000000 \
        "$@"
}

train_flash_case() {
    local name="$1"
    local steps="$2"
    local dense_policy="${3:-}"
    local config_path="$4"
    shift 4

    local output_dir="${OUT_DIR}/${name}"
    local -a env_prefix=(
        env
        HF_HUB_DOWNLOAD_TIMEOUT=120
        HF_HUB_ETAG_TIMEOUT=120
        TOKENIZERS_PARALLELISM=false
    )
    mkdir -p "${output_dir}"

    if [[ -n "${dense_policy}" ]]; then
        env_prefix+=("FLASHDEBERTA_EAGER_DENSE_MAX_SEQ_LEN=${dense_policy}")
    fi

    run_case \
        "${name}" \
        "${env_prefix[@]}" \
        conda run --name neobert --no-capture-output python tools/train_flashdeberta.py train "${config_path}" \
        --train.max_steps "${steps}" \
        --logging.logging_steps "${LOGGING_STEPS}" \
        --train.checkpoint.output_dir "${output_dir}" \
        --logging.output_dir "${output_dir}" \
        --logging.backend none \
        --logging.wandb.enabled false \
        --train.checkpoint.export_hf_final false \
        --train.checkpoint.save_steps 1000000 \
        "$@"
}

micro_case micro_eager_dense1024 --mode eager --seq-len 1024 --batch-size 8 --pad-ratio 0.0
micro_case micro_flash_dense1024 --mode flash --seq-len 1024 --batch-size 8 --pad-ratio 0.0
micro_case micro_eager_padded1024 --mode eager --seq-len 1024 --batch-size 8 --pad-ratio 0.35
micro_case micro_flash_padded1024 --mode flash --seq-len 1024 --batch-size 8 --pad-ratio 0.35
micro_case micro_eager_padded2048 --mode eager --seq-len 2048 --batch-size 4 --pad-ratio 0.35
micro_case micro_flash_padded2048 --mode flash --seq-len 2048 --batch-size 4 --pad-ratio 0.35
micro_case micro_eager_padded4096 --mode eager --seq-len 4096 --batch-size 2 --pad-ratio 0.35
micro_case micro_flash_padded4096 --mode flash --seq-len 4096 --batch-size 2 --pad-ratio 0.35

train_eager_case train_packed_eager "${PACKED_MAX_STEPS}" "${CONFIG_PATH}"
train_flash_case train_packed_flash "${PACKED_MAX_STEPS}" "" "${CONFIG_PATH}"
train_flash_case train_packed_flash_densepolicy "${PACKED_MAX_STEPS}" "1024" "${CONFIG_PATH}"
train_eager_case train_unpacked_eager "${UNPACKED_MAX_STEPS}" "${CONFIG_PATH}" --data.packing.enabled false
train_flash_case train_unpacked_flash "${UNPACKED_MAX_STEPS}" "" "${CONFIG_PATH}" --data.packing.enabled false

if [[ "${INCLUDE_DOCBLOCK}" == "1" ]]; then
    train_eager_case train_packed_docblock_eager "${DOCBLOCK_MAX_STEPS}" "${DOCBLOCK_CONFIG_PATH}"
    train_flash_case train_packed_docblock_flash "${DOCBLOCK_MAX_STEPS}" "" "${DOCBLOCK_CONFIG_PATH}"
fi

summary_micro() {
    local name="$1"
    local log_path="${OUT_DIR}/${name}.log"
    local elapsed_s="NA"
    local mean_ms active_tok_s slot_tok_s max_mem_gib flash_stats flash_stats_display

    if [[ -f "${OUT_DIR}/${name}.meta" ]]; then
        elapsed_s="$(awk -F= '/^elapsed_s=/{print $2}' "${OUT_DIR}/${name}.meta")"
    fi

    mean_ms="$(awk '
        {
            if (match($0, /mean_ms=[0-9.]+/)) {
                print substr($0, RSTART + 8, RLENGTH - 8);
            }
        }
    ' "${log_path}")"
    active_tok_s="$(awk '/^active_tok_per_s=/{sub(/^active_tok_per_s=/, "", $0); print $0}' "${log_path}")"
    slot_tok_s="$(awk '/^slot_tok_per_s=/{sub(/^slot_tok_per_s=/, "", $0); print $0}' "${log_path}")"
    max_mem_gib="$(awk '/^max_memory_gib=/{sub(/^max_memory_gib=/, "", $0); print $0}' "${log_path}")"
    flash_stats="$(awk '/^flash_stats=/{sub(/^flash_stats=/, "", $0); print $0}' "${log_path}")"
    flash_stats_display="${flash_stats:-\{\}}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${elapsed_s}" "${mean_ms:-NA}" "${active_tok_s:-NA}" "${slot_tok_s:-NA}" "${max_mem_gib:-NA} ${flash_stats_display}"
}

summary_train() {
    local name="$1"
    local log_path="${OUT_DIR}/${name}.log"
    local elapsed_s="NA"
    local avg_tok_s final_tok_s

    if [[ -f "${OUT_DIR}/${name}.meta" ]]; then
        elapsed_s="$(awk -F= '/^elapsed_s=/{print $2}' "${OUT_DIR}/${name}.meta")"
    fi

    avg_tok_s="$(awk -v min_step="${AVG_FROM_STEP}" '
        {
            step = "";
            tok = "";
            if (match($0, /step=[0-9]+/)) {
                step = substr($0, RSTART + 5, RLENGTH - 5) + 0;
            }
            if (match($0, /tok\/s=[0-9.]+/)) {
                tok = substr($0, RSTART + 6, RLENGTH - 6) + 0.0;
            }
            if (step != "" && tok != "") {
                if (step >= min_step) {
                    sum += tok;
                    count += 1;
                }
                last = tok;
            }
        }
        END {
            if (count > 0) {
                printf "%.2f", sum / count;
            } else {
                printf "NA";
            }
        }
    ' "${log_path}")"

    final_tok_s="$(awk '
        {
            if (match($0, /tok\/s=[0-9.]+/)) {
                last = substr($0, RSTART + 6, RLENGTH - 6);
            }
        }
        END {
            if (last != "") {
                print last;
            } else {
                print "NA";
            }
        }
    ' "${log_path}")"

    printf '%s\t%s\t%s\t%s\n' "${name}" "${elapsed_s}" "${avg_tok_s}" "${final_tok_s}"
}

{
    printf 'output_dir=%s\n' "${OUT_DIR}"
    printf '\n[microbench]\n'
    printf 'name\telapsed_s\tmean_ms\tactive_tok_per_s\tslot_tok_per_s\tmemory_and_stats\n'
    summary_micro micro_eager_dense1024
    summary_micro micro_flash_dense1024
    summary_micro micro_eager_padded1024
    summary_micro micro_flash_padded1024
    summary_micro micro_eager_padded2048
    summary_micro micro_flash_padded2048
    summary_micro micro_eager_padded4096
    summary_micro micro_flash_padded4096
    printf '\n[training]\n'
    printf 'name\telapsed_s\tavg_tok_s_from_step_%s\tfinal_logged_tok_s\n' "${AVG_FROM_STEP}"
    summary_train train_packed_eager
    summary_train train_packed_flash
    summary_train train_packed_flash_densepolicy
    summary_train train_unpacked_eager
    summary_train train_unpacked_flash
    if [[ "${INCLUDE_DOCBLOCK}" == "1" ]]; then
        summary_train train_packed_docblock_eager
        summary_train train_packed_docblock_flash
    fi
} | tee "${OUT_DIR}/summary.tsv"

echo
echo "Benchmark run complete."
echo "Summary: ${OUT_DIR}/summary.tsv"
