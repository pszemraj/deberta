#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUT_DIR="${ROOT_DIR}/local-scratch/benchmarks/flashdeberta/flashdeberta_bench_${STAMP}"
CONFIG_PATH="configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2.yaml"
DOCBLOCK_CONFIG_PATH="configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml"
OUT_DIR="${DEFAULT_OUT_DIR}"
MICRO_WARMUP=10
MICRO_STEPS=30
PACKED_MAX_STEPS=100
UNPACKED_MAX_STEPS=100
DOCBLOCK_MAX_STEPS=100
LOGGING_STEPS=10
AVG_FROM_STEP=20
INCLUDE_DOCBLOCK=0
RETRY_ATTEMPTS=3
RETRY_BACKOFF_SECONDS=5

usage() {
    cat <<'EOF'
Run the FlashDeBERTa benchmark matrix.

Options:
  --config PATH                    Base training YAML.
  --docblock-config PATH           Doc-block training YAML.
  --out-dir PATH                   Output directory (default: timestamped local-scratch path).
  --micro-warmup N                 Microbenchmark warmup iterations (default: 10).
  --micro-steps N                  Microbenchmark measured iterations (default: 30).
  --packed-max-steps N             Packed training steps (default: 100).
  --unpacked-max-steps N           Unpacked training steps (default: 100).
  --docblock-max-steps N           Doc-block training steps (default: 100).
  --logging-steps N                Training logging interval (default: 10).
  --avg-from-step N                First step included in throughput average (default: 20).
  --include-docblock               Run doc-block training cases.
  --retry-attempts N               Attempts for transient data/network failures (default: 3).
  --retry-backoff-seconds N        Initial exponential retry delay (default: 5).
  -h, --help                       Show this help.
EOF
}

while (($# > 0)); do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --docblock-config) DOCBLOCK_CONFIG_PATH="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --micro-warmup) MICRO_WARMUP="$2"; shift 2 ;;
        --micro-steps) MICRO_STEPS="$2"; shift 2 ;;
        --packed-max-steps) PACKED_MAX_STEPS="$2"; shift 2 ;;
        --unpacked-max-steps) UNPACKED_MAX_STEPS="$2"; shift 2 ;;
        --docblock-max-steps) DOCBLOCK_MAX_STEPS="$2"; shift 2 ;;
        --logging-steps) LOGGING_STEPS="$2"; shift 2 ;;
        --avg-from-step) AVG_FROM_STEP="$2"; shift 2 ;;
        --include-docblock) INCLUDE_DOCBLOCK=1; shift ;;
        --retry-attempts) RETRY_ATTEMPTS="$2"; shift 2 ;;
        --retry-backoff-seconds) RETRY_BACKOFF_SECONDS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p "${OUT_DIR}"

run_case() {
    local name="$1"
    shift

    local log_path="${OUT_DIR}/${name}.log"
    local meta_path="${OUT_DIR}/${name}.meta"
    local start_ts end_ts elapsed_s attempt attempts_run delay status

    echo "==> ${name}"
    echo "    log: ${log_path}"

    start_ts="$(date +%s)"
    : >"${log_path}"
    status="failed"
    attempts_run=0
    for ((attempt = 1; attempt <= RETRY_ATTEMPTS; attempt++)); do
        attempts_run="${attempt}"
        printf '\n== attempt %d/%d ==\n' "${attempt}" "${RETRY_ATTEMPTS}" >>"${log_path}"
        if (
            cd "${ROOT_DIR}"
            "$@"
        ) >>"${log_path}" 2>&1; then
            status="success"
            break
        fi
        if ! grep -Eiq \
            'timeout|connection(reset|error|aborted)?|temporar(il)?y unavailable|remote.*disconnect|incomplete(read| download)|chunkedencoding|http[^0-9]*5[0-9][0-9]|shard.*(unavailable|failed)' \
            "${log_path}"; then
            break
        fi
        if ((attempt < RETRY_ATTEMPTS)); then
            delay="$((RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))))"
            printf 'Transient failure; retrying in %ss.\n' "${delay}" | tee -a "${log_path}"
            sleep "${delay}"
        fi
    done
    end_ts="$(date +%s)"
    elapsed_s="$((end_ts - start_ts))"

    {
        printf 'name=%s\n' "${name}"
        printf 'elapsed_s=%s\n' "${elapsed_s}"
        printf 'log=%s\n' "${log_path}"
        printf 'attempts=%s\n' "${attempts_run}"
        printf 'status=%s\n' "${status}"
    } >"${meta_path}"

    if [[ "${status}" != "success" ]]; then
        echo "Case ${name} failed after ${attempts_run} attempt(s); see ${log_path}." >&2
        return 1
    fi
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

train_case() {
    local name="$1"
    local mode="$2"
    local steps="$3"
    local dense_policy="${4:-}"
    local config_path="$5"
    shift 5

    local output_dir="${OUT_DIR}/${name}"
    local -a env_prefix=(
        env
        HF_HUB_DOWNLOAD_TIMEOUT=120
        HF_HUB_ETAG_TIMEOUT=120
        TOKENIZERS_PARALLELISM=false
    )
    local -a mode_args=(--model.hf.attention_impl "${mode}")
    mkdir -p "${output_dir}"

    if [[ "${mode}" == "flash" && -n "${dense_policy}" ]]; then
        mode_args+=(--model.hf.flash.eager_dense_max_seq_len "${dense_policy}")
    fi

    run_case \
        "${name}" \
        "${env_prefix[@]}" \
        conda run --name neobert --no-capture-output deberta train "${config_path}" \
        "${mode_args[@]}" \
        --train.max_steps "${steps}" \
        --logging.logging_steps "${LOGGING_STEPS}" \
        --train.checkpoint.output_dir "${output_dir}" \
        --train.checkpoint.overwrite_output_dir true \
        --logging.output_dir "${output_dir}" \
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

train_case train_packed_eager eager "${PACKED_MAX_STEPS}" "" "${CONFIG_PATH}"
train_case train_packed_flash flash "${PACKED_MAX_STEPS}" "" "${CONFIG_PATH}"
train_case train_packed_flash_densepolicy flash "${PACKED_MAX_STEPS}" "1024" "${CONFIG_PATH}"
train_case train_unpacked_eager eager "${UNPACKED_MAX_STEPS}" "" "${CONFIG_PATH}" --data.packing.enabled false
train_case train_unpacked_flash flash "${UNPACKED_MAX_STEPS}" "" "${CONFIG_PATH}" --data.packing.enabled false

if [[ "${INCLUDE_DOCBLOCK}" == "1" ]]; then
    train_case train_packed_docblock_eager eager "${DOCBLOCK_MAX_STEPS}" "" "${DOCBLOCK_CONFIG_PATH}"
    train_case train_packed_docblock_flash flash "${DOCBLOCK_MAX_STEPS}" "" "${DOCBLOCK_CONFIG_PATH}"
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
