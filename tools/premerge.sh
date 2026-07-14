#!/usr/bin/env bash
# Pre-merge verification gate: lint, docstring contracts, full test suite, and
# the CUDA FlashDeBERTa parity harness, with a persistent log per invocation.
# CI deliberately covers only import/packaging contracts; this script is the
# authoritative correctness gate and must run on the GPU dev box before merge.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/local-scratch/premerge"
DOC_CHECK_SCRIPT="${HOME}/scripts/py/doc_check.py"
CONDA_RUN=(conda run --name neobert --no-capture-output)
ALL_STEPS=(lint docstrings tests parity)

usage() {
    cat <<'EOF'
Run the pre-merge verification gate and record a log tied to the current commit.

Usage: tools/premerge.sh [step ...]

Steps (default: all, in order):
  lint        ruff check + ruff format --check (non-mutating)
  docstrings  doc_check.py over src/
  tests       full pytest suite (CUDA tests run when a GPU is visible)
  parity      FlashDeBERTa route parity harness (requires CUDA)

The tests and parity steps take several minutes on a GPU box. Logs are written
to local-scratch/premerge/premerge_<stamp>_<sha>.log.
EOF
}

steps=()
while (($# > 0)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        lint|docstrings|tests|parity) steps+=("$1"); shift ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
((${#steps[@]})) || steps=("${ALL_STEPS[@]}")

cd "${ROOT_DIR}"
GIT_SHA="$(git rev-parse --short HEAD)"
DIRTY_COUNT="$(git status --porcelain | wc -l)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/premerge_${STAMP}_${GIT_SHA}.log"
exec > >(tee "${LOG_FILE}") 2>&1

current_step="startup"
on_failure() {
    echo
    echo "PREMERGE FAIL at ${GIT_SHA} (step: ${current_step}); log: ${LOG_FILE}"
}
trap on_failure ERR

echo "premerge gate @ $(date -Is)"
echo "commit: $(git rev-parse HEAD) (branch: $(git rev-parse --abbrev-ref HEAD))"
if ((DIRTY_COUNT > 0)); then
    echo "WARNING: ${DIRTY_COUNT} uncommitted change(s); this log does not vouch for commit ${GIT_SHA} alone."
fi
echo "steps: ${steps[*]}"

run_step() {
    current_step="$1"
    local start=${SECONDS}
    echo
    echo "=== ${current_step} ==="
    case "${current_step}" in
        lint)
            "${CONDA_RUN[@]}" ruff check .
            "${CONDA_RUN[@]}" ruff format --check .
            ;;
        docstrings)
            if [[ ! -f "${DOC_CHECK_SCRIPT}" ]]; then
                echo "doc_check.py not found at ${DOC_CHECK_SCRIPT}" >&2
                return 1
            fi
            "${CONDA_RUN[@]}" python "${DOC_CHECK_SCRIPT}" src/
            ;;
        tests)
            "${CONDA_RUN[@]}" pytest tests/
            ;;
        parity)
            "${CONDA_RUN[@]}" python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "parity requires a CUDA GPU")'
            "${CONDA_RUN[@]}" python tools/flashdeberta_parity_test.py
            ;;
    esac
    echo "--- ${current_step} OK ($((SECONDS - start))s)"
}

for step in "${steps[@]}"; do
    run_step "${step}"
done

echo
echo "PREMERGE PASS at ${GIT_SHA}; log: ${LOG_FILE}"
