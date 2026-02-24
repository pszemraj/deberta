#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
RUN_FILTER="${RUN_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

declare -a RUNS=(
  "configs/custom/compile_stability/hf1024_eager_gdes_700.yaml"
  "configs/custom/compile_stability/hf1024_compile_default_gdes_700.yaml"
  "configs/custom/compile_stability/hf1024_compile_default_none_700.yaml"
  "configs/custom/compile_stability/hf1024_compile_default_es_700.yaml"
  "configs/custom/compile_stability/hf512_compile_default_gdes_4000.yaml"
)

if [[ -n "$RUN_FILTER" ]]; then
  declare -a FILTERED=()
  for cfg in "${RUNS[@]}"; do
    if [[ "$cfg" == *"$RUN_FILTER"* ]]; then
      FILTERED+=("$cfg")
    fi
  done
  RUNS=("${FILTERED[@]}")
fi

if [[ "${#RUNS[@]}" -eq 0 ]]; then
  echo "No configs matched RUN_FILTER='$RUN_FILTER'"
  exit 2
fi

echo "Running ${#RUNS[@]} compile-default stability jobs"
echo "Project/output dirs are fixed in configs/custom/compile_stability/*"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR RUN_FILTER=${RUN_FILTER:-<none>} DRY_RUN=$DRY_RUN"

overall_status=0
for cfg in "${RUNS[@]}"; do
  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] START: $cfg"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] conda run --name neobert deberta train $cfg"
    continue
  fi

  if conda run --name neobert deberta train "$cfg"; then
    echo "[$(date '+%F %T')] DONE : $cfg"
  else
    rc=$?
    overall_status=$rc
    echo "[$(date '+%F %T')] FAIL : $cfg (exit=$rc)"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$rc"
    fi
  fi
done

exit "$overall_status"
