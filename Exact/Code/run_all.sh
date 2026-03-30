#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# Runs cvrp_ortools.py on every .vrp file in a dataset directory and stores
# all outputs (solution .txt and route .png) in a single output directory.
#
# Usage:
#   bash run_all.sh [OPTIONS]
#
# Options:
#   -d, --dataset     DIR   Dataset directory (default: ../../Dataset/A)
#   -o, --output      DIR   Output directory  (default: ../outputs)
#   -c, --capacity    INT   Vehicle capacity override (default: not set)
#   -k, --n_vehicles  INT   Number of vehicles override (default: not set)
#   -t, --time_limit  INT   Seconds per instance (default: 120)
#   -n, --max_nodes   INT   Max B&B nodes per instance (default: -1 unlimited)
#   -s, --strategy    STR   First solution strategy (default: path_cheapest_arc)
#   -l, --local_search STR  Local search metaheuristic (default: guided_local_search)
#       --log_search        Enable OR-Tools solver progress logging
#   -h, --help              Show this help message
#
# Examples:
#   bash run_all.sh
#   bash run_all.sh -d ../../Dataset/B -o ../outputs/B --time_limit 60
#   bash run_all.sh --capacity 100 --n_vehicles 5 --time_limit 120
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET_DIR="../../Dataset/A"
OUTPUT_DIR="../outputs/Ortools"
CAPACITY=""
N_VEHICLES=""
TIME_LIMIT=360
MAX_NODES=-1
STRATEGY="path_cheapest_arc"
LOCAL_SEARCH="guided_local_search"
LOG_SEARCH=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER="${SCRIPT_DIR}/cvrp_ortools.py"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '3,32p' "$0"   # print the header comment block
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)      DATASET_DIR="$2";  shift 2 ;;
        -o|--output)       OUTPUT_DIR="$2";   shift 2 ;;
        -c|--capacity)     CAPACITY="$2";     shift 2 ;;
        -k|--n_vehicles)   N_VEHICLES="$2";   shift 2 ;;
        -t|--time_limit)   TIME_LIMIT="$2";   shift 2 ;;
        -n|--max_nodes)    MAX_NODES="$2";    shift 2 ;;
        -s|--strategy)     STRATEGY="$2";     shift 2 ;;
        -l|--local_search) LOCAL_SEARCH="$2"; shift 2 ;;
           --log_search)   LOG_SEARCH=true;   shift   ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ ! -f "$SOLVER" ]]; then
    echo "ERROR: Solver not found at $SOLVER"
    echo "       Make sure cvrp_ortools.py is in the same directory as this script."
    exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Collect all .vrp files
mapfile -t VRP_FILES < <(find "$DATASET_DIR" -maxdepth 1 -name "*.vrp" | sort)

if [[ ${#VRP_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No .vrp files found in $DATASET_DIR"
    exit 1
fi

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Build reusable argument string ───────────────────────────────────────────
EXTRA_ARGS="--time_limit ${TIME_LIMIT} --max_nodes ${MAX_NODES} --strategy ${STRATEGY} --local_search ${LOCAL_SEARCH}"
[[ -n "$CAPACITY"   ]] && EXTRA_ARGS+=" --capacity ${CAPACITY}"
[[ -n "$N_VEHICLES" ]] && EXTRA_ARGS+=" --n_vehicles ${N_VEHICLES}"
[[ "$LOG_SEARCH" == true ]] && EXTRA_ARGS+=" --log_search"

# ── Print run configuration ───────────────────────────────────────────────────
echo "============================================================"
echo "  CVRP Batch Solver"
echo "============================================================"
echo "  Dataset dir  : $DATASET_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Solver       : $SOLVER"
echo "  Instances    : ${#VRP_FILES[@]} .vrp files found"
echo "  Time limit   : ${TIME_LIMIT}s per instance"
echo "  Max nodes    : $MAX_NODES"
echo "  Strategy     : $STRATEGY"
echo "  Local search : $LOCAL_SEARCH"
[[ -n "$CAPACITY"   ]] && echo "  Capacity     : $CAPACITY"
[[ -n "$N_VEHICLES" ]] && echo "  Vehicles     : $N_VEHICLES"
[[ "$LOG_SEARCH" == true ]] && echo "  Log search   : enabled"
echo "============================================================"
echo ""

# ── Run solver on each file ───────────────────────────────────────────────────
TOTAL=${#VRP_FILES[@]}
PASSED=0
FAILED=0
FAILED_FILES=()

START_TIME=$(date +%s)

for i in "${!VRP_FILES[@]}"; do
    VRP_FILE="${VRP_FILES[$i]}"
    INSTANCE=$(basename "$VRP_FILE" .vrp)
    IDX=$((i + 1))

    echo "------------------------------------------------------------"
    echo "  [$IDX/$TOTAL] $INSTANCE"
    echo "------------------------------------------------------------"

    if python3 "$SOLVER" "$VRP_FILE" \
            --output_dir "$OUTPUT_DIR" \
            $EXTRA_ARGS; then
        PASSED=$((PASSED + 1))
        echo "  ✓ Done → $OUTPUT_DIR/${INSTANCE}_solution.txt"
        echo "         → $OUTPUT_DIR/${INSTANCE}_plot.png"
    else
        FAILED=$((FAILED + 1))
        FAILED_FILES+=("$INSTANCE")
        echo "  ✗ FAILED: $INSTANCE"
    fi

    echo ""
done

# ── Final summary ─────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "============================================================"
echo "  BATCH COMPLETE"
echo "============================================================"
echo "  Total instances : $TOTAL"
echo "  Succeeded       : $PASSED"
echo "  Failed          : $FAILED"
echo "  Total time      : ${MINS}m ${SECS}s"
echo "  Output dir      : $OUTPUT_DIR"
echo ""

if [[ ${#FAILED_FILES[@]} -gt 0 ]]; then
    echo "  Failed instances:"
    for f in "${FAILED_FILES[@]}"; do
        echo "    - $f"
    done
    echo ""
fi

echo "  Output files written:"
TXT_COUNT=$(find "$OUTPUT_DIR" -name "*_solution.txt" | wc -l)
PNG_COUNT=$(find "$OUTPUT_DIR" -name "*_plot.png"     | wc -l)
echo "    ${TXT_COUNT} solution .txt files"
echo "    ${PNG_COUNT} route plot .png files"
echo "============================================================"