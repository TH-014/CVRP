#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# Runs both GA solvers (Original and Improved) on every .vrp file in a
# dataset directory and stores outputs (solution .txt and route .png) in
# their respective output sub-directories.
#
# Usage:
#   bash run_all.sh [OPTIONS]
#
# Options:
#   -d, --dataset     DIR   Dataset directory (default: ../../A/A)
#   -t, --time_limit  INT   Seconds per instance per algorithm (default: 120)
#   -p, --pop_size    INT   GA population size (default: 100)
#   -g, --generations INT   Number of GA generations (default: 300)
#   -s, --seed        INT   Random seed (default: 42)
#   -c, --capacity    INT   Vehicle capacity override (default: not set)
#   -k, --n_vehicles  INT   Number of vehicles override (default: not set)
#       --original_only     Run only the Original GA
#       --improved_only     Run only the Improved GA
#   -h, --help              Show this help message
#
# Examples:
#   bash run_all.sh
#   bash run_all.sh -d ../../E/E -t 60
#   bash run_all.sh -d ../../Golden/Golden -t 300 -p 50 -g 100
#   bash run_all.sh --improved_only -d ../../A/A
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET_DIR="../../Dataset/A"
TIME_LIMIT=120
POP_SIZE=100
GENERATIONS=300
SEED=42
CAPACITY=""
N_VEHICLES=""
RUN_ORIGINAL=true
RUN_IMPROVED=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_SOLVER="${SCRIPT_DIR}/cvrp_ga_original.py"
IMPR_SOLVER="${SCRIPT_DIR}/cvrp_ga_improved.py"
ORIG_OUT="${SCRIPT_DIR}/../outputs/original_ga"
IMPR_OUT="${SCRIPT_DIR}/../outputs/improved_ga"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '3,30p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)      DATASET_DIR="$2";  shift 2 ;;
        -t|--time_limit)   TIME_LIMIT="$2";   shift 2 ;;
        -p|--pop_size)     POP_SIZE="$2";     shift 2 ;;
        -g|--generations)  GENERATIONS="$2";  shift 2 ;;
        -s|--seed)         SEED="$2";         shift 2 ;;
        -c|--capacity)     CAPACITY="$2";     shift 2 ;;
        -k|--n_vehicles)   N_VEHICLES="$2";   shift 2 ;;
           --original_only) RUN_IMPROVED=false; shift ;;
           --improved_only) RUN_ORIGINAL=false; shift ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

mapfile -t VRP_FILES < <(find "$DATASET_DIR" -maxdepth 1 -name "*.vrp" | sort)
if [[ ${#VRP_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No .vrp files found in $DATASET_DIR"
    exit 1
fi

mkdir -p "$ORIG_OUT" "$IMPR_OUT"

# ── Build reusable extra args ─────────────────────────────────────────────────
COMMON_ARGS="--time_limit ${TIME_LIMIT} --pop_size ${POP_SIZE} --generations ${GENERATIONS} --seed ${SEED}"
[[ -n "$CAPACITY"   ]] && COMMON_ARGS+=" --capacity ${CAPACITY}"
[[ -n "$N_VEHICLES" ]] && COMMON_ARGS+=" --n_vehicles ${N_VEHICLES}"

# ── Print configuration ───────────────────────────────────────────────────────
echo "============================================================"
echo "  CVRP Metaheuristic Batch Solver"
echo "============================================================"
echo "  Dataset dir  : $DATASET_DIR"
echo "  Instances    : ${#VRP_FILES[@]} .vrp files"
echo "  Time limit   : ${TIME_LIMIT}s per instance per algorithm"
echo "  Population   : $POP_SIZE"
echo "  Generations  : $GENERATIONS"
echo "  Seed         : $SEED"
[[ -n "$CAPACITY"   ]] && echo "  Capacity     : $CAPACITY"
[[ -n "$N_VEHICLES" ]] && echo "  Vehicles     : $N_VEHICLES"
echo "  Run Original : $RUN_ORIGINAL"
echo "  Run Improved : $RUN_IMPROVED"
echo "  Orig output  : $ORIG_OUT"
echo "  Impr output  : $IMPR_OUT"
echo "============================================================"
echo ""

# ── Run solvers ───────────────────────────────────────────────────────────────
TOTAL=${#VRP_FILES[@]}
ORIG_PASSED=0; ORIG_FAILED=0
IMPR_PASSED=0; IMPR_FAILED=0
FAILED_FILES=()

START_TIME=$(date +%s)

for i in "${!VRP_FILES[@]}"; do
    VRP_FILE="${VRP_FILES[$i]}"
    INSTANCE=$(basename "$VRP_FILE" .vrp)
    IDX=$((i + 1))

    echo "------------------------------------------------------------"
    echo "  [$IDX/$TOTAL] $INSTANCE"
    echo "------------------------------------------------------------"

    if [[ "$RUN_ORIGINAL" == true ]]; then
        echo "  >> Original GA"
        if python3 "$ORIG_SOLVER" "$VRP_FILE" \
                --output_dir "$ORIG_OUT" $COMMON_ARGS; then
            ORIG_PASSED=$((ORIG_PASSED + 1))
            echo "     Done -> ${ORIG_OUT}/${INSTANCE}_solution.txt"
        else
            ORIG_FAILED=$((ORIG_FAILED + 1))
            FAILED_FILES+=("${INSTANCE} [original]")
            echo "     FAILED: $INSTANCE (original)"
        fi
        echo ""
    fi

    if [[ "$RUN_IMPROVED" == true ]]; then
        echo "  >> Improved GA"
        if python3 "$IMPR_SOLVER" "$VRP_FILE" \
                --output_dir "$IMPR_OUT" $COMMON_ARGS; then
            IMPR_PASSED=$((IMPR_PASSED + 1))
            echo "     Done -> ${IMPR_OUT}/${INSTANCE}_solution.txt"
        else
            IMPR_FAILED=$((IMPR_FAILED + 1))
            FAILED_FILES+=("${INSTANCE} [improved]")
            echo "     FAILED: $INSTANCE (improved)"
        fi
        echo ""
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "============================================================"
echo "  BATCH COMPLETE"
echo "============================================================"
echo "  Total instances   : $TOTAL"
[[ "$RUN_ORIGINAL" == true ]] && echo "  Original GA  pass : $ORIG_PASSED  fail: $ORIG_FAILED"
[[ "$RUN_IMPROVED" == true ]] && echo "  Improved GA  pass : $IMPR_PASSED  fail: $IMPR_FAILED"
echo "  Total time        : ${MINS}m ${SECS}s"
echo ""

if [[ ${#FAILED_FILES[@]} -gt 0 ]]; then
    echo "  Failed runs:"
    for f in "${FAILED_FILES[@]}"; do echo "    - $f"; done
    echo ""
fi

TXT_ORIG=$(find "$ORIG_OUT" -name "*_solution.txt" 2>/dev/null | wc -l)
PNG_ORIG=$(find "$ORIG_OUT" -name "*_plot.png"     2>/dev/null | wc -l)
TXT_IMPR=$(find "$IMPR_OUT" -name "*_solution.txt" 2>/dev/null | wc -l)
PNG_IMPR=$(find "$IMPR_OUT" -name "*_plot.png"     2>/dev/null | wc -l)

echo "  Output files:"
echo "    Original GA : ${TXT_ORIG} solution.txt  ${PNG_ORIG} plot.png"
echo "    Improved GA : ${TXT_IMPR} solution.txt  ${PNG_IMPR} plot.png"
echo ""
echo "  To generate results.csv run:"
echo "    python3 generate_csv.py"
echo "============================================================"
