#!/bin/bash
#
# Run all 3 parameter sweep configurations and save results.
#
# Usage: ./sweep_all.sh [benchmark_args...]
#
# Output is saved to sweep_results_<timestamp>.txt
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$SCRIPT_DIR/sweep_results_${TIMESTAMP}.txt"
LAYERS="$SCRIPT_DIR/glm_asr_triton_template-Used/layers.py"
ATTENTION="$SCRIPT_DIR/glm_asr_triton_template-Used/attention.py"
FOLDER="glm_asr_triton_template-Used"

# --- Backup originals so we can always restore ---
backup_files() {
    cp "$LAYERS" "$LAYERS.bak"
    cp "$ATTENTION" "$ATTENTION.bak"
    echo "[sweep_all] Backed up original files."
}

restore_files() {
    if [ -f "$LAYERS.bak" ]; then
        cp "$LAYERS.bak" "$LAYERS"
        rm -f "$LAYERS.bak"
    fi
    if [ -f "$ATTENTION.bak" ]; then
        cp "$ATTENTION.bak" "$ATTENTION"
        rm -f "$ATTENTION.bak"
    fi
    echo "[sweep_all] Restored original files."
}

trap restore_files EXIT

apply_config() {
    local TM=$1 TN=$2 TK=$3 BS=$4 FKV=$5

    # Restore from backup first so sed always operates on clean originals
    cp "$LAYERS.bak" "$LAYERS"
    cp "$ATTENTION.bak" "$ATTENTION"

    # layers.py: Linear class individual assignments
    sed -i -E "s/(    TILE_M = )[0-9]+/\1${TM}/" "$LAYERS"
    sed -i -E "s/(    TILE_N = )[0-9]+/\1${TN}/" "$LAYERS"
    sed -i -E "s/(    TILE_K = )[0-9]+/\1${TK}/" "$LAYERS"

    # layers.py: MLP / EncoderMLP tuple assignments
    sed -i -E "s/(    TILE_M, TILE_N, TILE_K = )[0-9]+, [0-9]+, [0-9]+/\1${TM}, ${TN}, ${TK}/" "$LAYERS"

    # layers.py: gelu/silu block size
    sed -i -E "s/(    block = )[0-9]+/\1${BS}/" "$LAYERS"

    # attention.py: FLASH_TILE_KV
    sed -i -E "s/(FLASH_TILE_KV = )[0-9]+/\1${FKV}/" "$ATTENTION"
}

# --- Header ---
{
    echo "============================================================"
    echo " GLM-ASR Triton Parameter Sweep Results"
    echo " Date: $(date)"
    echo " Extra args: $*"
    echo "============================================================"
    echo ""
} | tee "$RESULTS_FILE"

backup_files

# --- Configs ---
CONFIGS=(
    "1 Small   32  32  16  256  32"
    "2 Default 64  64  32  256  64"
    "3 Large   128 128 64  1024 128"
)

for entry in "${CONFIGS[@]}"; do
    read -r NUM NAME TM TN TK BS FKV <<< "$entry"

    {
        echo "------------------------------------------------------------"
        echo " Config $NUM: $NAME"
        echo "   TILE_M=$TM  TILE_N=$TN  TILE_K=$TK"
        echo "   BLOCK_SIZE=$BS  FLASH_TILE_KV=$FKV"
        echo "------------------------------------------------------------"
    } | tee -a "$RESULTS_FILE"

    apply_config "$TM" "$TN" "$TK" "$BS" "$FKV"

    echo "" | tee -a "$RESULTS_FILE"
    echo "[sweep_all] Running benchmark for Config $NUM ($NAME)..." | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"

    cd "$SCRIPT_DIR"
    bash benchmark.sh "$FOLDER" "$@" 2>&1 | tee -a "$RESULTS_FILE"

    echo "" | tee -a "$RESULTS_FILE"
    echo "[sweep_all] Config $NUM ($NAME) finished." | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
done

{
    echo "============================================================"
    echo " Sweep complete. Results saved to:"
    echo "   $RESULTS_FILE"
    echo "============================================================"
} | tee -a "$RESULTS_FILE"
