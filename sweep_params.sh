#!/bin/bash
#
# Tile/Block Size Parameter Sweep Script
# Usage: ./sweep_params.sh <config_number>
#
# Configurations:
#   1 = Small:   TILE_M=32, TILE_N=32, TILE_K=16, BLOCK_SIZE=256, FLASH_TILE_KV=32
#   2 = Default: TILE_M=64, TILE_N=64, TILE_K=32, BLOCK_SIZE=256, FLASH_TILE_KV=64
#   3 = Large:   TILE_M=128, TILE_N=128, TILE_K=64, BLOCK_SIZE=1024, FLASH_TILE_KV=128
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYERS="$SCRIPT_DIR/glm_asr_triton_template-Used/layers.py"
ATTENTION="$SCRIPT_DIR/glm_asr_triton_template-Used/attention.py"
FOLDER="glm_asr_triton_template-Used"

# --- Backup originals so we can always restore ---
backup_files() {
    cp "$LAYERS" "$LAYERS.bak"
    cp "$ATTENTION" "$ATTENTION.bak"
    echo "[sweep] Backed up original files."
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
    echo "[sweep] Restored original files."
}

# Restore on exit (Ctrl-C, error, or normal exit)
trap restore_files EXIT

# --- Apply a configuration ---
# Arguments: TILE_M TILE_N TILE_K BLOCK_SIZE FLASH_TILE_KV
apply_config() {
    local TM=$1 TN=$2 TK=$3 BS=$4 FKV=$5

    echo "[sweep] Applying config: TILE_M=$TM, TILE_N=$TN, TILE_K=$TK, BLOCK_SIZE=$BS, FLASH_TILE_KV=$FKV"

    # --- layers.py ---

    # Linear class: individual class-level assignments (lines ~840-842)
    #   TILE_M = <old>  ->  TILE_M = <new>
    #   TILE_N = <old>  ->  TILE_N = <new>
    #   TILE_K = <old>  ->  TILE_K = <new>
    sed -i -E "s/(    TILE_M = )[0-9]+/\1${TM}/" "$LAYERS"
    sed -i -E "s/(    TILE_N = )[0-9]+/\1${TN}/" "$LAYERS"
    sed -i -E "s/(    TILE_K = )[0-9]+/\1${TK}/" "$LAYERS"

    # MLP and EncoderMLP classes: tuple assignment (lines ~1048, ~1175)
    #   TILE_M, TILE_N, TILE_K = 64, 64, 32  ->  TILE_M, TILE_N, TILE_K = TM, TN, TK
    sed -i -E "s/(    TILE_M, TILE_N, TILE_K = )[0-9]+, [0-9]+, [0-9]+/\1${TM}, ${TN}, ${TK}/" "$LAYERS"

    # gelu() and silu(): block = 256  (standalone functions, lines ~799, ~816)
    sed -i -E "s/(    block = )[0-9]+/\1${BS}/" "$LAYERS"

    # --- attention.py ---

    # FLASH_TILE_KV = <old>  ->  FLASH_TILE_KV = <new>
    sed -i -E "s/(FLASH_TILE_KV = )[0-9]+/\1${FKV}/" "$ATTENTION"

    echo "[sweep] Configuration applied."
}

# --- Validate argument ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_number> [benchmark_args...]"
    echo ""
    echo "  config_number: 1 (Small), 2 (Default), 3 (Large)"
    echo "  benchmark_args: extra arguments passed to benchmark.sh"
    exit 1
fi

CONFIG="$1"
shift  # remaining args passed to benchmark

case "$CONFIG" in
    1)
        CONFIG_NAME="Small"
        TM=32; TN=32; TK=16; BS=256; FKV=32
        ;;
    2)
        CONFIG_NAME="Default"
        TM=64; TN=64; TK=32; BS=256; FKV=64
        ;;
    3)
        CONFIG_NAME="Large"
        TM=128; TN=128; TK=64; BS=1024; FKV=128
        ;;
    *)
        echo "Error: Invalid config number '$CONFIG'. Use 1, 2, or 3."
        exit 1
        ;;
esac

echo "============================================"
echo " Parameter Sweep - Config $CONFIG ($CONFIG_NAME)"
echo "============================================"

backup_files
apply_config "$TM" "$TN" "$TK" "$BS" "$FKV"

echo ""
echo "[sweep] Running benchmark for config $CONFIG ($CONFIG_NAME)..."
echo ""

cd "$SCRIPT_DIR"
bash benchmark.sh "$FOLDER" "$@"

echo ""
echo "[sweep] Config $CONFIG ($CONFIG_NAME) complete."
