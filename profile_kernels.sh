#!/bin/bash
# =============================================================================
# Triton GPU Kernel Profiling Script for DICE Cluster (NVIDIA A100)
# =============================================================================
# Usage:
#   bash profile_kernels.sh <folder_name>
#   bash profile_kernels.sh glm_asr_triton_template-Used
#
# Prerequisites:
#   - NVIDIA Nsight Systems (nsys) and Nsight Compute (ncu) installed
#   - CUDA toolkit available
#   - Python environment with torch, triton
#
# On DICE cluster, load modules first:
#   module load cuda/12.x
#   module load nsight-systems
#   module load nsight-compute
# =============================================================================

set -euo pipefail

FOLDER="${1:-glm_asr_triton_template-Used}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${SCRIPT_DIR}/profiling_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "Profiling: $FOLDER"
echo "Output directory: $OUTDIR"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"


# =============================================================================
# 1. NSIGHT SYSTEMS -- Kernel-level Timeline Traces
# =============================================================================
# Captures a full timeline of CUDA kernels, memory operations, and NVTX markers.
# Produces a .nsys-rep file for viewing in Nsight Systems GUI and a .sqlite
# export for programmatic analysis.

echo ""
echo "=== Section 1: Nsight Systems Timeline Profiling ==="
echo ""

NSYS_OUTPUT="${OUTDIR}/nsys_${FOLDER}_${TIMESTAMP}"

# Capture timeline with CUDA kernel tracing and memory usage tracking
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --output="${NSYS_OUTPUT}" \
    --force-overwrite=true \
    --stats=true \
    python "${SCRIPT_DIR}/benchmark_student.py" "$FOLDER" --warmup 1 --runs 1

echo ""
echo "Nsight Systems report saved to: ${NSYS_OUTPUT}.nsys-rep"

# Export to SQLite for scripted analysis
nsys export \
    --type=sqlite \
    --output="${NSYS_OUTPUT}.sqlite" \
    "${NSYS_OUTPUT}.nsys-rep"

echo "SQLite export saved to: ${NSYS_OUTPUT}.sqlite"

# Print kernel summary from nsys stats
echo ""
echo "--- Top 20 CUDA Kernels by Time ---"
nsys stats --report cuda_gpu_kern_sum "${NSYS_OUTPUT}.nsys-rep" 2>/dev/null | head -30 || true

echo ""
echo "--- CUDA Memory Operations Summary ---"
nsys stats --report cuda_gpu_mem_size_sum "${NSYS_OUTPUT}.nsys-rep" 2>/dev/null | head -20 || true


# =============================================================================
# 2. NSIGHT COMPUTE -- Detailed Kernel Metrics
# =============================================================================
# Profiles individual Triton kernels with full hardware counter collection.
# --set full collects all available metrics (throughput, occupancy, memory, etc.)
#
# Kernel names in Triton are mangled. Use --kernel-name-base to match by
# substring. Common Triton kernel names from the codebase:
#   linear_kernel_tf32, swiglu_fused_kernel, linear_gelu_kernel,
#   flash_attention_kernel, rmsnorm_kernel, layernorm_kernel,
#   compute_freqs_kernel, silu_kernel, gelu_kernel,
#   attention_scores_kernel, softmax_inplace_kernel, attention_output_kernel,
#   split_k_kernel, split_k_reduce_kernel, embedding_kernel, softmax_kernel

echo ""
echo "=== Section 2: Nsight Compute Kernel Metrics ==="
echo ""

# Target kernel list -- these are the key Triton kernels
KERNELS=(
    "linear_kernel_tf32"
    "swiglu_fused_kernel"
    "linear_gelu_kernel"
    "flash_attention_kernel"
    "compute_freqs_kernel"
    "rmsnorm_kernel"
    "layernorm_kernel"
)

# Full metrics collection for all target kernels
for KERNEL in "${KERNELS[@]}"; do
    echo "--- Profiling kernel: $KERNEL ---"
    NCU_OUTPUT="${OUTDIR}/ncu_${KERNEL}_${TIMESTAMP}"

    ncu \
        --set full \
        --kernel-name "$KERNEL" \
        --launch-count 3 \
        --target-processes all \
        --export "${NCU_OUTPUT}" \
        python "${SCRIPT_DIR}/benchmark_student.py" "$FOLDER" --warmup 0 --runs 1 \
        2>&1 | tee "${NCU_OUTPUT}.log" || true

    echo "  Report saved to: ${NCU_OUTPUT}.ncu-rep"
    echo ""
done

# Collect specific metrics for the matmul kernel (most performance-critical)
echo "--- Detailed matmul metrics: linear_kernel_tf32 ---"
NCU_MATMUL="${OUTDIR}/ncu_matmul_detailed_${TIMESTAMP}"

ncu \
    --kernel-name "linear_kernel_tf32" \
    --launch-count 5 \
    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_thread_inst_executed_op_hmma_pred_on.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,\
sm__inst_executed_pipe_tensor.sum,\
sm__inst_executed_pipe_fma.sum,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_blocks,\
launch__registers_per_thread,\
launch__shared_mem_per_block_allocated \
    --csv \
    --export "${NCU_MATMUL}" \
    python "${SCRIPT_DIR}/benchmark_student.py" "$FOLDER" --warmup 0 --runs 1 \
    2>&1 | tee "${NCU_MATMUL}.csv" || true

echo "  Detailed matmul report: ${NCU_MATMUL}.ncu-rep"


# =============================================================================
# 3. WARP OCCUPANCY ANALYSIS
# =============================================================================
# Occupancy metrics show how effectively the GPU's streaming multiprocessors
# are utilized. Key metrics:
#   - Theoretical occupancy: max warps per SM given register/shared mem usage
#   - Achieved occupancy: actual average warps active per cycle
#   - Register usage: registers per thread (limits occupancy if too high)
#   - Shared memory: per-block allocation (limits concurrent blocks)

echo ""
echo "=== Section 3: Warp Occupancy Analysis ==="
echo ""

NCU_OCCUPANCY="${OUTDIR}/ncu_occupancy_${TIMESTAMP}"

for KERNEL in "${KERNELS[@]}"; do
    echo "--- Occupancy for: $KERNEL ---"
    ncu \
        --kernel-name "$KERNEL" \
        --launch-count 1 \
        --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_blocks,\
launch__occupancy_limit_warps,\
launch__registers_per_thread,\
launch__shared_mem_per_block_allocated,\
launch__shared_mem_per_block_driver,\
launch__block_size,\
launch__grid_size,\
launch__waves_per_multiprocessor,\
sm__maximum_warps_per_active_cycle_pct \
        --csv \
        python "${SCRIPT_DIR}/benchmark_student.py" "$FOLDER" --warmup 0 --runs 1 \
        2>&1 | tee "${OUTDIR}/occupancy_${KERNEL}_${TIMESTAMP}.csv" || true
    echo ""
done


# =============================================================================
# 4. CLOCK CYCLE AND MEMORY LATENCY ANALYSIS
# =============================================================================
# Measures execution efficiency at the instruction level:
#   - Kernel execution duration in GPU clock cycles
#   - Memory access latency (L1, L2, HBM)
#   - Tensor core vs CUDA core instruction counts

echo ""
echo "=== Section 4: Clock Cycle & Memory Latency Analysis ==="
echo ""

for KERNEL in "linear_kernel_tf32" "flash_attention_kernel" "swiglu_fused_kernel"; do
    echo "--- Clock cycles & memory for: $KERNEL ---"
    ncu \
        --kernel-name "$KERNEL" \
        --launch-count 3 \
        --metrics \
gpu__time_duration.avg,\
gpu__time_duration.sum,\
sm__cycles_active.avg,\
sm__cycles_elapsed.avg,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__bytes.sum.per_second,\
sm__inst_executed_pipe_tensor.sum,\
sm__inst_executed_pipe_fma.sum,\
sm__inst_executed_pipe_alu.sum,\
sm__sass_thread_inst_executed_op_hmma_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
        --csv \
        python "${SCRIPT_DIR}/benchmark_student.py" "$FOLDER" --warmup 0 --runs 1 \
        2>&1 | tee "${OUTDIR}/cycles_${KERNEL}_${TIMESTAMP}.csv" || true
    echo ""
done

# Compute tensor core utilization ratio
echo "--- Tensor Core vs CUDA Core Utilization ---"
echo "From the CSV output above, compute:"
echo "  tensor_ratio = sm__inst_executed_pipe_tensor.sum / (sm__inst_executed_pipe_tensor.sum + sm__inst_executed_pipe_fma.sum)"
echo "  A ratio > 0.5 indicates good tensor core utilization"
echo ""


# =============================================================================
# 5. TF32 vs FP32 COMPARISON
# =============================================================================
# TF32 uses 19-bit precision (10-bit mantissa) on A100 tensor cores, providing
# up to 8x throughput vs FP32 CUDA cores. To measure impact:
#
# Method 1: Compare with allow_tf32 enabled vs disabled
# Method 2: Compare tensor core instruction counts

echo ""
echo "=== Section 5: TF32 vs FP32 Comparison ==="
echo ""

echo "Profiling with TF32 enabled (default)..."
NCU_TF32="${OUTDIR}/ncu_tf32_enabled_${TIMESTAMP}"
ncu \
    --kernel-name "linear_kernel_tf32" \
    --launch-count 3 \
    --metrics \
gpu__time_duration.avg,\
sm__inst_executed_pipe_tensor.sum,\
sm__inst_executed_pipe_fma.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_hmma_pred_on.sum \
    --csv \
    python "${SCRIPT_DIR}/benchmark_student.py" "$FOLDER" --warmup 0 --runs 1 \
    2>&1 | tee "${NCU_TF32}.csv" || true

echo ""
echo "To compare with FP32 (TF32 disabled), run:"
echo "  CUDA_ALLOW_TF32=0 ncu --kernel-name 'linear_kernel_tf32' --launch-count 3 \\"
echo "    --metrics gpu__time_duration.avg,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_fma.sum \\"
echo "    --csv python benchmark_student.py $FOLDER --warmup 0 --runs 1"
echo ""
echo "Expected: With TF32 enabled, tensor pipe instructions should dominate."
echo "          With TF32 disabled, FMA pipe instructions dominate, and kernel is slower."


# =============================================================================
# 6. OUTPUT PARSING -- Extract Key Metrics from NCU CSV Output
# =============================================================================
# These commands extract specific metrics from the CSV output files generated
# above. Useful for building comparison tables in the report.

echo ""
echo "=== Section 6: Output Parsing Examples ==="
echo ""

echo "--- Extract kernel execution times ---"
for f in "${OUTDIR}"/ncu_*.csv; do
    if [ -f "$f" ]; then
        echo "File: $(basename "$f")"
        grep -i "gpu__time_duration" "$f" 2>/dev/null | head -5 || true
        echo ""
    fi
done

echo "--- Extract SM and DRAM throughput ---"
for f in "${OUTDIR}"/ncu_*.csv; do
    if [ -f "$f" ]; then
        echo "File: $(basename "$f")"
        grep -iE "(sm__throughput|dram__throughput)" "$f" 2>/dev/null | head -5 || true
        echo ""
    fi
done

echo "--- Extract occupancy metrics ---"
for f in "${OUTDIR}"/occupancy_*.csv; do
    if [ -f "$f" ]; then
        echo "File: $(basename "$f")"
        grep -iE "(warps_active|occupancy_limit|registers_per_thread)" "$f" 2>/dev/null | head -10 || true
        echo ""
    fi
done

echo "--- Extract tensor core ratio ---"
echo "Use the following awk command on any NCU CSV to compute tensor core ratio:"
echo ""
echo "  awk -F',' '/pipe_tensor/{t=\$NF} /pipe_fma/{f=\$NF} END{if(t+f>0) printf \"Tensor core ratio: %.2f%%\\n\", t/(t+f)*100}' file.csv"
echo ""

echo "--- Quick summary table ---"
echo "To build a summary table of kernel times across all profiled kernels:"
echo ""
echo "  echo 'Kernel,Time(us)' > summary.csv"
echo "  for f in ${OUTDIR}/cycles_*.csv; do"
echo "    kernel=\$(basename \$f | sed 's/cycles_//;s/_${TIMESTAMP}.csv//')"
echo "    time=\$(grep 'gpu__time_duration.avg' \$f | tail -1 | awk -F',' '{print \$NF}')"
echo "    echo \"\$kernel,\$time\" >> summary.csv"
echo "  done"
echo "  cat summary.csv | column -t -s','"


# =============================================================================
# 7. PYTHON TIMING (torch.cuda.Event)
# =============================================================================
# For fine-grained kernel timing without Nsight overhead, use the dedicated
# Python timing script. This measures wall-clock kernel execution times using
# CUDA events, which have microsecond precision.

echo ""
echo "=== Section 7: Python CUDA Event Timing ==="
echo ""
echo "Running Python timing script..."

python "${SCRIPT_DIR}/profile_timing.py" "$FOLDER" --warmup 2 --runs 5 \
    2>&1 | tee "${OUTDIR}/timing_${FOLDER}_${TIMESTAMP}.log" || true


# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Profiling Complete"
echo "============================================================"
echo ""
echo "Results saved in: $OUTDIR"
echo ""
echo "Files generated:"
ls -la "$OUTDIR"/*"${TIMESTAMP}"* 2>/dev/null || echo "  (check directory)"
echo ""
echo "To view Nsight Systems timeline:"
echo "  nsys-ui ${NSYS_OUTPUT}.nsys-rep"
echo ""
echo "To view Nsight Compute reports:"
echo "  ncu-ui ${OUTDIR}/ncu_*.ncu-rep"
echo ""
