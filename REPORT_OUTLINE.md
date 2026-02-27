# Report Outline: Optimizing Triton GPU Kernels for GLM-ASR Inference

## 1. Introduction

### 1.1 Problem Statement
- GLM-ASR is a speech-to-text model comprising an Audio Encoder (32 layers), Projector, and Text Decoder (28 layers)
- The assignment requires implementing core GPU kernels in Triton and optimizing them for performance
- The baseline (`glm_asr_triton_example`) provides a working reference but leaves significant performance on the table
- Goal: maximize inference throughput while maintaining 100% transcription accuracy

### 1.2 Model Architecture Overview
- Audio Encoder: 32 layers, hidden=1280, 20 heads (head_dim=64), intermediate=5120, LayerNorm + GELU, partial RoPE (50%)
- Projector: pools 4 audio frames (5120 -> 4096 -> 3584), GELU activation
- Text Decoder: 28 layers, hidden=3584, 28 Q heads / 4 KV heads (GQA, head_dim=128), intermediate=18944, RMSNorm + SwiGLU, full RoPE (base=500000)
- Data flow: Audio (WAV) -> Mel Spectrogram -> Conv Subsampler -> Audio Encoder -> Projector -> Text Decoder -> Text

### 1.3 Scope and Contributions
- Enumerate the 7 optimizations applied (brief summary list)
- State the hardware target: NVIDIA A100 (Ampere), DICE cluster
- Note correctness is preserved: 100% transcription accuracy on benchmark

---

## 2. Background: GPU Execution Model and Triton

### 2.1 CUDA Execution Hierarchy
- Thread -> Warp (32 threads) -> Thread Block -> Grid
- Warp scheduling: hardware schedules warps on SMs; latency hiding via warp-level parallelism
- Occupancy: ratio of active warps to maximum warps per SM; higher occupancy hides memory latency
- Shared memory and register pressure as occupancy limiters

### 2.2 Tensor Cores and Precision Formats
- FP32 (standard): single-precision, no tensor core acceleration on matmul
- TF32 (Tensor Float 32): 19-bit mantissa input, FP32 accumulator; ~8x throughput over FP32 on A100 tensor cores
- How Triton exposes TF32: `allow_tf32=True` parameter on `tl.dot()`
- Trade-off: TF32 truncates mantissa from 23 bits to 10 bits for inputs, but accumulates in full FP32 -- negligible accuracy impact for inference

### 2.3 Triton Programming Model
- `@triton.jit` kernels: Python-like syntax compiled to PTX
- Grid/block abstraction: `tl.program_id()`, tile-based programming
- Key primitives: `tl.load`, `tl.store`, `tl.dot`, `tl.sum`, `tl.max`
- `@triton.autotune`: compile-time benchmarking of multiple kernel configurations
- Triton compiler handles register allocation, shared memory management, instruction scheduling

### 2.4 Memory Hierarchy and Latency
- Global memory (HBM): high bandwidth (~2 TB/s on A100) but high latency (~400 cycles)
- Shared memory (SRAM): low latency (~20 cycles) but limited capacity (48-164 KB per SM)
- Software pipelining (num_stages): overlaps memory loads with computation; double/triple buffering to hide HBM latency

---

## 3. Methodology

### 3.1 Experimental Setup
- Hardware: NVIDIA A100 GPU on DICE cluster
- Software: Triton (version), PyTorch (version), CUDA (version)
- Benchmark: `benchmark.sh` (end-to-end latency, 3 runs after warmup), `benchmark_detailed.sh` (per-operator profiling)
- Correctness criterion: 100% word accuracy on reference transcription ("Concord returned to its place amidst the tents.")

### 3.2 Baseline Characterization
- `glm_asr_triton_example`: working reference implementation
  - All `tl.dot()` calls use default FP32 (no `allow_tf32`)
  - No `@triton.autotune` -- single fixed tile configuration
  - `Linear.BACKEND = "torch"` -- all matmuls dispatched to cuBLAS, Triton kernels never invoked
  - No kernel fusion: MLP executes gate_proj, silu, up_proj, multiply, down_proj as separate operations
  - Attention: 3-kernel path (scores + softmax + output) materializes full attention matrix
- Identify bottleneck operations using `benchmark_detailed.sh`: Linear (matmul) dominates wall-clock time

### 3.3 Optimization Strategy
- Approach: iterative, one optimization at a time, measure impact, verify correctness after each change
- Parameter sweeps using `sweep_params.sh` and `sweep_all.sh` to explore tile configurations
- Profiling-driven: focus on the highest-cost operators first (Linear/matmul)

---

## 4. Optimizations Implemented

### 4.1 Optimization 1: TF32 Tensor Core Utilization
- **What**: Added `allow_tf32=True` to all 5 `tl.dot()` calls across `layers.py` (linear_kernel_tf32, linear_gelu_kernel, swiglu_fused_kernel)
- **Why**: A100 tensor cores deliver ~8x peak throughput for TF32 matmul vs FP32 CUDA cores (156 TFLOPS TF32 vs 19.5 TFLOPS FP32)
- **How TF32 works**: Input FP32 values have mantissa truncated to 10 bits (TF19 format), matrix multiply-accumulate performed by tensor cores, result accumulated in full FP32
- **Impact**: Dramatic speedup on all matmul-heavy operations (Linear, attention, MLP)
- **Correctness**: TF32 truncation is negligible for inference -- verified 100% accuracy maintained

### 4.2 Optimization 2: @triton.autotune for Kernel Configuration Search
- **What**: Added `@triton.autotune` decorators to 4 kernels with multiple configurations:
  - `linear_kernel_tf32`: 5 configs ranging from (BLOCK_M=32, BLOCK_N=32, BLOCK_K=16) to (BLOCK_M=128, BLOCK_N=128, BLOCK_K=64), with num_warps in {2, 4, 8} and num_stages in {2, 3, 4}
  - `linear_gelu_kernel`: 3 configs
  - `swiglu_fused_kernel`: 3 configs
  - `flash_attention_kernel`: 3 TILE_KV configs (32, 64, 128)
- **How autotune works**: Triton compiles all configs, benchmarks each on the actual problem size (keyed on `M`, `N`, `K`), and caches the fastest
- **Why multiple configs matter**: Optimal tile size depends on matrix dimensions -- small M (decode, M=1) favors small tiles for occupancy; large M (prefill) favors large tiles for compute density
- **Software pipelining**: `num_stages=3` or `4` enables triple/quadruple buffering -- while the current tile's data is being computed, the next tile's data is being loaded from HBM, hiding memory latency
- **Impact on warp scheduling**: Different `num_warps` settings affect SM occupancy; 2 warps for small tiles, 8 warps for large tiles

### 4.3 Optimization 3: Adaptive Linear Backend Dispatch
- **What**: Changed `Linear.BACKEND` from `"torch"` to `"adaptive"`
- **The problem**: In the baseline, `BACKEND = "torch"` meant ALL matmuls went through cuBLAS -- the Triton `linear_kernel_tf32` was never executed, making all TF32 and autotune work dead code
- **Adaptive dispatch logic**:
  - M < 32 (autoregressive decode, M=1): Use split-K matmul for better SM occupancy on thin matrices
  - M >= 32 (encoder prefill, projector): Use autotuned Triton matmul
  - CPU tensors: Fall back to torch/cuBLAS
- **Why split-K for small M**: When M=1, a standard tiled matmul only launches ceil(1/TILE_M) = 1 row of tiles -- poor GPU utilization. Split-K partitions the K dimension across multiple thread blocks, improving parallelism
- **Impact**: This was the single most critical change -- it activated all Triton kernel optimizations that were previously bypassed

### 4.4 Optimization 4: Tile Size Tuning and Parameter Sweeps
- **What**: Empirically determined optimal tile sizes on DICE cluster hardware using `sweep_params.sh` and `sweep_all.sh`
- **Sweep methodology**: Tested BLOCK_M in {32, 64, 128}, BLOCK_N in {32, 64, 128}, BLOCK_K in {16, 32, 64} across representative matrix sizes from the model
- **Results**: 32x32x16 found optimal as default for DICE cluster A100
  - Set as class-level defaults: `Linear.TILE_M/N/K`, `MLP.TILE_M/N/K`, `EncoderMLP.TILE_M/N/K`
  - Padding functions use these tile sizes to align matrix dimensions
- **Why 32x32x16 on this hardware**: Smaller tiles give higher occupancy (more blocks fit per SM), which matters for the variable matrix sizes in GLM-ASR (encoder hidden=1280, decoder hidden=3584, intermediate up to 18944)
- **Autotune still explores larger tiles**: The fixed defaults are fallback; autotune tests larger configs and selects per problem size

### 4.5 Optimization 5: Kernel Fusion
- **Fused SwiGLU kernel** (`swiglu_fused_kernel`):
  - Combines: gate_proj matmul + SiLU activation + up_proj matmul + elementwise multiply
  - Before: 4 separate kernel launches + 3 intermediate tensors written/read from HBM
  - After: 1 kernel launch, gate and up matmuls computed in parallel within the same tile loop, SiLU applied in-register, single output write
  - Saves: 3 kernel launch overheads (~5-10 us each) + eliminates intermediate HBM traffic for gate/up results
  - Used in Text Decoder MLP (28 layers x autoregressive steps)

- **Fused Linear+GELU kernel** (`linear_gelu_kernel`):
  - Combines: fc1 matmul + GELU activation
  - Before: 2 kernel launches + intermediate tensor
  - After: 1 kernel, GELU applied to matmul result in-register before store
  - Used in Audio Encoder MLP (32 layers) and Projector

- **Benefits of fusion**:
  - Reduced kernel launch overhead (significant when model has 60+ layers)
  - Eliminated intermediate memory traffic (each intermediate is potentially GBs for large hidden sizes)
  - Better instruction-level parallelism within the fused kernel

### 4.6 Optimization 6: Fused RoPE Triton Kernel
- **What**: New `apply_rope_kernel` that performs the full rotary position embedding in a single Triton kernel on GPU
- **Before** (PyTorch ops in `_apply_rope_single`): multiple operations -- slicing x1/x2, broadcasting cos/sin, multiply, subtract, add, concatenate -- each launching separate CUDA kernels via PyTorch
- **After**: Single Triton kernel loads x1, x2, cos, sin; computes `x1*cos - x2*sin` and `x2*cos + x1*sin`; passes through unrotated dimensions; stores result
- **Impact**: Eliminates ~6-8 PyTorch kernel launches per RoPE application, executed for both Q and K at every attention layer (60 layers total)
- **Note on dispatch**: Used when `head_dim_padded <= MAX_ROPE_DIM` and tensor is on CUDA; falls back to PyTorch path otherwise

### 4.7 Optimization 7: FlashAttention-Style Fused Attention
- **What**: `flash_attention_kernel` implementing online softmax that streams through K/V in tiles
- **Algorithm**: For each query position, iterate over K/V in tiles of size TILE_KV:
  1. Compute Q @ K_tile^T (partial scores)
  2. Update running max (m_new = max(m_prev, max(scores)))
  3. Rescale previous accumulator by exp(m_prev - m_new)
  4. Compute exp(scores - m_new) for current tile
  5. Update running sum-of-exp and weighted output accumulator
  6. Final normalization: acc / l_total
- **Memory savings**: Never materializes the full (batch*heads, seq_q, seq_k) attention matrix
- **Dispatch logic**: Used when seq_k <= 1024 and no explicit attention mask (decode path); falls back to 3-kernel path for prefill with masks
- **Autotuned**: TILE_KV tested at {32, 64, 128} to find optimal streaming granularity

---

## 5. Results and Analysis

### 5.1 End-to-End Latency Comparison
- Table: Baseline (triton_example) vs Optimized (triton_template) -- average inference time over 3 runs
- Speedup factor
- Both achieve 100% transcription accuracy

### 5.2 Per-Operator Profiling Breakdown
- Table from `benchmark_detailed.sh` output for both baseline and optimized
- Identify which operators improved the most (Linear/matmul expected to show largest gains)
- Attention timing: flash path vs 3-kernel path
- MLP timing: fused vs unfused
- Normalization and activation timing (element-wise ops -- smaller impact)

### 5.3 Optimization Impact Analysis
- **TF32 effect**: isolate by comparing with/without `allow_tf32` on matmul kernels
- **Autotune effect**: compare fixed config vs autotuned selection
- **Adaptive dispatch effect**: compare `BACKEND="torch"` (cuBLAS only) vs `BACKEND="adaptive"` (Triton active)
- **Fusion effect**: compare `MLP.FUSED=True` vs `MLP.FUSED=False`, `EncoderMLP.FUSED=True` vs `False`
- **Flash attention effect**: compare flash path vs 3-kernel path on decode step

### 5.4 Hardware Utilization Analysis
- Tensor core utilization: TF32 matmul should show tensor core activity in Nsight Compute
- SM occupancy: how autotune tile sizes affect occupancy (smaller tiles = more blocks per SM)
- Memory bandwidth utilization: kernel fusion reduces total HBM traffic
- Kernel launch overhead: compare total number of kernel launches per inference step (fused vs unfused)

---

## 6. Comparison with cuTile Approach

### 6.1 cuTile Overview
- NVIDIA cuTile: tile-based GPU programming via CuPy
- Explicit TF32 via `ct.astype(x, ct.tfloat32)` + `ct.mma()` -- manual precision casting
- Explicit latency hints and occupancy hints in the programming model
- Different kernel launch mechanism (CuPy-based)

### 6.2 Triton vs cuTile: Programming Model
- Triton: implicit tile management, compiler handles shared memory and register allocation
- cuTile: explicit tile loads with shape/index, manual control over MMA operations
- Triton `@triton.autotune` vs cuTile manual configuration
- Triton `allow_tf32` (one flag) vs cuTile explicit `ct.astype` + `ct.mma` calls

### 6.3 Performance Comparison
- Compare `glm_asr_triton_template` (optimized) vs `glm_asr_cutile_example` (baseline cuTile)
- If cuTile optimized version is available, compare optimized-vs-optimized
- Discuss: Triton compiler optimizations vs cuTile explicit control -- which gives better results on A100?

### 6.4 Trade-offs
- Triton: easier to write, portable across GPU architectures, compiler does heavy lifting
- cuTile: more explicit hardware control, potentially better for Blackwell-native features
- For this assignment: Triton autotune + TF32 achieves strong performance with less manual tuning

---

## 7. Discussion

### 7.1 Key Insights
- The most impactful optimization was **activating the Triton backend** (adaptive dispatch) -- without it, all other Triton optimizations were dead code
- TF32 provides a large "free" speedup on A100 with negligible accuracy impact
- Kernel fusion matters most in the decoder (28 layers x many autoregressive steps) where launch overhead accumulates
- FlashAttention's memory savings become important for longer sequences but the latency benefit on short sequences (decode M=1) is primarily from reduced kernel launches
- Autotune is valuable because the model has diverse matrix sizes (1280, 3584, 4096, 5120, 18944) -- no single tile config is universally optimal

### 7.2 Limitations
- Autotune incurs first-run compilation overhead (mitigated by Triton cache)
- Flash attention kernel limited to seq_k <= 1024 (falls back for longer encoder sequences)
- RoPE fusion only applies when head_dim fits in a single tile
- Parameter sweep was limited to DICE cluster A100 -- results may differ on other GPUs
- Autotuned configs are tuned for DICE cluster A100 -- different GPUs may prefer different configs

### 7.3 Relation to CUDA Core Concepts
- **Warp scheduling**: Autotune's `num_warps` parameter directly controls how many warps are launched per thread block; more warps help hide memory latency but increase register pressure
- **Tensor core utilization**: `allow_tf32=True` enables the 4th-gen tensor cores on A100; without it, matmul falls back to CUDA cores at 1/8 throughput
- **Kernel launch overhead**: Each CUDA kernel launch incurs ~5-10 microsecond overhead on the CPU side; fusion reduces the total count by 3-4x per layer
- **Software pipelining**: `num_stages > 1` tells Triton to emit cp.async instructions, overlapping the next tile's global memory load with the current tile's MMA computation
- **Memory coalescing**: Triton compiler ensures contiguous memory access patterns within tiles, critical for achieving peak HBM bandwidth

---

## 8. Conclusion

### 8.1 Summary of Results
- Enumerate all 7 optimizations and their individual/combined impact
- Overall speedup achieved vs baseline
- Correctness maintained (100% accuracy)

### 8.2 Lessons Learned
- Importance of profiling before optimizing (the torch backend bypass was invisible without investigation)
- Triton's abstraction level hits a sweet spot: high-level enough for productivity, low-level enough for meaningful optimization
- Autotune + TF32 as a "minimum viable optimization" that delivers substantial gains with minimal code changes

---

## References

- Triton Documentation: https://triton-lang.org/
- Triton Matrix Multiplication Tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- FlashAttention-2: Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691
- Attention Is All You Need: Vaswani, A. et al. (2017). arXiv:1706.03762
- RoFormer (RoPE): Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864
- NVIDIA A100 Tensor Core GPU Architecture: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
- NVIDIA TF32 Precision: https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/

---

## Appendix

### A. Autotune Configuration Details
- Full list of autotune configs for each kernel (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
- Which config was selected by autotune for each matrix size in the model

### B. Parameter Sweep Results
- Table of (TILE_M, TILE_N, TILE_K) -> inference latency from sweep_params.sh
- Visualization: heatmap of tile configurations vs latency

### C. Code Excerpts
- Key code snippets showing: TF32 enable, autotune decorator, adaptive dispatch logic, fused SwiGLU kernel, flash attention online softmax loop

### D. Nsight Profiling Data
- Nsight Compute metrics for key kernels: SM occupancy, tensor core utilization, memory throughput
- Commands used: `./benchmark_detailed.sh glm_asr_triton_template --nsys`
