#!/usr/bin/env python3
"""
CUDA Event Timing for GLM-ASR Triton Kernels

Measures per-layer and per-operation kernel execution times using
torch.cuda.Event for microsecond-precision GPU timing.

Usage:
    python profile_timing.py <folder_name> [--warmup N] [--runs N]
    python profile_timing.py glm_asr_triton_template-Used --warmup 2 --runs 5
"""

import argparse
import sys
import os
import time
import importlib
import numpy as np

import torch


class KernelTimer:
    """GPU kernel timer using CUDA events."""

    def __init__(self):
        self.records = {}

    def start(self, name):
        if name not in self.records:
            self.records[name] = {"times": [], "start": None, "end": None}
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.records[name]["start"] = start_event
        self.records[name]["end"] = end_event
        start_event.record()

    def stop(self, name):
        if name in self.records and self.records[name]["start"] is not None:
            self.records[name]["end"].record()

    def sync_and_collect(self):
        """Synchronize all events and collect times."""
        torch.cuda.synchronize()
        for name, rec in self.records.items():
            if rec["start"] is not None and rec["end"] is not None:
                try:
                    elapsed = rec["start"].elapsed_time(rec["end"])
                    rec["times"].append(elapsed)
                except RuntimeError:
                    pass
            rec["start"] = None
            rec["end"] = None

    def summary(self):
        """Print timing summary."""
        print("\n" + "=" * 70)
        print("  Per-Operation Timing Summary (ms)")
        print("=" * 70)
        print(f"{'Operation':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
        print("-" * 70)

        total = 0
        for name in sorted(self.records.keys()):
            times = self.records[name]["times"]
            if not times:
                continue
            arr = np.array(times)
            mean = arr.mean()
            total += mean
            print(f"{name:<35} {mean:>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f} {len(arr):>6}")

        print("-" * 70)
        print(f"{'Total (sum of means)':<35} {total:>8.3f}")
        print("=" * 70)
        return self.records


def patch_linear_for_timing(layers_module, timer):
    """Monkey-patch Linear.__call__ to time each invocation."""
    original_call = layers_module.Linear.__call__

    call_counter = [0]

    def timed_call(self, x):
        call_counter[0] += 1
        name = f"Linear({self.in_features}x{self.out_features})"
        timer.start(name)
        result = original_call(self, x)
        timer.stop(name)
        return result

    layers_module.Linear.__call__ = timed_call
    return original_call


def patch_mlp_for_timing(layers_module, timer):
    """Monkey-patch MLP.__call__ to time."""
    original_call = layers_module.MLP.__call__

    def timed_call(self, x):
        timer.start("MLP_SwiGLU")
        result = original_call(self, x)
        timer.stop("MLP_SwiGLU")
        return result

    layers_module.MLP.__call__ = timed_call
    return original_call


def patch_encoder_mlp_for_timing(layers_module, timer):
    """Monkey-patch EncoderMLP.__call__ to time."""
    if not hasattr(layers_module, 'EncoderMLP'):
        return None

    original_call = layers_module.EncoderMLP.__call__

    def timed_call(self, x):
        timer.start("EncoderMLP_GELU")
        result = original_call(self, x)
        timer.stop("EncoderMLP_GELU")
        return result

    layers_module.EncoderMLP.__call__ = timed_call
    return original_call


def patch_attention_for_timing(attention_module, timer):
    """Monkey-patch attention dispatch for timing."""
    if not hasattr(attention_module, 'triton_attention'):
        return None

    original_fn = attention_module.triton_attention

    def timed_fn(*args, **kwargs):
        timer.start("Attention")
        result = original_fn(*args, **kwargs)
        timer.stop("Attention")
        return result

    attention_module.triton_attention = timed_fn
    return original_fn


def patch_rope_for_timing(rope_module, timer):
    """Monkey-patch RoPE apply for timing."""
    if not hasattr(rope_module, 'apply_rotary_pos_emb'):
        return None

    original_fn = rope_module.apply_rotary_pos_emb

    def timed_fn(*args, **kwargs):
        timer.start("RoPE")
        result = original_fn(*args, **kwargs)
        timer.stop("RoPE")
        return result

    rope_module.apply_rotary_pos_emb = timed_fn
    return original_fn


def patch_norms_for_timing(layers_module, timer):
    """Monkey-patch RMSNorm and LayerNorm for timing."""
    originals = {}

    if hasattr(layers_module, 'RMSNorm'):
        orig = layers_module.RMSNorm.__call__

        def timed_rmsnorm(self, x):
            timer.start("RMSNorm")
            result = orig(self, x)
            timer.stop("RMSNorm")
            return result

        layers_module.RMSNorm.__call__ = timed_rmsnorm
        originals['RMSNorm'] = orig

    if hasattr(layers_module, 'LayerNorm'):
        orig = layers_module.LayerNorm.__call__

        def timed_layernorm(self, x):
            timer.start("LayerNorm")
            result = orig(self, x)
            timer.stop("LayerNorm")
            return result

        layers_module.LayerNorm.__call__ = timed_layernorm
        originals['LayerNorm'] = orig

    return originals


def benchmark_with_timing(folder_name, num_warmup=2, num_runs=5):
    """Run benchmark with instrumented timing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    sys.path.insert(0, folder_path)

    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv']:
            del sys.modules[mod_name]

    # Import modules
    layers = importlib.import_module("layers")
    try:
        attention = importlib.import_module("attention")
    except ImportError:
        attention = None
    try:
        rope = importlib.import_module("rope")
    except ImportError:
        rope = None

    # Create timer
    timer = KernelTimer()

    # Patch modules for timing
    patch_linear_for_timing(layers, timer)
    patch_mlp_for_timing(layers, timer)
    patch_encoder_mlp_for_timing(layers, timer)
    if attention:
        patch_attention_for_timing(attention, timer)
    if rope:
        patch_rope_for_timing(rope, timer)
    patch_norms_for_timing(layers, timer)

    # Load model
    print(f"Loading model from {folder_name}...")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test audio
    sys.path.insert(0, script_dir)
    from benchmark_student import load_test_audio, prepare_inputs_torch, decode_output
    audio_array, expected, duration = load_test_audio()
    input_features, input_ids, input_features_mask = prepare_inputs_torch(
        audio_array, processor, device
    )

    generate_fn = model.generate
    if hasattr(model, 'generate_v8b'):
        generate_fn = model.generate_v8b
    elif hasattr(model, 'generate_v8'):
        generate_fn = model.generate_v8
    elif hasattr(model, 'generate_v6'):
        generate_fn = model.generate_v6

    print(f"Using generate function: {generate_fn.__name__}")

    # Warmup (no timing)
    print(f"\nWarmup ({num_warmup} runs)...")
    for i in range(num_warmup):
        with torch.no_grad():
            try:
                _ = generate_fn(
                    input_features, input_ids=input_ids,
                    input_features_mask=input_features_mask,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
            except TypeError:
                _ = generate_fn(
                    input_features, input_ids=input_ids,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
        torch.cuda.synchronize()
    # Clear warmup records
    timer.records.clear()

    # Timed runs
    print(f"\nBenchmarking ({num_runs} runs with per-op timing)...")
    e2e_times = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            try:
                output = generate_fn(
                    input_features, input_ids=input_ids,
                    input_features_mask=input_features_mask,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )
            except TypeError:
                output = generate_fn(
                    input_features, input_ids=input_ids,
                    max_new_tokens=100, temperature=1.0, top_k=1
                )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        e2e_times.append(elapsed_ms)

        timer.sync_and_collect()

        tokens = output.shape[1] - input_ids.shape[1]
        print(f"  Run {i+1}: {elapsed_ms:.1f}ms ({tokens} tokens, {elapsed_ms/tokens:.2f} ms/token)")

    # Print results
    e2e_arr = np.array(e2e_times)
    print(f"\nEnd-to-end: {e2e_arr.mean():.1f}ms +/- {e2e_arr.std():.1f}ms")
    print(f"Tokens: {tokens}")
    print(f"Speed: {e2e_arr.mean()/tokens:.2f} ms/token")

    # Decode and check
    generated_np = output.detach().cpu().numpy()
    transcription = decode_output(generated_np, processor)
    print(f"\nTranscription: {transcription}")

    # Print per-operation timing
    timer.summary()

    # Clean up
    sys.path.remove(folder_path)


def main():
    parser = argparse.ArgumentParser(description="Profile GLM-ASR kernel timing")
    parser.add_argument("folder", type=str, help="Folder name (e.g., glm_asr_triton_template-Used)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    args = parser.parse_args()

    print("=" * 70)
    print("GLM-ASR Kernel Timing Profiler")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")

    benchmark_with_timing(args.folder, args.warmup, args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
