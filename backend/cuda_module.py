"""
cuda_module.py
──────────────
CUDA kernel for parallel classification accuracy computation.
Uses numba.cuda — compatible with Google Colab T4 and local NVIDIA GPUs.

Each GPU thread checks if prediction[i] == true_label[i].
Results are reduced on host to compute final accuracy.

Kernel config: 256 threads per block, ceil(n/256) blocks
"""

import numpy as np
import time

# ── Try importing numba CUDA ───────────────────────────────────────────────────
try:
    from numba import cuda
    _CUDA_AVAILABLE = True
    # Quick device check
    try:
        cuda.select_device(0)
        _GPU_AVAILABLE = True
    except Exception:
        _GPU_AVAILABLE = False
except ImportError:
    _CUDA_AVAILABLE = False
    _GPU_AVAILABLE  = False


# ── CUDA Kernel definition ─────────────────────────────────────────────────────
if _CUDA_AVAILABLE:
    @cuda.jit
    def _accuracy_kernel(y_true, y_pred, correct):
        """
        CUDA kernel — 1 thread per sample.
        correct[idx] = 1 if y_pred[idx] == y_true[idx] else 0
        """
        idx = cuda.grid(1)
        if idx < y_true.size:
            correct[idx] = 1 if y_true[idx] == y_pred[idx] else 0


def gpu_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy using CUDA kernel.
    Falls back to CPU if GPU not available.

    Parameters
    ----------
    y_true : int32 array of true labels
    y_pred : int32 array of predicted labels

    Returns
    -------
    accuracy : float in [0, 1]
    """
    if not _GPU_AVAILABLE:
        # CPU fallback
        return float(np.mean(y_true == y_pred))

    n       = len(y_true)
    correct = np.zeros(n, dtype=np.int32)

    # Transfer to GPU
    d_y_true  = cuda.to_device(y_true.astype(np.int32))
    d_y_pred  = cuda.to_device(y_pred.astype(np.int32))
    d_correct = cuda.to_device(correct)

    # Launch: 256 threads/block
    threads = 256
    blocks  = (n + threads - 1) // threads
    _accuracy_kernel[blocks, threads](d_y_true, d_y_pred, d_correct)

    # Copy back and reduce
    correct = d_correct.copy_to_host()
    return float(correct.sum()) / n


def benchmark_cpu_vs_gpu(y_true: np.ndarray, y_pred: np.ndarray,
                          n_repeats: int = 100):
    """
    Benchmark CPU vs GPU accuracy computation.
    Returns dict with times and speedup.
    """
    # CPU
    start = time.perf_counter()
    for _ in range(n_repeats):
        cpu_val = float(np.mean(y_true == y_pred))
    cpu_time = (time.perf_counter() - start) / n_repeats

    # GPU
    if _GPU_AVAILABLE:
        # Warm-up
        _ = gpu_accuracy(y_true[:100], y_pred[:100])
        start = time.perf_counter()
        for _ in range(n_repeats):
            gpu_val = gpu_accuracy(y_true, y_pred)
        gpu_time = (time.perf_counter() - start) / n_repeats
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
    else:
        gpu_time = cpu_time
        gpu_val  = cpu_val
        speedup  = 1.0

    return {
        'cpu_accuracy'  : cpu_val,
        'gpu_accuracy'  : gpu_val,
        'cpu_time_us'   : cpu_time * 1e6,
        'gpu_time_us'   : gpu_time * 1e6,
        'speedup'       : speedup,
        'gpu_available' : _GPU_AVAILABLE,
    }


def is_gpu_available() -> bool:
    return _GPU_AVAILABLE
