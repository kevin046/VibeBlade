"""VibeBlade Core — C++/Python bridge for AVX-512 kernels."""

import numpy as np

# Try C++ backend first, fall back to numpy
_CPP_BACKEND = False

def _init_backend():
    global _CPP_BACKEND, ts_rotor_unpack, ts_drelu_activation
    try:
        from vibeblade._vibeblade_core import ts_rotor_unpack as _cpp_rotor, ts_drelu_activation as _cpp_drelu
        ts_rotor_unpack = _cpp_rotor
        ts_drelu_activation = _cpp_drelu
        _CPP_BACKEND = True
    except ImportError:
        # Pure numpy fallback
        from vibeblade.quant import unpack_nibbles
        
        def _numpy_rotor_unpack(pinned_ram, output, rotor_matrix, n):
            unpacked = unpack_nibbles(np.frombuffer(pinned_ram, dtype=np.uint8), n)
            reconstructed = np.zeros(n, dtype=np.float32)
            group_size = 4
            for i in range(0, n, group_size):
                rotor = rotor_matrix[i:i+group_size, i:i+group_size] if rotor_matrix.ndim == 2 else rotor_matrix[:group_size].reshape(4, 4)
                g = min(group_size, n - i)
                reconstructed[i:i+g] = unpacked[i:i+g].astype(np.float32) * np.diag(rotor)[:g]
            np.copyto(output, reconstructed)
        
        def _numpy_drelu(input_arr, output, n):
            result = np.maximum(np.frombuffer(input_arr, dtype=np.float32)[:n], 0.0)
            np.copyto(output, result)
        
        ts_rotor_unpack = _numpy_rotor_unpack
        ts_drelu_activation = _numpy_drelu

_init_backend()

def rotor_unpack(weights: np.ndarray, rotor_matrix: np.ndarray) -> np.ndarray:
    """High-level rotor unpack: 4-bit weights -> float32 via SO(4) rotation.
    
    Args:
        weights: 4-bit packed weights (uint8 array)
        rotor_matrix: SO(4) rotation matrix or array of rotation matrices
    
    Returns:
        float32 reconstructed weights
    """
    n = len(weights) * 2  # each byte = 2 4-bit values
    output = np.zeros(n, dtype=np.float32)
    ts_rotor_unpack(weights, output, rotor_matrix, n)
    return output

def drelu(x: np.ndarray) -> np.ndarray:
    """High-level dReLU activation.
    
    Args:
        x: float32 input array
    
    Returns:
        activated values (x where x > 0, else 0)
    """
    n = len(x)
    output = np.zeros(n, dtype=np.float32)
    ts_drelu_activation(x, output, n)
    return output

__all__ = ["rotor_unpack", "drelu", "_CPP_BACKEND", "ts_rotor_unpack", "ts_drelu_activation"]
