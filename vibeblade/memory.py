"""VibeBlade Unified Memory Pool - Spoofs System RAM as Virtual VRAM

Extended with activation buffer allocation for MoE hot/cold split execution.
Pre-allocated mlock'd buffers prevent page faults when activations cross
PCIe between CPU and GPU during MoE expert dispatch.
"""

from __future__ import annotations

import mmap
import os
import ctypes
import platform
import numpy as np


class UnifiedMemoryPool:
    """The 'Cheat' Layer: Spoofs System RAM as Virtual VRAM."""
    
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.fd = os.open(model_path, os.O_RDONLY)
        self.size = os.path.getsize(model_path)
        
        # Allocate pinned, non-swappable memory
        self.buffer = mmap.mmap(
            self.fd, 
            self.size, 
            access=mmap.ACCESS_READ
        )
        self._lock_memory()

    def _lock_memory(self):
        """Prevents the OS from swapping these weights to disk."""
        addr = ctypes.c_void_p.from_buffer(self.buffer)
        if platform.system() != "Windows":
            try:
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.mlock(addr, ctypes.c_size_t(self.size))
            except OSError as e:
                print(f"Warning: mlock failed: {e}")
        else:
            try:
                ctypes.windll.kernel32.VirtualLock(addr, ctypes.c_size_t(self.size))
            except OSError as e:
                print(f"Warning: VirtualLock failed: {e}")

    def get_addr(self) -> int:
        """Return the memory address of the pinned buffer."""
        return ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))

    def read(self, offset: int, size: int) -> bytes:
        """Read bytes from the unified memory pool."""
        # FIX #5: Validate bounds to prevent out-of-range reads
        if offset < 0 or size < 0:
            raise ValueError(f"Invalid read parameters: offset={offset}, size={size}")
        if offset + size > self.size:
            raise ValueError(
                f"Read out of bounds: offset={offset} + size={size} > pool size={self.size}"
            )
        self.buffer.seek(offset)
        return self.buffer.read(size)

    def close(self):
        """Release memory resources."""
        if hasattr(self, 'buffer'):
            self.buffer.close()
        if hasattr(self, 'fd'):
            os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ActivationBufferPool:
    """Pre-allocated pinned memory buffers for MoE activation exchange.

    During hot/cold MoE execution, tiny activation tensors (~KB) cross
    PCIe between CPU and GPU. By pre-allocating and mlock'ing these buffers,
    we prevent page faults that would stall the decode loop.

    Usage:
        pool = ActivationBufferPool(count=4, buffer_size=16384)  # 4 × 16KB
        buf = pool.acquire()     # get a pinned numpy array
        # ... use buf for CPU↔GPU activation transfer ...
        pool.release(buf)        # return to pool
    """

    def __init__(
        self,
        count: int = 4,
        buffer_size: int = 16384,
        dtype: np.dtype = np.float32,
    ):
        """Allocate a pool of pinned buffers.

        Args:
            count: number of buffers in the pool
            buffer_size: size of each buffer in bytes
            dtype: numpy dtype for the buffers
        """
        self._count = count
        self._buffer_size = buffer_size
        self._dtype = np.dtype(dtype)
        self._n_elements = buffer_size // self._dtype.itemsize

        self._available: list[np.ndarray] = []
        self._all_buffers: list[np.ndarray] = []

        for _ in range(count):
            buf = np.zeros(self._n_elements, dtype=self._dtype)
            self._pin_buffer(buf)
            self._all_buffers.append(buf)
            self._available.append(buf)

    def _pin_buffer(self, buf: np.ndarray) -> None:
        """Pin a numpy buffer in physical RAM (prevent swapping)."""
        if not buf.flags["C_CONTIGUOUS"] and not buf.flags["F_CONTIGUOUS"]:
            return  # can't pin non-contiguous arrays
        try:
            addr = ctypes.c_void_p(buf.ctypes.data)
            size = ctypes.c_size_t(buf.nbytes)
            if platform.system() != "Windows":
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.mlock(addr, size)
            else:
                ctypes.windll.kernel32.VirtualLock(addr, size)
        except OSError:
            pass  # non-fatal — just won't be pinned

    @property
    def buffer_shape(self) -> tuple[int, ...]:
        """Shape of each buffer."""
        return (self._n_elements,)

    @property
    def buffer_bytes(self) -> int:
        """Size of each buffer in bytes."""
        return self._buffer_size

    @property
    def available_count(self) -> int:
        """Number of buffers currently available in the pool."""
        return len(self._available)

    @property
    def total_count(self) -> int:
        """Total number of buffers in the pool."""
        return self._count

    def acquire(self) -> np.ndarray | None:
        """Get a pinned buffer from the pool.

        Returns None if all buffers are in use (caller must wait or skip).
        """
        if self._available:
            return self._available.pop()
        return None

    def release(self, buf: np.ndarray) -> None:
        """Return a buffer to the pool."""
        # Zero it out for cleanliness (optional but prevents data leaks)
        buf[:] = 0
        self._available.append(buf)

    def allocate_buffer(self, size_bytes: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate a single new pinned buffer (not pooled).

        Args:
            size_bytes: size in bytes
            dtype: numpy dtype

        Returns:
            Pinned numpy array
        """
        dt = np.dtype(dtype)
        n = size_bytes // dt.itemsize
        buf = np.zeros(n, dtype=dt)
        self._pin_buffer(buf)
        return buf

    def stats(self) -> dict:
        """Pool utilization stats."""
        return {
            "total_buffers": self._count,
            "available": self.available_count,
            "in_use": self._count - self.available_count,
            "buffer_bytes": self._buffer_size,
            "total_bytes": self._count * self._buffer_size,
            "dtype": str(self._dtype),
        }

    def close(self) -> None:
        """Release all buffers."""
        self._available.clear()
        self._all_buffers.clear()

    def __repr__(self) -> str:
        s = self.stats()
        return (f"ActivationBufferPool(buffers={s['available']}/{s['total_buffers']}, "
                f"{s['buffer_bytes']}B each, {s['dtype']})")
