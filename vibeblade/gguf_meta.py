"""
Lightweight GGUF metadata reader for auto-tuning.

Only reads the few header KV pairs needed (arch, n_embd, n_layer, n_head).
Stops as soon as all keys are found — skips tokenizer arrays entirely.
"""
import struct
import os

GGUF_MAGIC = 0x46554747  # 'GGUF' as little-endian uint32

# GGUF value types
_T_UINT32 = 5
_T_INT32 = 6
_T_FLOAT32 = 7
_T_STRING = 9
_T_UINT64 = 10
_T_ARRAY = 15

# Keys we care about
_TARGET_KEYS = {
    "general.architecture",
    "llama.embedding_length",
    "llama.block_count",
    "llama.attention.head_count",
}

# Architecture-specific key prefixes (llama is most common)
_ARCH_PREFIXES = ["llama.", "qwen2.", "phi", "gemma.", "mistral.", "falcon.", "starcoder2."]


def _read_le_u64(f):
    return struct.unpack("<Q", f.read(8))[0]

def _read_le_u32(f):
    return struct.unpack("<I", f.read(4))[0]

def _read_le_i32(f):
    return struct.unpack("<i", f.read(4))[0]

def _read_le_f32(f):
    return struct.unpack("<f", f.read(4))[0]


def _skip_string(f):
    length = _read_le_u64(f)
    if length > 0:
        f.seek(length, 1)  # seek past string data


def _skip_value(f, vtype):
    """Skip a value of given type without reading it."""
    if vtype in (_T_UINT32, _T_INT32, _T_FLOAT32):
        f.seek(4, 1)
    elif vtype == _T_UINT64:
        f.seek(8, 1)
    elif vtype == _T_STRING:
        _skip_string(f)
    elif vtype == _T_ARRAY:
        elem_type = _read_le_u32(f)
        length = _read_le_u64(f)
        # Skip all elements
        for _ in range(length):
            _skip_value(f, elem_type)
    # Other types: uint8, int8, uint16, int16, bool — 1-2 bytes
    elif vtype in (1, 2, 8):  # uint8, int8, bool
        f.seek(1, 1)
    elif vtype in (3, 4):  # uint16, int16
        f.seek(2, 1)
    else:
        # Unknown type — bail out
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def _read_value_fast(f, vtype):
    """Read a simple scalar value. Returns None for arrays."""
    if vtype == _T_UINT32:
        return _read_le_u32(f)
    elif vtype == _T_INT32:
        return _read_le_i32(f)
    elif vtype == _T_FLOAT32:
        return _read_le_f32(f)
    elif vtype == _T_STRING:
        length = _read_le_u64(f)
        if length > 1024:  # only read short strings
            f.seek(length, 1)
            return None
        return f.read(length).decode("utf-8", errors="replace")
    elif vtype == _T_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    else:
        _skip_value(f, vtype)
        return None


def read_model_meta(path: str) -> dict:
    """Read architecture metadata from GGUF header.

    Stops reading once all target keys are found.
    Returns dict with: arch, n_embd, n_layer, n_head.
    """
    found = {}
    remaining = set(_TARGET_KEYS)

    with open(path, "rb") as f:
        magic = _read_le_u32(f)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")

        _read_le_u32(f)  # version
        n_tensors = _read_le_u64(f)
        n_kv = _read_le_u64(f)

        for _ in range(n_kv):
            # Read key string
            klen = _read_le_u64(f)
            if klen > 1024:
                f.seek(klen, 1)
                key = ""
            else:
                key = f.read(klen).decode("utf-8", errors="replace")

            vtype = _read_le_u32(f)

            if key in remaining:
                val = _read_value_fast(f, vtype)
                if val is not None:
                    found[key] = val
                    remaining.discard(key)
                    if not remaining:
                        break
            else:
                _skip_value(f, vtype)

    return found


def get_model_arch_params(path: str) -> dict:
    """Extract architecture metadata for auto-tuning.

    Returns dict with: arch, n_embd, n_layer, n_head, est_params_B
    """
    # Resolve symlinks
    path = os.path.realpath(path)
    meta = read_model_meta(path)

    arch = meta.get("general.architecture", "llama")
    n_embd = meta.get("llama.embedding_length", 0)
    n_layer = meta.get("llama.block_count", 0)
    n_head = meta.get("llama.attention.head_count", 0)

    if n_embd > 0 and n_layer > 0:
        est = (n_embd * n_embd * n_layer * 12) / 1e9
    else:
        # Very rough: file size / 0.6 bytes per param for Q4_K_M
        est = os.path.getsize(path) / 0.6 / 1e9

    return {
        "arch": arch,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "est_params_B": round(est, 2),
    }
