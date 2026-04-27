"""Tests for vibeblade.loader — GGUF model loader."""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest

from vibeblade.loader import (
    GGUFLoader,
    GGUF_MAGIC,
    GGUF_TYPE_F32,
    load_model,
    estimate_model_size_gb,
)

# Metadata value type constants (used in KV pairs)
_VAL_STRING = 8
_VAL_UINT32 = 4
_VAL_UINT64 = 10
_VAL_FLOAT32 = 6
_VAL_INT32 = 5
_VAL_BOOL = 7
_VAL_ARRAY = 9


# ------------------------------------------------------------------
# Helper: create a minimal valid GGUF binary
# ------------------------------------------------------------------

def _write_kv_string(f, key: str, value: str) -> None:
    """Write a single KV pair where the value is a GGUF STRING."""
    key_bytes = key.encode("utf-8")
    val_bytes = value.encode("utf-8")
    f.write(struct.pack("<Q", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", _VAL_STRING))
    f.write(struct.pack("<Q", len(val_bytes)))
    f.write(val_bytes)


def _write_kv_uint32(f, key: str, value: int) -> None:
    """Write a single KV pair where the value is a GGUF UINT32."""
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<Q", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", _VAL_UINT32))
    f.write(struct.pack("<I", value))


def create_mock_gguf(
    tmpdir: str,
    kv_pairs: list | None = None,
    n_tensors: int = 0,
) -> str:
    """Create a minimal GGUF binary and return its path."""
    path = os.path.join(tmpdir, "model.gguf")

    if kv_pairs is None:
        kv_pairs = [
            ("general.architecture", "string", "llama"),
            ("llama.context_length", "uint32", 2048),
        ]

    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", 3))                 # version = 3
        f.write(struct.pack("<Q", n_tensors))         # n_tensors
        f.write(struct.pack("<Q", len(kv_pairs)))     # n_kv

        # KV pairs
        for key, vtype, value in kv_pairs:
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)

            if vtype == "string":
                f.write(struct.pack("<I", _VAL_STRING))
                val_bytes = value.encode("utf-8")
                f.write(struct.pack("<Q", len(val_bytes)))
                f.write(val_bytes)
            elif vtype == "uint32":
                f.write(struct.pack("<I", _VAL_UINT32))
                f.write(struct.pack("<I", value))
            elif vtype == "uint64":
                f.write(struct.pack("<I", _VAL_UINT64))
                f.write(struct.pack("<Q", value))
            elif vtype == "float32":
                f.write(struct.pack("<I", _VAL_FLOAT32))
                f.write(struct.pack("<f", value))
            elif vtype == "int32":
                f.write(struct.pack("<I", _VAL_INT32))
                f.write(struct.pack("<i", value))
            elif vtype == "bool":
                f.write(struct.pack("<I", _VAL_BOOL))
                f.write(struct.pack("?", value))
            elif vtype == "string_array":
                f.write(struct.pack("<I", _VAL_ARRAY))
                f.write(struct.pack("<I", _VAL_STRING))
                strings = value
                f.write(struct.pack("<Q", len(strings)))
                for s in strings:
                    sb = s.encode("utf-8")
                    f.write(struct.pack("<Q", len(sb)))
                    f.write(sb)
            else:
                raise ValueError(f"Unsupported mock kv type: {vtype}")

    return path


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestCreateMockGGUF:
    """Verify the helper itself produces readable files."""

    def test_create_mock_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 12

    def test_mock_file_has_correct_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            with open(path, "rb") as f:
                (magic,) = struct.unpack("<I", f.read(4))
            assert magic == GGUF_MAGIC


class TestLoaderOpensValidFile:

    def test_loader_opens_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            with GGUFLoader(path) as loader:
                assert loader.version == 3
                assert loader.n_tensors == 0
                assert loader.n_kv == 2


class TestLoaderRejectsInvalidMagic:

    def test_loader_rejects_invalid_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.gguf")
            with open(path, "wb") as f:
                f.write(b"NOTG")
            with pytest.raises(ValueError, match="Not a GGUF"):
                GGUFLoader(path)


class TestMetadataValues:

    def test_metadata_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            with GGUFLoader(path) as loader:
                meta = loader.metadata
                assert meta["general.architecture"] == "llama"
                assert meta["llama.context_length"] == 2048

    def test_multiple_value_types(self) -> None:
        kv_pairs = [
            ("general.architecture", "string", "llama"),
            ("llama.context_length", "uint32", 4096),
            ("llama.embedding_length", "uint32", 4096),
            ("llama.attention.head_count", "uint32", 32),
            ("llama.attention.layer_norm_rms_epsilon", "float32", 1e-5),
            ("general.file_type", "int32", 7),
            ("general.quantization_version", "uint64", 2),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir, kv_pairs=kv_pairs)
            with GGUFLoader(path) as loader:
                meta = loader.metadata
                assert meta["general.architecture"] == "llama"
                assert meta["llama.context_length"] == 4096
                assert meta["llama.embedding_length"] == 4096
                assert meta["llama.attention.head_count"] == 32
                assert abs(meta["llama.attention.layer_norm_rms_epsilon"] - 1e-5) < 1e-12
                assert meta["general.file_type"] == 7
                assert meta["general.quantization_version"] == 2

    def test_bool_metadata(self) -> None:
        kv_pairs = [("general.flag", "bool", True)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir, kv_pairs=kv_pairs)
            with GGUFLoader(path) as loader:
                assert loader.metadata["general.flag"] is True

    def test_string_array_metadata(self) -> None:
        kv_pairs = [
            ("general.architecture", "string", "llama"),
            ("tokenizer.ggml.tokens", "string_array", ["<unk>", "<s>", "</s>"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir, kv_pairs=kv_pairs)
            with GGUFLoader(path) as loader:
                assert loader.metadata["tokenizer.ggml.tokens"] == [
                    "<unk>", "<s>", "</s>"
                ]


class TestLoadModelConvenience:

    def test_load_model_convenience(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            result = load_model(path)
            assert "metadata" in result
            assert "tensors" in result
            assert "config" in result
            assert result["metadata"]["general.architecture"] == "llama"
            assert result["config"]["architecture"] == "llama"
            assert result["config"]["context_length"] == 2048
            assert isinstance(result["tensors"], dict)


class TestContextManager:

    def test_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            with GGUFLoader(path) as loader:
                _ = loader.metadata
            assert loader._f.closed is True or loader._mmap is not None

    def test_explicit_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir)
            loader = GGUFLoader(path)
            loader.close()
            assert loader._f.closed or loader._mmap is None


class TestTensorLoading:
    """Test reading tensor infos and actual tensor data from mock GGUF."""

    @staticmethod
    def _create_gguf_with_tensor(tmpdir: str) -> str:
        """Create a GGUF file with one FLOAT32 tensor of shape (2, 3)."""
        path = os.path.join(tmpdir, "tensor.gguf")
        tensor_data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", 3))          # version
            f.write(struct.pack("<Q", 1))           # n_tensors = 1
            f.write(struct.pack("<Q", 1))           # n_kv = 1

            # KV pair
            _write_kv_string(f, "general.architecture", "llama")

            # Tensor info
            name = b"test.weight"
            f.write(struct.pack("<Q", len(name)))
            f.write(name)
            f.write(struct.pack("<I", 2))           # n_dims = 2
            f.write(struct.pack("<Q", 3))           # dim[0] = 3 (columns)
            f.write(struct.pack("<Q", 2))           # dim[1] = 2 (rows)
            f.write(struct.pack("<I", GGUF_TYPE_F32))  # dtype = F32 (type 0)
            f.write(struct.pack("<Q", 0))           # offset

            f.write(tensor_data)

        return path

    def test_tensor_infos_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            with GGUFLoader(path) as loader:
                assert len(loader.tensor_infos) == 1
                info = loader.tensor_infos[0]
                assert info["name"] == "test.weight"
                assert info["shape"] == (2, 3)
                assert info["dtype"] == GGUF_TYPE_F32

    def test_load_single_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            with GGUFLoader(path) as loader:
                arr = loader.load_tensor("test.weight")
                assert arr.shape == (2, 3)
                assert arr.dtype == np.float32
                expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
                np.testing.assert_array_equal(arr, expected)

    def test_load_all_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            with GGUFLoader(path) as loader:
                tensors = loader.load_all_tensors()
                assert "test.weight" in tensors
                assert tensors["test.weight"].shape == (2, 3)

    def test_load_missing_tensor_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            with GGUFLoader(path) as loader:
                with pytest.raises(KeyError, match="not.found"):
                    loader.load_tensor("not.found")

    def test_load_model_with_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            result = load_model(path)
            assert "test.weight" in result["tensors"]
            assert result["tensors"]["test.weight"].shape == (2, 3)


class TestLoadModelConfigExtraction:

    def test_config_extracts_common_fields(self) -> None:
        kv_pairs = [
            ("general.architecture", "string", "llama"),
            ("llama.context_length", "uint32", 4096),
            ("llama.embedding_length", "uint32", 4096),
            ("llama.block_count", "uint32", 32),
            ("llama.feed_forward_length", "uint32", 11008),
            ("llama.attention.head_count", "uint32", 32),
            ("llama.attention.head_count_kv", "uint32", 8),
            ("llama.vocab_size", "uint32", 32000),
            ("llama.rope.freq_base", "float32", 10000.0),
            ("llama.attention.layer_norm_rms_epsilon", "float32", 1e-6),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir, kv_pairs=kv_pairs)
            result = load_model(path)
            config = result["config"]
            assert config["architecture"] == "llama"
            assert config["context_length"] == 4096
            assert config["embedding_length"] == 4096
            assert config["block_count"] == 32
            assert config["feed_forward_length"] == 11008
            assert config["attention.head_count"] == 32
            assert config["attention.head_count_kv"] == 8
            assert config["vocab_size"] == 32000
            assert config["rope.freq_base"] == 10000.0
            assert config["attention.layer_norm_rms_epsilon"] == pytest.approx(1e-6)

    def test_config_empty_without_arch(self) -> None:
        kv_pairs = [("some.key", "uint32", 42)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_mock_gguf(tmpdir, kv_pairs=kv_pairs)
            result = load_model(path)
            assert result["config"] == {}


class TestEstimateModelSize:

    def test_estimate_70b_q4(self):
        """70B params at Q4_0 ≈ 36.7 GiB (39.4 GB)."""
        gb = estimate_model_size_gb(70_000_000_000, 2)  # Q4_0
        assert 35 < gb < 42

    def test_estimate_7b_q4(self):
        """7B params at Q4_0 ≈ 3.9 GB."""
        gb = estimate_model_size_gb(7_000_000_000, 2)
        assert 3.5 < gb < 4.5

    def test_estimate_7b_f16(self):
        """7B params at F16 = 14 GB."""
        gb = estimate_model_size_gb(7_000_000_000, 1)  # GGUF_TYPE_F16 = 1
        assert 13 < gb < 15


class TestTensorNames:

    @staticmethod
    def _create_gguf_with_tensor(tmpdir):
        path = os.path.join(tmpdir, "tensor.gguf")
        tensor_data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 1))  # n_tensors=1
            f.write(struct.pack("<Q", 1))  # n_kv=1
            # KV pair
            _write_kv_string(f, "general.architecture", "llama")
            # Tensor info
            name = b"test.weight"
            f.write(struct.pack("<Q", len(name)))
            f.write(name)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", 3))
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<I", GGUF_TYPE_F32))
            f.write(struct.pack("<Q", 0))
            f.write(tensor_data)
        return path

    def test_tensor_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            with GGUFLoader(path) as loader:
                assert loader.tensor_names() == ["test.weight"]

    def test_tensor_info_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._create_gguf_with_tensor(tmpdir)
            with GGUFLoader(path) as loader:
                info = loader.tensor_info("test.weight")
                assert info["shape"] == (2, 3)
