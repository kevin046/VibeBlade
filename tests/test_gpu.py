"""Tests for VibeBlade GPU backends and integration."""

import numpy as np
import pytest


class TestGPUBackendAuto:
    """Test GPUBackend with auto-selection (NumPy fallback)."""

    def test_import(self):
        from vibeblade.gpu import available_backends
        assert isinstance(available_backends(), list)
        assert "numpy" in available_backends()

    def test_numpy_backend_creation(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        assert backend.backend_name == "numpy"
        assert not backend.is_gpu
        assert "NumPy" in backend._device_name

    def test_auto_backend(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="auto")
        # On CI/non-GPU machines, should fall back to numpy
        assert backend.backend_name in ("numpy", "metal", "vulkan")

    def test_drelu(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        x = np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
        result = backend.drelu(x)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.5, 2.0])

    def test_silu(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = backend.silu(x)
        # silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        np.testing.assert_array_almost_equal(result[0], 0.0, decimal=4)
        np.testing.assert_array_almost_equal(result[1], 0.7311, decimal=3)
        np.testing.assert_array_almost_equal(result[2], -0.2689, decimal=3)

    def test_rms_norm(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = backend.rms_norm(x, weight, eps=1e-6)
        rms = np.sqrt(np.mean(x ** 2) + 1e-6)
        expected = x / rms
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_matmul(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = backend.matmul(a, b)
        expected = a @ b
        np.testing.assert_array_almost_equal(result, expected)

    def test_softmax(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = backend.softmax(x)
        np.testing.assert_array_almost_equal(result.sum(), 1.0, decimal=5)

    def test_invalid_backend_raises(self):
        from vibeblade.gpu import GPUBackend
        with pytest.raises(ValueError, match="Unknown backend"):
            GPUBackend(backend="cuda")

    def test_repr(self):
        from vibeblade.gpu import GPUBackend
        backend = GPUBackend(backend="numpy")
        r = repr(backend)
        assert "numpy" in r
        assert "GPUBackend" in r


class TestPackageIntegration:
    """Test that everything imports from the top-level package."""

    def test_top_level_imports(self):
        import vibeblade
        # Core
        assert hasattr(vibeblade, "VibeBladeModel")
        assert hasattr(vibeblade, "TextGenerator")
        # GPU
        assert hasattr(vibeblade, "GPUBackend")
        assert hasattr(vibeblade, "available_backends")
        # Grammar
        assert hasattr(vibeblade, "GrammarConstraint")
        assert hasattr(vibeblade, "RegexGrammar")
        assert hasattr(vibeblade, "JsonSchemaGrammar")
        assert hasattr(vibeblade, "EbnfGrammar")
        # Version
        assert vibeblade.__version__ == "1.4.0-alpha"

    def test_grammar_generates_json(self):
        import vibeblade
        vocab = ['{', '}', '"', 'n', 'a', 'm', 'e', ':', ' ', '1', '2', '3', ',', '0']
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        gc = vibeblade.GrammarConstraint.from_json_schema(vocab, schema)
        mask = gc.get_token_mask()
        assert mask[vocab.index('{')]  # Must allow { to start JSON object

    def test_gpu_backend_works_with_grammar(self):
        """Verify GPU backend + grammar can coexist."""
        from vibeblade import GPUBackend, GrammarConstraint

        backend = GPUBackend(backend="numpy")

        # GPU backend for compute, grammar for constraints
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8, 4).astype(np.float32)
        result = backend.matmul(x, w)
        assert result.shape == (4, 4)

        vocab = ['a', 'b', 'c', '1', '2', '3']
        gc = GrammarConstraint.from_regex(vocab, '[abc]+')
        mask = gc.get_token_mask()
        assert len(mask) == len(vocab)
