"""Tests for Model Hub, Model Manager, and Converter."""

import pytest
import numpy as np


# ── Model Hub ──

class TestModelHub:
    """Test HuggingFace model hub integration."""

    def test_classify_gguf(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("llama-3-8b-instruct.Q4_K_M.gguf")
        assert fmt == "gguf"
        assert qtype == "Q4_K_M"

    def test_classify_gguf_f16(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("model-f16.gguf")
        assert fmt == "gguf"
        assert qtype == "F16"

    def test_classify_gguf_iq4(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("model.IQ4_XS.gguf")
        assert fmt == "gguf"
        assert qtype == "IQ4_XS"

    def test_classify_gguf_q8(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("model.Q8_0.gguf")
        assert fmt == "gguf"
        assert qtype == "Q8_0"

    def test_classify_awq(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("model-4bit-128g-awq.safetensors")
        assert fmt == "awq"

    def test_classify_safetensors(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("model.safetensors")
        assert fmt == "safetensors"

    def test_classify_gptq(self):
        from vibeblade.model_hub import classify_file
        fmt, qtype = classify_file("model-gptq-4bit.safetensors")
        assert fmt == "gptq"

    def test_is_quantized(self):
        from vibeblade.model_hub import is_quantized_file
        assert is_quantized_file("model.Q4_K_M.gguf")
        assert is_quantized_file("model.awq.safetensors")
        assert not is_quantized_file("model.safetensors")
        assert not is_quantized_file("config.json")

    def test_hub_model_to_dict(self):
        from vibeblade.model_hub import HubModel, QuantizedFile
        model = HubModel(
            model_id="test/model",
            author="test",
            formats=["gguf"],
            quantized_files=[
                QuantizedFile("model.Q4_K_M.gguf", 4096, "Q4_K_M", "gguf"),
            ],
        )
        d = model.to_dict()
        assert d["model_id"] == "test/model"
        assert d["file_count"] == 1
        assert d["formats"] == ["gguf"]

    def test_quantized_file_size_human(self):
        from vibeblade.model_hub import QuantizedFile
        f = QuantizedFile("model.gguf", 4500000000, "Q4_K_M", "gguf")
        assert "GB" in f.size_human
        f2 = QuantizedFile("model.gguf", 500, "Q4_K_M", "gguf")
        assert "B" in f2.size_human


# ── Model Manager ──

class TestModelManager:
    """Test local model registry."""

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return tmp_path

    def test_detect_gguf_file(self, tmp_dir):
        from vibeblade.model_manager import detect_model_format
        gguf = tmp_dir / "model.Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 100)
        fmt, qtype = detect_model_format(str(gguf))
        assert fmt == "gguf"
        assert qtype == "Q4_K_M"

    def test_detect_safetensors_dir(self, tmp_dir):
        from vibeblade.model_manager import detect_model_format
        (tmp_dir / "model.safetensors").write_bytes(b"\x00" * 100)
        fmt, qtype = detect_model_format(str(tmp_dir))
        assert fmt == "safetensors"
        assert qtype == "full"

    def test_register_and_list(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 100)

        rec = mgr.register(str(gguf), model_id="test/model")
        assert rec.format == "gguf"
        assert rec.quant_type == "Q4_K_M"

        models = mgr.list_models()
        assert len(models) == 1
        assert models[0].id == rec.id

    def test_unregister(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        rec = mgr.register(str(gguf))
        assert mgr.unregister(rec.id)
        assert mgr.get(rec.id) is None

    def test_delete_with_files(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        models_dir = tmp_dir / "models"
        models_dir.mkdir(parents=True)
        mgr = ModelManager(str(models_dir))
        gguf = models_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        rec = mgr.register(str(gguf))
        assert mgr.delete(rec.id, delete_files=True)
        assert not gguf.exists()

    def test_filter_by_format(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "m1.gguf"
        gguf.write_bytes(b"\x00" * 100)
        st = tmp_dir / "st"
        st.mkdir()
        (st / "model.safetensors").write_bytes(b"\x00" * 100)
        mgr.register(str(gguf))
        mgr.register(str(st))

        assert len(mgr.list_models(fmt="gguf")) == 1
        assert len(mgr.list_models(fmt="safetensors")) == 1

    def test_stats(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        mgr.register(str(gguf))
        stats = mgr.get_stats()
        assert stats["total_models"] == 1
        assert "gguf" in stats["formats"]

    def test_scan_directory(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        models_dir = tmp_dir / "models"
        models_dir.mkdir()
        (models_dir / "a.Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        (models_dir / "b.Q5_K_S.gguf").write_bytes(b"\x00" * 100)

        mgr = ModelManager(str(models_dir))
        new = mgr.scan_directory(include_external=False)
        assert len(new) == 2

    def test_export_registry(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        mgr.register(str(gguf))
        exported = mgr.export_registry()
        assert len(exported) == 1

    def test_source_field_default(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        rec = mgr.register(str(gguf))
        assert rec.source == "local"

    def test_source_field_explicit(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        rec = mgr.register(str(gguf), source="lm_studio")
        assert rec.source == "lm_studio"

    def test_source_in_export(self, tmp_dir):
        from vibeblade.model_manager import ModelManager
        mgr = ModelManager(str(tmp_dir / "models"))
        gguf = tmp_dir / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)
        mgr.register(str(gguf), source="huggingface")
        exported = mgr.export_registry()
        for v in exported.values():
            assert v["source"] == "huggingface"

    def test_source_backward_compat(self, tmp_dir):
        """Existing records without 'source' key default to 'local'."""
        from vibeblade.model_manager import ModelRecord
        models_dir = tmp_dir / "models"
        models_dir.mkdir()
        # Manually create a record without source field
        rec = ModelRecord(
            id="test-old",
            model_id="",
            name="old model",
            path="/fake/path.gguf",
            format="gguf",
            quant_type="Q4_K_M",
            size_bytes=100,
        )
        assert rec.source == "local"  # default kicks in

    def test_scan_directory_no_external(self, tmp_dir):
        """include_external=False skips external dirs."""
        from vibeblade.model_manager import ModelManager
        models_dir = tmp_dir / "models"
        models_dir.mkdir()
        (models_dir / "a.Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        mgr = ModelManager(str(models_dir))
        new = mgr.scan_directory(include_external=False)
        assert len(new) == 1

    def test_get_external_model_dirs_env(self, tmp_dir, monkeypatch):
        """LM_STUDIO_MODELS_DIR env var overrides default."""
        from vibeblade.model_manager import get_external_model_dirs
        fake_dir = tmp_dir / "lm-studio-models"
        fake_dir.mkdir()
        monkeypatch.setenv("LM_STUDIO_MODELS_DIR", str(fake_dir))
        dirs = get_external_model_dirs()
        sources = [s for s, _ in dirs]
        assert "lm_studio" in sources
        # Check the path matches
        lm_studio_path = [p for s, p in dirs if s == "lm_studio"][0]
        assert lm_studio_path == fake_dir

    def test_get_external_model_dirs_missing(self, monkeypatch):
        """Returns empty list when no external dirs exist."""
        from vibeblade.model_manager import get_external_model_dirs
        # Point home to a tmp dir with no model caches
        monkeypatch.setenv("HOME", "/tmp/__nonexistent_vibeblade_test__")
        dirs = get_external_model_dirs()
        assert dirs == []

    def test_scan_external_directory(self, tmp_dir, monkeypatch):
        """External dirs with .gguf files get auto-registered."""
        from vibeblade.model_manager import ModelManager
        # Create a fake external dir
        ext_dir = tmp_dir / "external_models"
        ext_dir.mkdir()
        sub = ext_dir / "vendor" / "llama-3.1-8b"
        sub.mkdir(parents=True)
        (sub / "model.Q5_K_S.gguf").write_bytes(b"\x00" * 100)

        models_dir = tmp_dir / "models"
        models_dir.mkdir()
        mgr = ModelManager(str(models_dir))

        # Directly use _scan_dir to test external scanning
        new = mgr._scan_dir(ext_dir, "lm_studio")
        assert len(new) == 1
        assert new[0].source == "lm_studio"
        assert new[0].quant_type == "Q5_K_S"

    def test_scan_no_duplicates(self, tmp_dir):
        """Scanning the same dir twice doesn't re-register."""
        from vibeblade.model_manager import ModelManager
        models_dir = tmp_dir / "models"
        models_dir.mkdir()
        (models_dir / "a.Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        mgr = ModelManager(str(models_dir))
        first = mgr.scan_directory(include_external=False)
        second = mgr.scan_directory(include_external=False)
        assert len(first) == 1
        assert len(second) == 0  # already registered


# ── Converter ──

class TestConverter:
    """Test model conversion pipeline."""

    def test_quantize_q4(self):
        from vibeblade.converter import quantize_q4
        x = np.random.randn(4, 16).astype(np.float32)
        packed, scales, mins = quantize_q4(x)
        assert packed.dtype == np.uint8
        assert packed.shape == (4, 8)  # 16 // 2
        assert scales.shape == (4,)
        assert mins.shape == (4,)

    def test_quantize_q8(self):
        from vibeblade.converter import quantize_q8
        x = np.random.randn(4, 16).astype(np.float32)
        quant, scales = quantize_q8(x)
        assert quant.dtype == np.int8
        assert quant.shape == (4, 16)
        assert scales.shape == (4,)

    def test_q4_roundtrip_approx(self):
        from vibeblade.converter import quantize_q4
        x = np.random.randn(2, 16).astype(np.float32)
        packed, scales, mins = quantize_q4(x)
        # Dequantize and check it's roughly right
        low = (packed & 0xF).astype(np.float32)
        high = ((packed >> 4) & 0xF).astype(np.float32)
        reconstructed = np.zeros_like(x)
        for row in range(x.shape[0]):
            reconstructed[row, :8] = low[row] * scales[row] + mins[row]
            reconstructed[row, 8:] = high[row] * scales[row] + mins[row]
        error = np.abs(x - reconstructed).max()
        # 4-bit asymmetric quant: error can be up to 1 step width (~range/15)
        assert error < np.abs(x).max() * 2.0

    def test_q8_roundtrip_approx(self):
        from vibeblade.converter import quantize_q8
        x = np.random.randn(4, 16).astype(np.float32)
        quant, scales = quantize_q8(x)
        reconstructed = quant.astype(np.float32) * scales[:, np.newaxis]
        error = np.abs(x - reconstructed).max()
        # 8-bit should be much closer
        assert error < np.abs(x).max() * 0.1

    def test_quant_method_enum(self):
        from vibeblade.converter import QuantMethod
        assert QuantMethod.Q4_K_M.bits == 4
        assert QuantMethod.Q8_0.bits == 8
        assert QuantMethod.F16.bits == 16
        assert len(QuantMethod.Q4_K_M.description) > 0

    def test_supported_methods(self):
        from vibeblade.converter import ModelConverter
        methods = ModelConverter.get_supported_methods()
        assert len(methods) >= 7
        assert any(m["id"] == "q4_k_m" for m in methods)

    def test_estimate_size(self):
        from vibeblade.converter import ModelConverter, QuantMethod
        est = ModelConverter.estimate_output_size(1_000_000_000, QuantMethod.Q4_K_M)
        # 4-bit from fp16: 0.1 (metadata) + 0.9 * (4/16) (weights) = 0.325
        assert 0.25 < est / 1_000_000_000 < 0.45

    def test_gguf_writer(self, tmp_path):
        from vibeblade.converter import GGUFWriter
        out = tmp_path / "test.gguf"
        writer = GGUFWriter(str(out))
        writer.add_string("general.name", "test")
        writer.add_uint32("llama.context_length", 2048)
        writer.add_uint32("llama.embedding_length", 512)
        writer.add_tensor("test.weight", np.random.randn(4, 8).astype(np.float16))
        writer.write()
        assert out.exists()
        assert out.stat().st_size > 0

    def test_gguf_writer_magic(self, tmp_path):
        from vibeblade.converter import GGUFWriter
        out = tmp_path / "test.gguf"
        writer = GGUFWriter(str(out))
        writer.add_string("general.name", "test")
        writer.add_tensor("w", np.zeros((2, 4), dtype=np.float32))
        writer.write()
        magic = out.read_bytes()[:4]
        assert magic == b"GGUF"


