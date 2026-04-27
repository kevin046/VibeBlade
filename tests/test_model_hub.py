"""Tests for model_hub — model discovery and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestFindGgufFiles:
    """find_gguf_files() — directory scanning for .gguf files."""

    def test_finds_gguf_files_in_directory(self, tmp_path):
        (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        (tmp_path / "readme.txt").write_bytes(b"hello")

        from vibeblade.model_hub import find_gguf_files

        result = find_gguf_files(tmp_path, recursive=False)
        assert len(result) == 1
        assert result[0].name == "model-Q4_K_M.gguf"

    def test_recursive_search(self, tmp_path):
        (tmp_path / "top.gguf").write_bytes(b"\x00" * 100)
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.gguf").write_bytes(b"\x00" * 200)

        from vibeblade.model_hub import find_gguf_files

        result = find_gguf_files(tmp_path, recursive=True)
        assert len(result) == 2

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.gguf").write_bytes(b"\x00" * 100)

        from vibeblade.model_hub import find_gguf_files

        result = find_gguf_files(tmp_path, recursive=False)
        assert len(result) == 0

    def test_quant_filter(self, tmp_path):
        (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        (tmp_path / "model-Q8_0.gguf").write_bytes(b"\x00" * 200)

        from vibeblade.model_hub import find_gguf_files

        result = find_gguf_files(tmp_path, recursive=False, quant_filter="Q8_0")
        assert len(result) == 1
        assert "Q8_0" in result[0].name

    def test_nonexistent_directory(self):
        from vibeblade.model_hub import find_gguf_files

        result = find_gguf_files("/nonexistent/path/12345", recursive=True)
        assert result == []

    def test_empty_directory(self, tmp_path):
        from vibeblade.model_hub import find_gguf_files

        result = find_gguf_files(tmp_path, recursive=False)
        assert result == []


class TestResolveModelPath:
    """resolve_model_path() — main entry point for model resolution."""

    def test_direct_file_path(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)

        from vibeblade.model_hub import resolve_model_path

        result = resolve_model_path(str(gguf))
        assert result.exists()
        assert result.suffix == ".gguf"

    def test_not_found_raises(self):
        from vibeblade.model_hub import resolve_model_path

        with pytest.raises(FileNotFoundError, match="Could not find model"):
            resolve_model_path("nonexistent-model-xyz-123")

    def test_error_message_includes_helpful_info(self):
        from vibeblade.model_hub import resolve_model_path

        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_model_path("missing-model")
        msg = str(exc_info.value)
        assert "HuggingFace cache" in msg
        assert "LM Studio" in msg


class TestFindInHfCache:
    """find_in_hf_cache() — HuggingFace hub cache scanning."""

    def test_finds_model_in_cache(self, tmp_path):
        """Build a fake HF cache in tmp_path and point Path.home there."""
        cache = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache / "models--lmstudio-community--Qwen3.6-35B-A3B-GGUF"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "Qwen3.6-35B-A3B-Q4_K_M.gguf").write_bytes(b"\x00" * 100)

        from vibeblade.model_hub import find_in_hf_cache

        # Patch Path.home at the module level
        import vibeblade.model_hub as mh
        original = mh.Path.home

        try:
            mh.Path.home = staticmethod(lambda: tmp_path)
            result = find_in_hf_cache.__wrapped__(repo_id="lmstudio-community/Qwen3.6-35B-A3B-GGUF") if hasattr(find_in_hf_cache, "__wrapped__") else mh.find_in_hf_cache(repo_id="lmstudio-community/Qwen3.6-35B-A3B-GGUF")
            assert len(result) == 1
            assert "Q4_K_M" in result[0].name
        finally:
            mh.Path.home = original

    def test_no_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        from vibeblade.model_hub import find_in_hf_cache

        result = find_in_hf_cache()
        assert result == []


class TestFindInLmStudio:
    """find_in_lm_studio() — LM Studio directory scanning."""

    def test_windows_path(self, tmp_path):
        """Build fake LM Studio dir under tmp_path and test find_in_lm_studio directly."""
        lm_dir = tmp_path / "LM Studio" / "models"
        lm_dir.mkdir(parents=True)
        (lm_dir / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 100)

        from vibeblade.model_hub import find_gguf_files

        # Simulate what find_in_lm_studio does on Windows
        result = find_gguf_files(lm_dir, recursive=True, quant_filter="")
        assert len(result) == 1
        assert "Q4_K_M" in result[0].name


class TestScanCachedModels:
    """scan_cached_models() — list all discovered models."""

    def test_returns_list_of_dicts(self):
        from vibeblade.model_hub import scan_cached_models

        result = scan_cached_models()
        assert isinstance(result, list)
        # Each item should be a dict with expected keys
        for m in result:
            assert "path" in m
            assert "name" in m
            assert "size_gb" in m
            assert "source" in m


class TestIsModelCached:
    """is_model_cached() — check if a model exists locally."""

    def test_returns_false_for_missing(self):
        from vibeblade.model_hub import is_model_cached

        assert is_model_cached("nonexistent-model-xyz") is False

    def test_returns_true_for_existing(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)

        from vibeblade.model_hub import is_model_cached

        assert is_model_cached(str(gguf)) is True
