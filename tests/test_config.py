"""Comprehensive tests for vibeblade.config module."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vibeblade.config import (
    ConfigError,
    OffloadConfig,
    OffloadMode,
    VibeBladeConfig,
    _system_ram_bytes,
    default_config,
    load_config,
    parse_size,
)


# ===================================================================
# parse_size
# ===================================================================

class TestParseSize:
    """Tests for the ``parse_size`` helper."""

    def test_gigabytes(self):
        assert parse_size("16GB") == 16 * (1 << 30)

    def test_megabytes(self):
        assert parse_size("256MB") == 256 * (1 << 20)

    def test_kilobytes(self):
        assert parse_size("512KB") == 512 * (1 << 10)

    def test_bytes(self):
        assert parse_size("1024B") == 1024

    def test_bare_number(self):
        assert parse_size(4096) == 4096

    def test_bare_string_number(self):
        assert parse_size("4096") == 4096

    def test_terabytes(self):
        assert parse_size("2TB") == 2 * (1 << 40)

    def test_case_insensitive(self):
        assert parse_size("8gb") == 8 * (1 << 30)
        assert parse_size("4Gb") == 4 * (1 << 30)

    def test_float_gigabytes(self):
        assert parse_size("1.5GB") == int(1.5 * (1 << 30))

    def test_whitespace_tolerance(self):
        assert parse_size("  16 GB  ") == 16 * (1 << 30)

    def test_empty_raises(self):
        with pytest.raises(ConfigError, match="Empty"):
            parse_size("")

    def test_invalid_string_raises(self):
        with pytest.raises(ConfigError, match="Cannot parse"):
            parse_size("hello")

    def test_float_passthrough(self):
        assert parse_size(4.0) == 4


# ===================================================================
# OffloadMode enum
# ===================================================================

class TestOffloadMode:
    """Tests for the ``OffloadMode`` enum."""

    def test_values(self):
        assert OffloadMode.RAM_ONLY.value == "RAM_ONLY"
        assert OffloadMode.HYBRID_SSD.value == "HYBRID_SSD"

    def test_members(self):
        assert len(OffloadMode) == 2
        assert set(OffloadMode) == {OffloadMode.RAM_ONLY, OffloadMode.HYBRID_SSD}

    def test_from_string(self):
        assert OffloadMode("RAM_ONLY") is OffloadMode.RAM_ONLY
        assert OffloadMode("HYBRID_SSD") is OffloadMode.HYBRID_SSD

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            OffloadMode("DISK_ONLY")


# ===================================================================
# OffloadConfig validation
# ===================================================================

class TestOffloadConfigValidation:
    """Tests for OffloadConfig dataclass validation."""

    def test_negative_vram_raises(self):
        with pytest.raises(ConfigError, match="vram_limit must be positive"):
            OffloadConfig(vram_limit=-1)

    def test_zero_vram_raises(self):
        with pytest.raises(ConfigError, match="vram_limit must be positive"):
            OffloadConfig(vram_limit=0)

    def test_negative_ram_raises(self):
        with pytest.raises(ConfigError, match="ram_limit must be positive"):
            OffloadConfig(ram_limit=-100)

    def test_zero_ram_raises(self):
        with pytest.raises(ConfigError, match="ram_limit must be positive"):
            OffloadConfig(ram_limit=0)

    def test_hot_threshold_zero_raises(self):
        with pytest.raises(ConfigError, match="hot_threshold must be in"):
            OffloadConfig(hot_threshold=0.0)

    def test_hot_threshold_over_one_raises(self):
        with pytest.raises(ConfigError, match="hot_threshold must be in"):
            OffloadConfig(hot_threshold=1.5)

    def test_hot_threshold_one_ok(self):
        cfg = OffloadConfig(hot_threshold=1.0)
        assert cfg.hot_threshold == 1.0

    def test_hybrid_ssd_requires_ssd_path(self):
        with pytest.raises(ConfigError, match="ssd_path is required"):
            OffloadConfig(mode=OffloadMode.HYBRID_SSD, ssd_path=None)

    def test_hybrid_ssd_bad_buffer_ratio(self):
        with pytest.raises(ConfigError, match="ram_buffer_ratio"):
            OffloadConfig(
                mode=OffloadMode.HYBRID_SSD,
                ssd_path="/tmp/cache",
                ram_buffer_ratio=0.0,
            )

    def test_hybrid_ssd_negative_preemptive(self):
        with pytest.raises(ConfigError, match="ssd_preemptive_layers"):
            OffloadConfig(
                mode=OffloadMode.HYBRID_SSD,
                ssd_path="/tmp/cache",
                ssd_preemptive_layers=-1,
            )

    def test_valid_hybrid_ssd(self):
        cfg = OffloadConfig(
            mode=OffloadMode.HYBRID_SSD,
            ssd_path="/mnt/nvme/cache",
            ram_buffer_ratio=0.3,
            ssd_preemptive_layers=4,
        )
        assert cfg.mode == OffloadMode.HYBRID_SSD
        assert cfg.ssd_path == "/mnt/nvme/cache"


# ===================================================================
# default_config
# ===================================================================

class TestDefaultConfig:
    """Tests for the ``default_config`` factory."""

    def test_returns_vibeblade_config(self):
        cfg = default_config()
        assert isinstance(cfg, VibeBladeConfig)
        assert isinstance(cfg.offload_strategy, OffloadConfig)

    def test_default_mode_is_ram_only(self):
        cfg = default_config()
        assert cfg.offload_strategy.mode == OffloadMode.RAM_ONLY

    def test_default_vram_limit(self):
        cfg = default_config()
        assert cfg.offload_strategy.vram_limit == 16 * (1 << 30)

    def test_default_hot_threshold(self):
        cfg = default_config()
        assert cfg.offload_strategy.hot_threshold == 0.15

    def test_default_ram_limit_is_positive(self):
        cfg = default_config()
        assert cfg.offload_strategy.ram_limit > 0

    def test_default_ssd_path_none_for_ram_only(self):
        cfg = default_config()
        assert cfg.offload_strategy.ssd_path is None

    def test_default_buffer_ratio(self):
        cfg = default_config()
        assert cfg.offload_strategy.ram_buffer_ratio == 0.25

    def test_default_preemptive_layers(self):
        cfg = default_config()
        assert cfg.offload_strategy.ssd_preemptive_layers == 2


# ===================================================================
# load_config — RAM_ONLY mode
# ===================================================================

class TestLoadConfigRAMOnly:
    """Tests for loading RAM_ONLY mode configs."""

    YAML_RAM_ONLY = textwrap.dedent("""\
        offload_strategy:
          mode: "RAM_ONLY"
          vram_limit: "16GB"
          ram_limit: "256GB"
          hot_threshold: 0.15
    """)

    def test_load_ram_only(self, tmp_path):
        p = tmp_path / "vibeblade.yaml"
        p.write_text(self.YAML_RAM_ONLY)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.RAM_ONLY
        assert cfg.offload_strategy.vram_limit == 16 * (1 << 30)
        assert cfg.offload_strategy.ram_limit == 256 * (1 << 30)
        assert cfg.offload_strategy.hot_threshold == pytest.approx(0.15)

    def test_minimal_ram_only(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "RAM_ONLY"
        """)
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.RAM_ONLY
        # Should pick up defaults for other fields
        assert cfg.offload_strategy.vram_limit > 0


# ===================================================================
# load_config — HYBRID_SSD mode
# ===================================================================

class TestLoadConfigHybridSSD:
    """Tests for loading HYBRID_SSD mode configs."""

    YAML_HYBRID = textwrap.dedent("""\
        offload_strategy:
          mode: "HYBRID_SSD"
          vram_limit: "16GB"
          ram_limit: "32GB"
          ssd_path: "/mnt/nvme/vibeblade_cache"
          hot_threshold: 0.15
          ram_buffer_ratio: 0.25
          ssd_preemptive_layers: 2
    """)

    def test_load_hybrid_ssd(self, tmp_path):
        p = tmp_path / "hybrid.yaml"
        p.write_text(self.YAML_HYBRID)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.HYBRID_SSD
        assert cfg.offload_strategy.vram_limit == 16 * (1 << 30)
        assert cfg.offload_strategy.ram_limit == 32 * (1 << 30)
        assert cfg.offload_strategy.ssd_path == "/mnt/nvme/vibeblade_cache"
        assert cfg.offload_strategy.ram_buffer_ratio == pytest.approx(0.25)
        assert cfg.offload_strategy.ssd_preemptive_layers == 2

    def test_hybrid_ssd_defaults_for_optional(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "HYBRID_SSD"
              ssd_path: "/tmp/ts_cache"
        """)
        p = tmp_path / "hybrid_min.yaml"
        p.write_text(yaml)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.HYBRID_SSD
        assert cfg.offload_strategy.ssd_path == "/tmp/ts_cache"
        assert cfg.offload_strategy.ram_buffer_ratio == 0.25  # default
        assert cfg.offload_strategy.ssd_preemptive_layers == 2  # default


# ===================================================================
# Comments and blank lines
# ===================================================================

class TestCommentsAndBlankLines:
    """Test that the parser ignores comments and blank lines."""

    YAML_COMMENTS = textwrap.dedent("""\
        # Top-level config for VibeBlade
        # Author: turbo@example.com

        offload_strategy:
          # Expert offload strategy
          mode: "RAM_ONLY"        # All cold experts stay in RAM
          vram_limit: "16GB"      # GPU VRAM budget
          ram_limit: "128GB"

          # Threshold for hot expert classification
          hot_threshold: 0.15     # top 15% of experts
    """)

    def test_comments_ignored(self, tmp_path):
        p = tmp_path / "comments.yaml"
        p.write_text(self.YAML_COMMENTS)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.RAM_ONLY
        assert cfg.offload_strategy.vram_limit == 16 * (1 << 30)
        assert cfg.offload_strategy.ram_limit == 128 * (1 << 30)
        assert cfg.offload_strategy.hot_threshold == pytest.approx(0.15)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        # Empty file → no offload_strategy → defaults used
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.RAM_ONLY

    def test_only_comments_file(self, tmp_path):
        p = tmp_path / "only_comments.yaml"
        p.write_text("# just a comment\n# another one\n")
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.RAM_ONLY

    def test_comment_inside_string_preserved(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "RAM_ONLY"
              ssd_path: "/path/with#hash"
        """)
        p = tmp_path / "hash_path.yaml"
        p.write_text(yaml)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.ssd_path == "/path/with#hash"


# ===================================================================
# Unknown keys
# ===================================================================

class TestUnknownKeys:
    """Test that unknown keys raise ConfigError."""

    def test_unknown_top_level_key(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "RAM_ONLY"
            unknown_key: 42
        """)
        p = tmp_path / "bad_top.yaml"
        p.write_text(yaml)
        with pytest.raises(ConfigError, match="Unknown top-level keys"):
            load_config(str(p))

    def test_unknown_offload_key(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "RAM_ONLY"
              foobar: "unknown"
        """)
        p = tmp_path / "bad_offload.yaml"
        p.write_text(yaml)
        with pytest.raises(ConfigError, match="Unknown offload_strategy keys"):
            load_config(str(p))

    def test_multiple_unknown_keys(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "RAM_ONLY"
              bad_key_1: 1
              bad_key_2: 2
        """)
        p = tmp_path / "multi_bad.yaml"
        p.write_text(yaml)
        with pytest.raises(ConfigError, match="Unknown offload_strategy keys"):
            load_config(str(p))


# ===================================================================
# Missing optional fields get defaults
# ===================================================================

class TestMissingOptionalFields:
    """Verify that omitted optional fields receive their defaults."""

    def test_partial_config(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "RAM_ONLY"
              vram_limit: "8GB"
        """)
        p = tmp_path / "partial.yaml"
        p.write_text(yaml)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.vram_limit == 8 * (1 << 30)
        # hot_threshold should use default
        assert cfg.offload_strategy.hot_threshold == 0.15
        # ram_limit should use default (system RAM)
        assert cfg.offload_strategy.ram_limit > 0

    def test_mode_only(self, tmp_path):
        yaml = textwrap.dedent("""\
            offload_strategy:
              mode: "HYBRID_SSD"
              ssd_path: "/dev/null"
        """)
        p = tmp_path / "mode_only.yaml"
        p.write_text(yaml)
        cfg = load_config(str(p))
        assert cfg.offload_strategy.mode == OffloadMode.HYBRID_SSD
        assert cfg.offload_strategy.vram_limit == 16 * (1 << 30)  # default
        assert cfg.offload_strategy.ram_buffer_ratio == 0.25  # default


# ===================================================================
# File not found
# ===================================================================

class TestFileNotFound:

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path/vibeblade.yaml")


# ===================================================================
# _system_ram_bytes helper
# ===================================================================

class TestSystemRamBytes:

    def test_returns_positive(self):
        ram = _system_ram_bytes()
        assert ram > 0
        assert isinstance(ram, int)

    def test_reasonable_range(self):
        ram = _system_ram_bytes()
        # Assume at least 1 GiB and at most 4 TiB for any test machine
        assert 1 * (1 << 30) <= ram <= 4 * (1 << 40)


# ===================================================================
# VibeBladeConfig wrapping
# ===================================================================

class TestVibeBladeConfig:

    def test_default_construction(self):
        cfg = VibeBladeConfig()
        assert isinstance(cfg.offload_strategy, OffloadConfig)

    def test_custom_offload(self):
        oc = OffloadConfig(mode=OffloadMode.RAM_ONLY, vram_limit=8 * (1 << 30))
        cfg = VibeBladeConfig(offload_strategy=oc)
        assert cfg.offload_strategy.vram_limit == 8 * (1 << 30)
