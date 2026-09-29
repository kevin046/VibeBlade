"""Tests for the LAYER_MAP offload customization (config.py).

Run: python3 tests/test_layer_map.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vibeblade.config import (  # noqa: E402
    ConfigError,
    OffloadConfig,
    OffloadMode,
    layer_tier_summary,
    load_config,
    parse_layer_map,
)


def test_basic_map():
    p = parse_layer_map("gpu:0-15, ram:16-23, ssd:24-31")
    assert p[0] == "gpu" and p[16] == "ram" and p[31] == "ssd" and len(p) == 32
    assert layer_tier_summary(p) == "gpu: 16 layers, ram: 8 layers, ssd: 8 layers"
    print("PASS basic map:", layer_tier_summary(p))


def test_aliases_single_reversed():
    # non-overlapping ranges; reversed range (3-2 == 2-3) normalizes
    p = parse_layer_map("vram:0-1, cpu:8, memory:9-12, disk:3-2")
    assert p[0] == "gpu" and p[8] == "ram" and p[12] == "ram"
    assert p[2] == "ssd" and p[3] == "ssd"
    print("PASS aliases / single index / reversed range")


def test_overlap_rejected():
    try:
        parse_layer_map("gpu:0-15, ram:10-20")
        raise AssertionError("overlap not rejected")
    except ConfigError as e:
        print("PASS overlap rejected:", e)


def test_ssd_path_required():
    try:
        OffloadConfig(mode=OffloadMode.LAYER_MAP, layer_map="gpu:0-7, ssd:8-15")
        raise AssertionError("missing ssd_path not rejected")
    except ConfigError as e:
        print("PASS ssd_path required:", e)


def test_layer_map_config():
    oc = OffloadConfig(mode=OffloadMode.LAYER_MAP, layer_map="gpu:0-15, ssd:16-31", ssd_path="/tmp/vb")
    assert oc.parsed_layer_map[16] == "ssd" and oc.parsed_layer_map[15] == "gpu"
    print("PASS LAYER_MAP config:", layer_tier_summary(oc.parsed_layer_map))


def test_gpu_auto_mode():
    # vram_limit accepts human strings via the yaml path; direct constructor takes bytes
    oc = OffloadConfig(mode=OffloadMode.GPU_AUTO, vram_limit=8 << 30)
    assert oc.mode == OffloadMode.GPU_AUTO and oc.vram_limit == 8 << 30
    print("PASS GPU_AUTO mode")


def test_yaml_roundtrip():
    yaml = (
        "offload_strategy:\n"
        '  mode: "LAYER_MAP"\n'
        '  layer_map: "gpu:0-15, ram:16-23, ssd:24-31"\n'
        '  ssd_path: "/mnt/nvme/cache"\n'
        '  vram_limit: "24GB"\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml)
        path = f.name
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)
    assert cfg.offload_strategy.mode == OffloadMode.LAYER_MAP
    assert cfg.offload_strategy.parsed_layer_map[24] == "ssd"
    assert cfg.offload_strategy.vram_limit == 24 << 30
    print("PASS yaml round-trip:", layer_tier_summary(cfg.offload_strategy.parsed_layer_map))


def test_unknown_key_rejected():
    yaml = 'offload_strategy:\n  mode: "GPU_AUTO"\n  bogus_key: "1"\n'
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml)
        path = f.name
    try:
        try:
            load_config(path)
            raise AssertionError("unknown key not rejected")
        except ConfigError as e:
            print("PASS unknown key rejected:", e)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_basic_map()
    test_aliases_single_reversed()
    test_overlap_rejected()
    test_ssd_path_required()
    test_layer_map_config()
    test_gpu_auto_mode()
    test_yaml_roundtrip()
    test_unknown_key_rejected()
    print("\nALL LAYER-MAP TESTS PASS")
