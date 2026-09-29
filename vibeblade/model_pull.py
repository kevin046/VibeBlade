"""VibeBlade Model Pull — install ANY compatible model straight from HuggingFace.

`vibeblade pull bartowski/Llama-3.2-3B-Instruct-GGUF` downloads the repo's GGUF
file(s) into ~/.vibeblade/models (or --dir), auto-detects quantization, picks
the best default variant, registers the model in the local registry, and prints
a ready-to-run serve command.

Design notes:
- ZERO dependencies: stdlib urllib streaming download with HTTP Range resume.
  (HuggingFace public repos need no auth; set HF_TOKEN env for gated repos.)
- Compatible = GGUF (any quant this loader handles) or safetensors dirs.
- Progress: one line per file, human-readable size + speed; resumable —
  re-running `pull` resumes a partial .part download instead of restarting.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from vibeblade.model_manager import ModelManager, detect_model_format

HF_BASE = "https://huggingface.co"
# Enterprise reality: some networks/IPs are gated by HF's CDN. Endpoint chain:
#   1. HF_ENDPOINT env (org mirror/proxy)  2. huggingface.co  3. hf-mirror.com
_ENDPOINT_CHAIN = [
    os.environ.get("HF_ENDPOINT", "").rstrip("/"),
    HF_BASE,
    "https://hf-mirror.com",
]
_ENDPOINT_CHAIN = [e for e in _ENDPOINT_CHAIN if e]
_worked_endpoint: str | None = None
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

_REPO_RE = re.compile(r"^(?P<org>[\w.-]+)/(?P<repo>[\w.-]+)$")
_CHUNK = 1 << 20  # 1 MiB


class PullError(RuntimeError):
    """Raised when a model cannot be pulled from HuggingFace."""


@dataclass
class PullResult:
    repo_id: str
    files: list[str] = field(default_factory=list)
    model_dir: str = ""
    model_id: str = ""
    total_bytes: int = 0

    def summary(self) -> str:
        files = "\n".join(f"  {f}" for f in self.files)
        return (
            f"Pulled {self.repo_id} → {self.model_dir}\n"
            f"Files:\n{files}\n"
            f"Registered as model id: {self.model_id}\n"
            f"Run it:\n  vibeblade serve --model {self.model_id}"
        )


def _auth_headers() -> dict:
    if _HF_TOKEN:
        return {"Authorization": f"Bearer {_HF_TOKEN}"}
    return {}


def _http_get(url: str, timeout: int = 30) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "VibeBlade-Pull/1.0", **_auth_headers()})
    return req


def list_repo_files(repo_id: str, revision: str = "main") -> list[str]:
    """List files in a HF repo via the public API (no auth for public repos).

    Tries the endpoint chain (env mirror → huggingface.co → hf-mirror) and
    remembers the first endpoint that works for subsequent downloads.
    """
    global _worked_endpoint
    chain = ([_worked_endpoint] if _worked_endpoint else []) + [
        e for e in _ENDPOINT_CHAIN if e != _worked_endpoint
    ]
    last_err: Exception | None = None
    for base in chain:
        url = f"{base}/api/models/{repo_id}/tree/{revision}?recursive=true"
        req = _http_get(url)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                import json

                tree = json.loads(resp.read().decode("utf-8"))
            _worked_endpoint = base
            return [entry["path"] for entry in tree if entry.get("type") == "file"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise PullError(
                    f"Repo not found: {repo_id} (revision {revision})"
                ) from e
            last_err = e
            continue  # try next endpoint
        except urllib.error.URLError as e:
            last_err = e
            continue
    if last_err and isinstance(last_err, urllib.error.HTTPError) and last_err.code in (401, 403):
        raise PullError(
            f"Repo {repo_id} is gated or blocked for this network — set HF_TOKEN, "
            f"or point HF_ENDPOINT at a mirror you can reach"
        ) from last_err
    raise PullError(f"Cannot reach any HuggingFace endpoint for {repo_id}: {last_err}")


def timeout_s(s: int) -> int:
    return s


def _pick_gguf_files(files: list[str], pattern: str | None = None) -> list[str]:
    """Choose GGUF file(s) to download.

    Preference: single monolithic *.gguf; else sharded index is ignored and all
    shards matching the same quant are chosen. `pattern` is a substring filter
    (e.g. 'Q4_K_M') applied case-insensitively.
    """
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if pattern:
        ggufs = [f for f in ggufs if pattern.lower() in f.lower()]
        if not ggufs:
            raise PullError(f"No GGUF matching '{pattern}' in repo")
    if not ggufs:
        return []
    # Sharded models: prefer the LARGEST quant family group (same prefix)
    if len(ggufs) > 1 and pattern is None:
        groups: dict[str, list[str]] = {}
        for f in ggufs:
            stem = f.rsplit("/", 1)[-1]
            base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", stem, flags=re.IGNORECASE)
            base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
            groups.setdefault(base, []).append(f)
        # Group key = family; pick family whose name sorts LAST usually = bigger quant,
        # but better: prefer Q4_K_M if present, else largest total size hint from name
        for preferred in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q3_K_M", "Q8_0"):
            for base, fs in groups.items():
                if preferred.lower() in base.lower():
                    return fs
        # fall back: biggest family by member count
        return max(groups.values(), key=len)
    return ggufs


def _pick_aux_files(files: list[str]) -> list[str]:
    """Small metadata files worth grabbing (tokenizer/configs) for GGUF-less repos."""
    keep = []
    for f in files:
        low = f.lower()
        if low.endswith((".json", ".txt")) and "/".count(f) <= 1:
            if any(k in low for k in ("tokenizer", "config", "special_tokens", "generation")):
                keep.append(f)
    return keep


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n}TB"


def _download(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Streaming download with resume. Writes dest.part, renames on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "VibeBlade-Pull/1.0", **_auth_headers()}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable = already complete
            part.rename(dest)
            return
        raise
    total = expected_size or int(resp.headers.get("Content-Length", 0)) + start
    mode = "ab" if start > 0 else "wb"
    t0 = time.time()
    done = start
    with open(part, mode) as fh, resp:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                speed = done / max(time.time() - t0, 0.001)
                print(f"\r  {_fmt_size(done)} / {_fmt_size(total)} ({pct:5.1f}%) @ {_fmt_size(int(speed))}/s", end="", flush=True)
    print()
    if total and done != total:
        raise PullError(f"Size mismatch for {dest.name}: got {done}, expected {total} (retry to resume)")
    part.rename(dest)


def pull_model(
    repo_id: str,
    *,
    dest_dir: str = "",
    pattern: str | None = None,
    revision: str = "main",
    include_safetensors: bool = False,
) -> PullResult:
    """Pull a model from HuggingFace into the local VibeBlade registry."""
    m = _REPO_RE.match(repo_id)
    if not m:
        raise PullError(f"Expected repo id like 'org/model', got: {repo_id!r}")

    files = list_repo_files(repo_id, revision)
    ggufs = _pick_gguf_files(files, pattern)
    aux = _pick_aux_files(files)

    sfts = []
    if include_safetensors or (not ggufs and any(f.endswith(".safetensors") for f in files)):
        sfts = [f for f in files if f.endswith(".safetensors")]
        if not ggufs and not sfts:
            raise PullError(f"No GGUF or safetensors files in {repo_id}")

    # Dedupe while preserving order (aux files can overlap safetensors picks)
    seen: set[str] = set()
    targets = []
    for f in ggufs + sfts + aux:
        if f not in seen:
            seen.add(f)
            targets.append(f)
    if not targets:
        raise PullError(f"No downloadable model files found in {repo_id}")

    models_dir = Path(dest_dir) if dest_dir else Path.home() / ".vibeblade" / "models"
    model_dir = models_dir / repo_id.split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    downloaded: list[str] = []
    for rel in targets:
        base = _worked_endpoint or HF_BASE
        url = f"{base}/{repo_id}/resolve/{revision}/{rel}"
        # HF tree paths use forward slashes; keep subpaths
        dest = model_dir / rel
        print(f"↓ {rel}")
        _download(url, dest)
        downloaded.append(str(dest))
        total += dest.stat().st_size

    # Register in the model registry
    mgr = ModelManager(str(models_dir))
    mgr.scan_directory(str(models_dir), include_external=False)
    fmt, quant = detect_model_format(str(model_dir))
    rec = mgr.register(
        path=str(model_dir),
        model_id=repo_id,
        format=fmt,
        quant_type=quant,
        source="huggingface",
    )
    return PullResult(
        repo_id=repo_id,
        files=[Path(p).name for p in downloaded],
        model_dir=str(model_dir),
        model_id=rec.model_id if rec else repo_id.split("/")[-1],
        total_bytes=total,
    )
