"""
llama.cpp ctypes backend for VibeBlade.

Clean Python wrapper around llama.cpp C API via ctypes.
Uses a C helper (libparams_helper.so) for struct param construction.
"""
from __future__ import annotations

import ctypes
import os
import time as _time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional


# ============================================================
# Library loading — lazy proxy so libs load only when actually used
# ============================================================
def _find_lib(base_name: str, candidates: list[Path]) -> str:
  for c in candidates:
    p = c / base_name
    if p.exists():
      return str(p)
  raise FileNotFoundError(f"{base_name} not found in {[str(c) for c in candidates]}")

_BASE = Path(__file__).parent.parent
_LLAMA_BIN = _BASE / "llama.cpp" / "build" / "bin"

_LLAMA_CANDIDATES = [
  _LLAMA_BIN,
  _BASE / "build" / "bin",
  Path(os.environ.get("LLAMA_BUILD_DIR", "")) / "bin",
]


class _LazyCDLL:
  """Proxy that defers ctypes.CDLL loading + argtype setup until first attribute access."""

  _lib = None
  _setup = False

  def _load(self, candidates: list[Path], lib_name: str) -> ctypes.CDLL:
    if _LazyCDLL._lib is None:
      _LazyCDLL._lib = ctypes.CDLL(_find_lib(lib_name, candidates))
    return _LazyCDLL._lib

  def _setup_lib(self, lib):
    """Run once after library is first loaded."""
    if _LazyCDLL._setup:
      return
    _LazyCDLL._setup = True

    # --- Model ---
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = ctypes.c_void_p
    # llama_model_load_from_file / llama_init_from_model take structs BY VALUE.
    # The helper stores them in static buffers; we memmove into opaque ctypes structs.
    class _OpaqueParams(ctypes.Structure):
      _fields_ = [("_pad", ctypes.c_byte * 256)]
    lib._opaque_params_cls = _OpaqueParams  # stash for load()
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _OpaqueParams]
    lib.llama_model_load_from_file.restype = ctypes.c_void_p
    lib.llama_model_free.argtypes = [ctypes.c_void_p]
    lib.llama_model_free.restype = None
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p

    # --- Vocab ---
    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_n_tokens.restype = ctypes.c_int32
    lib.llama_vocab_eos.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_eos.restype = ctypes.c_int32

    # --- Context ---
    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = ctypes.c_void_p
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, _OpaqueParams]
    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free.restype = None

    # --- Decode ---
    lib.llama_decode.argtypes = [ctypes.c_void_p, LLamaBatch]
    lib.llama_decode.restype = ctypes.c_int32
    lib.llama_get_logits.argtypes = [ctypes.c_void_p]
    lib.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    # --- Tokenize / Detokenize ---
    lib.llama_tokenize.argtypes = [
      ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
      ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
      ctypes.c_bool, ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32
    lib.llama_token_to_piece.argtypes = [
      ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p,
      ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32
    lib.llama_detokenize.argtypes = [
      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
      ctypes.c_char_p, ctypes.c_int32, ctypes.c_bool, ctypes.c_bool,
    ]
    lib.llama_detokenize.restype = ctypes.c_int32

    # --- Batch ---
    lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.llama_batch_init.restype = LLamaBatch
    lib.llama_batch_free.argtypes = [LLamaBatch]
    lib.llama_batch_free.restype = None

    # --- KV cache ---
    lib.llama_get_memory.argtypes = [ctypes.c_void_p]
    lib.llama_get_memory.restype = ctypes.c_void_p
    lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.llama_memory_clear.restype = None

    # --- Sampling ---
    lib.llama_sampler_init_greedy.argtypes = []
    lib.llama_sampler_init_greedy.restype = ctypes.c_void_p
    lib.llama_sampler_init_grammar.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.llama_sampler_init_grammar.restype = ctypes.c_void_p
    lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    lib.llama_sampler_init_top_k.restype = ctypes.c_void_p
    lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_uint32]
    lib.llama_sampler_init_top_p.restype = ctypes.c_void_p
    lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    lib.llama_sampler_init_temp.restype = ctypes.c_void_p
    lib.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llama_sampler_chain_add.restype = None
    lib.llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    lib.llama_sampler_sample.restype = ctypes.c_int32
    lib.llama_sampler_reset.argtypes = [ctypes.c_void_p]
    lib.llama_sampler_reset.restype = None
    lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
    lib.llama_sampler_free.restype = None

    # --- Threads / timing ---
    lib.llama_synchronize.argtypes = [ctypes.c_void_p]
    lib.llama_synchronize.restype = None
    lib.llama_set_n_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.llama_set_n_threads.restype = None
    lib.llama_n_threads_batch.argtypes = [ctypes.c_void_p]
    lib.llama_n_threads_batch.restype = ctypes.c_int32

    # --- TurboSparse ---
    lib.llama_turbosparse_enable.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.llama_turbosparse_enable.restype = None
    lib.llama_turbosparse_disable.argtypes = [ctypes.c_void_p]
    lib.llama_turbosparse_disable.restype = None
    lib.llama_turbosparse_is_enabled.argtypes = [ctypes.c_void_p]
    lib.llama_turbosparse_is_enabled.restype = ctypes.c_bool
    lib.llama_turbosparse_get_threshold.argtypes = [ctypes.c_void_p]
    lib.llama_turbosparse_get_threshold.restype = ctypes.c_float

  def __getattr__(self, name):
    if _LazyCDLL._lib is None:
      lib = self._load(_LLAMA_CANDIDATES, "libllama.so")
      self._setup_lib(lib)
    return getattr(_LazyCDLL._lib, name)


class _LazyHelperCDLL:
  """Proxy that defers loading of libparams_helper.so until first attribute access."""

  _lib = None

  def _load(self, candidates: list[Path], lib_name: str) -> ctypes.CDLL:
    if _LazyHelperCDLL._lib is None:
      _LazyHelperCDLL._lib = ctypes.CDLL(_find_lib(lib_name, candidates))
    return _LazyHelperCDLL._lib

  def __getattr__(self, name):
    if _LazyHelperCDLL._lib is None:
      self._load([_LLAMA_BIN], "libparams_helper.so")
    return getattr(_LazyHelperCDLL._lib, name)


# Module-level proxies — loading only happens on first actual use
_lib = _LazyCDLL()
_helper = _LazyHelperCDLL()


# ============================================================
# C struct: llama_batch
# ============================================================
class LLamaBatch(ctypes.Structure):
  """C llama_batch struct — 7 fields, NO logit_count."""
  _fields_ = [
    ("n_tokens", ctypes.c_int32),
    ("token", ctypes.POINTER(ctypes.c_int32)),
    ("embd", ctypes.POINTER(ctypes.c_float)),
    ("pos", ctypes.POINTER(ctypes.c_int32)),
    ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
    ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
    ("logits", ctypes.POINTER(ctypes.c_int8)),
  ]


# ============================================================
# Result dataclass
# ============================================================
@dataclass
class GenerateResult:
  text: str
  tokens: list[int]
  tokens_per_second: float
  prompt_tokens: int
  stop_reason: str  # "eos" | "stop_token" | "max_tokens"
  time_prefill: float = 0.0
  time_decode: float = 0.0
  time_total: float = 0.0


# ============================================================
# Module-level LRU cache for token → text
# ============================================================
@lru_cache(maxsize=65536)
def _detokenize_cached(vocab_ptr: int, token_id: int) -> str:
  """Cached single-token detokenization."""
  piece_buf = ctypes.create_string_buffer(256)
  n = _lib.llama_token_to_piece(
    ctypes.cast(vocab_ptr, ctypes.c_void_p),
    token_id, piece_buf, 256, 0, False,
  )
  if n > 0:
    return piece_buf.value.decode("utf-8", errors="replace")
  return ""


# ============================================================
# LlamaCppBackend
# ============================================================
class LlamaCppBackend:
    """High-level llama.cpp wrapper via ctypes."""

    def __init__(self):
        self._model = None
        self._ctx = None
        self._vocab = None
        self._n_ctx = 0
        self._n_threads = 4
        self._n_threads_batch = 4
        self._eos = 0
        self._sampler = None
        self._loaded = False
        self._decode_batch = None

    def config(self) -> dict:
        if not self._loaded:
            return {}
        return {
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "n_threads_batch": self._n_threads_batch,
            "n_vocab": _lib.llama_vocab_n_tokens(self._vocab),
        }

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------
    def load(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4,
             n_threads_batch: int = None) -> None:
        if n_threads_batch is None:
            n_threads_batch = n_threads
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_threads_batch = n_threads_batch

        # Set params via C helper (avoids ctypes struct mismatch).
        # The helper stores structs in static buffers; we memmove the raw bytes
        # into opaque ctypes.Structure instances so they're passed BY VALUE on the stack.
        _helper.override_model_params.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _helper.override_model_params(1, 0, 0)   # use_mmap=1, use_mlock=0, n_gpu_layers=0
        _helper.override_context_params.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32]
        _helper.override_context_params(n_ctx, n_threads, n_threads_batch)

        _helper.get_default_model_params.restype = ctypes.c_void_p
        _helper.get_default_context_params.restype = ctypes.c_void_p
        _helper.get_model_params_size.argtypes = []
        _helper.get_model_params_size.restype = ctypes.c_int32
        _helper.get_context_params_size.argtypes = []
        _helper.get_context_params_size.restype = ctypes.c_int32

        mptr = _helper.get_default_model_params()
        cptr = _helper.get_default_context_params()
        msize = _helper.get_model_params_size()
        csize = _helper.get_context_params_size()

        OpaqueParams = _lib._opaque_params_cls
        m_struct = OpaqueParams()
        c_struct = OpaqueParams()
        ctypes.memmove(ctypes.addressof(m_struct), mptr, msize)
        ctypes.memmove(ctypes.addressof(c_struct), cptr, csize)

        # Load model
        if isinstance(model_path, str):
            model_path = model_path.encode("utf-8")
        self._model = _lib.llama_model_load_from_file(model_path, m_struct)
        if not self._model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        self._vocab = _lib.llama_model_get_vocab(self._model)
        self._eos = _lib.llama_vocab_eos(self._vocab)

        # Create context
        self._ctx = _lib.llama_init_from_model(self._model, c_struct)
        if not self._ctx:
            _lib.llama_model_free(self._model)
            raise RuntimeError("Failed to create context")

        # Thread control
        _lib.llama_set_n_threads(self._ctx, n_threads, n_threads_batch)

        # Default greedy sampler
        self._sampler = _lib.llama_sampler_init_greedy()

        # Pre-allocate single-token decode batch (reused every step)
        self._decode_batch = _lib.llama_batch_init(1, 0, 1)
        self._decode_batch.n_seq_id[0] = 1
        self._decode_batch.seq_id[0][0] = 0

        self._loaded = True

    # ----------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------
    def _set_sampler(self, temperature: float = 0.0, top_k: int = 40,
                     top_p: float = 0.95, seed: int = 42,
                     grammar: Optional[str] = None) -> None:
        if self._sampler:
            _lib.llama_sampler_free(self._sampler)

        if grammar:
            self._sampler = _lib.llama_sampler_init_grammar(
                self._vocab, grammar.encode("utf-8"), b"")
        else:
            self._sampler = _lib.llama_sampler_init_greedy()

        if temperature > 0.0:
            t = _lib.llama_sampler_init_temp(temperature)
            _lib.llama_sampler_chain_add(self._sampler, t)
            if top_k > 0:
                k = _lib.llama_sampler_init_top_k(top_k)
                _lib.llama_sampler_chain_add(self._sampler, k)
            if top_p < 1.0:
                p = _lib.llama_sampler_init_top_p(top_p, 1)
                _lib.llama_sampler_chain_add(self._sampler, p)

    # ----------------------------------------------------------
    # Tokenize / Detokenize
    # ----------------------------------------------------------
    def tokenize(self, text: str, add_bos: bool = False) -> list[int]:
        if isinstance(text, str):
            text = text.encode("utf-8")
        buf = (ctypes.c_int32 * 1024)()
        n = _lib.llama_tokenize(self._vocab, text, len(text), buf, 1024, add_bos, True)
        if n < 0:
            buf = (ctypes.c_int32 * (-n))()
            n = _lib.llama_tokenize(self._vocab, text, len(text), buf, -n, add_bos, True)
        return list(buf[:n])

    def detokenize(self, tokens: list[int]) -> str:
        """Decode token IDs → text. LRU-cached per-token."""
        pieces = []
        vp = self._vocab
        for t in tokens:
            piece = _detokenize_cached(vp, t)
            if piece:
                pieces.append(piece)
        return "".join(pieces)

    def detokenize_batch(self, tokens: list[int]) -> str:
        """Decode using llama_detokenize (single C call, faster for large batches)."""
        if not tokens:
            return ""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        buf = ctypes.create_string_buffer(len(tokens) * 16)
        n = _lib.llama_detokenize(self._vocab, arr, len(tokens), buf, len(buf), True, True)
        if n > 0:
            return buf.value[:n].decode("utf-8", errors="replace")
        return self.detokenize(tokens)

    # ----------------------------------------------------------
    # Batch helpers
    # ----------------------------------------------------------
    def _make_batch(self, tokens: list[int], pos_offset: int = 0) -> LLamaBatch:
        """Create llama_batch. Reuses pre-allocated batch for single-token decode."""
        n = len(tokens)
        if n == 1:
            b = self._decode_batch
            b.n_tokens = 1
            b.token[0] = tokens[0]
            b.pos[0] = pos_offset
            b.n_seq_id[0] = 1
            b.seq_id[0][0] = 0
            b.logits[0] = 1  # request logits for this token
            return b
        # Prefill: allocate new batch
        batch = _lib.llama_batch_init(n, 0, 1)
        batch.n_tokens = n
        for i, tok in enumerate(tokens):
            batch.token[i] = tok
            batch.pos[i] = pos_offset + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 0  # no logits for intermediate tokens
        batch.logits[n - 1] = 1  # request logits for last token only
        return batch

    def _free_prefill_batch(self, batch: LLamaBatch) -> None:
        """Free a batch allocated for prefill (not the cached decode batch)."""
        if batch is not self._decode_batch:
            _lib.llama_batch_free(batch)

    # ----------------------------------------------------------
    # Prefill
    # ----------------------------------------------------------
    def prefill(self, tokens: list[int]) -> list[float]:
        n = len(tokens)
        if n > self._n_ctx:
            raise RuntimeError(f"Prompt too long: {n} > {self._n_ctx}")
        batch = self._make_batch(tokens, pos_offset=0)
        ret = _lib.llama_decode(self._ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        logits_ptr = _lib.llama_get_logits_ith(self._ctx, n - 1)
        n_vocab = _lib.llama_vocab_n_tokens(self._vocab)
        return [logits_ptr[i] for i in range(n_vocab)]

    # ----------------------------------------------------------
    # Generate
    # ----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
        add_bos: bool = False,
        seed: int = 42,
        grammar: Optional[str] = None,
    ) -> GenerateResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Reset KV cache
        mem = _lib.llama_get_memory(self._ctx)
        _lib.llama_memory_clear(mem, True)
        _lib.llama_synchronize(self._ctx)

        self._set_sampler(temperature=temperature, top_k=top_k, top_p=top_p,
                          seed=seed, grammar=grammar)
        prompt_tokens = self.tokenize(prompt, add_bos=add_bos)
        n_prompt = len(prompt_tokens)
        if n_prompt >= self._n_ctx:
            raise RuntimeError(f"Prompt ({n_prompt} tokens) exceeds context ({self._n_ctx})")

        # Prefill
        t0 = _time.time()
        self.prefill(prompt_tokens)
        t_prefill = _time.time()

        output_tokens: list[int] = []
        cur_pos = n_prompt

        for _ in range(max_tokens):
            if cur_pos >= self._n_ctx:
                break
            next_token = _lib.llama_sampler_sample(self._sampler, self._ctx, -1)
            if next_token == self._eos:
                break
            if stop_tokens and next_token in stop_tokens:
                output_tokens.append(next_token)
                break
            output_tokens.append(next_token)
            cur_pos += 1
            batch = self._make_batch([next_token], pos_offset=cur_pos - 1)
            ret = _lib.llama_decode(self._ctx, batch)
            if ret != 0:
                print(f"[WARNING] llama_decode failed: {ret}")
                break
            _lib.llama_sampler_reset(self._sampler)

        t_end = _time.time()
        t_decode = t_end - t_prefill
        text = self.detokenize_batch(output_tokens) if len(output_tokens) > 5 else self.detokenize(output_tokens)

        if len(output_tokens) >= max_tokens:
            stop_reason = "max_tokens"
        elif stop_tokens and output_tokens and output_tokens[-1] in stop_tokens:
            stop_reason = "stop_token"
        else:
            stop_reason = "eos"

        tps = len(output_tokens) / t_decode if t_decode > 0 else 0.0
        return GenerateResult(
            text=text, tokens=output_tokens, tokens_per_second=tps,
            prompt_tokens=n_prompt, stop_reason=stop_reason,
            time_prefill=t_prefill - t0, time_decode=t_decode, time_total=t_end - t0,
        )

    # ----------------------------------------------------------
    # Stream
    # ----------------------------------------------------------
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.95,
        stop_tokens: Optional[list[int]] = None,
        add_bos: bool = False,
        seed: int = 42,
        grammar: Optional[str] = None,
    ) -> Generator[int, None, GenerateResult]:
        """Streaming generator. Yields token IDs, final yield = GenerateResult."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        mem = _lib.llama_get_memory(self._ctx)
        _lib.llama_memory_clear(mem, True)
        _lib.llama_synchronize(self._ctx)

        self._set_sampler(temperature=temperature, top_k=top_k, top_p=top_p,
                          seed=seed, grammar=grammar)
        prompt_tokens = self.tokenize(prompt, add_bos=add_bos)
        n_prompt = len(prompt_tokens)
        if n_prompt >= self._n_ctx:
            raise RuntimeError(f"Prompt ({n_prompt}) exceeds context ({self._n_ctx})")

        t0 = _time.time()
        # Prefill
        batch = self._make_batch(prompt_tokens, pos_offset=0)
        ret = _lib.llama_decode(self._ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        t_prefill = _time.time()

        output_tokens: list[int] = []
        cur_pos = n_prompt
        stop_reason = "max_tokens"

        for _ in range(max_tokens):
            if cur_pos >= self._n_ctx:
                stop_reason = "max_tokens"
                break
            next_token = _lib.llama_sampler_sample(self._sampler, self._ctx, -1)
            if next_token == self._eos:
                stop_reason = "eos"
                break
            if stop_tokens and next_token in stop_tokens:
                output_tokens.append(next_token)
                stop_reason = "stop_token"
                break
            output_tokens.append(next_token)
            cur_pos += 1
            yield next_token
            batch = self._make_batch([next_token], pos_offset=cur_pos - 1)
            ret = _lib.llama_decode(self._ctx, batch)
            if ret != 0:
                print(f"[WARNING] llama_decode failed: {ret}")
                break
            _lib.llama_sampler_reset(self._sampler)

        t_end = _time.time()
        t_decode = t_end - t_prefill
        tps = len(output_tokens) / t_decode if t_decode > 0 else 0.0
        yield GenerateResult(
            text=self.detokenize_batch(output_tokens) if len(output_tokens) > 5 else self.detokenize(output_tokens),
            tokens=output_tokens, tokens_per_second=tps,
            prompt_tokens=n_prompt, stop_reason=stop_reason,
            time_prefill=t_prefill - t0, time_decode=t_decode, time_total=t_end - t0,
        )

    # ----------------------------------------------------------
    # TurboSparse (activation sparsity for MoE models)
    # ----------------------------------------------------------
    def turbosparse_enable(self, threshold: float = 0.1) -> None:
        """Enable TurboSparse activation pruning in MoE FFN matmuls.

        Args:
            threshold: magnitude threshold — activations with |a| < threshold
                       are zeroed, skipping their matmul contribution.
        """
        if not self._ctx:
            raise RuntimeError("No context loaded")
        _lib.llama_turbosparse_enable(self._ctx, threshold)

    def turbosparse_disable(self) -> None:
        """Disable TurboSparse activation pruning."""
        if not self._ctx:
            raise RuntimeError("No context loaded")
        _lib.llama_turbosparse_disable(self._ctx)

    def turbosparse_is_enabled(self) -> bool:
        return bool(_lib.llama_turbosparse_is_enabled(self._ctx))

    def turbosparse_get_threshold(self) -> float:
        return float(_lib.llama_turbosparse_get_threshold(self._ctx))

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    def free(self) -> None:
        if self._decode_batch:
            _lib.llama_batch_free(self._decode_batch)
            self._decode_batch = None
        if self._sampler:
            _lib.llama_sampler_free(self._sampler)
            self._sampler = None
        if self._ctx:
            _lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            _lib.llama_model_free(self._model)
            self._model = None
        self._loaded = False

    def close(self) -> None:
        """Alias for free()."""
        self.free()

    def set_turbosparse(self, enabled: bool, threshold: float = 0.1) -> None:
        """Enable/disable TurboSparse activation sparsity."""
        if enabled:
            _lib.llama_turbosparse_enable(None, ctypes.c_float(threshold))
        else:
            _lib.llama_turbosparse_disable(None)

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
