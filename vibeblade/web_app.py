"""
VibeBlade Web App — FastAPI application factory for the ChatGPT-like web UI.

Created as a separate importable module so uvicorn can reload it:
    uvicorn vibeblade.web_app:app --reload

Reads VIBEBLADE_BACKEND_URL from environment (set by CLI or defaults to
http://localhost:8000).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `web.app` is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from web.app import app as _app, DEFAULT_BACKEND_URL  # noqa: E402

# Override backend URL from environment if set
_backend_url = os.environ.get("VIBEBLADE_BACKEND_URL", DEFAULT_BACKEND_URL)

import web.app as _web_module  # noqa: E402
_web_module.DEFAULT_BACKEND_URL = _backend_url

app = _app
