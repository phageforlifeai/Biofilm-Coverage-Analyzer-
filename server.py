#!/usr/bin/env python3
"""
server.py
-----------------
Standalone HTTP web UI & JSON API for the Biofilm Coverage Analyzer.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import analyzer as biofilm
import api as biofilm_api

ROOT = Path(__file__).resolve().parent
STATIC = ROOT
MAX_BODY_BYTES = max(1, int(os.environ.get("BIOFILM_MAX_UPLOAD_MB", "25"))) * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "BiofilmAnalyzer/1.0"

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200, extra: dict | None = None):
        self._send(code, json.dumps(obj, default=str).encode(), "application/json; charset=utf-8", extra=extra)

    def _read_json(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY_BYTES:
            self._json({"error": f"payload too large: {n} bytes (limit {MAX_BODY_BYTES})"}, 413)
            return None
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode() or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json({"error": "invalid JSON body"}, 400)
            return None
        if not isinstance(body, dict):
            self._json({"error": "JSON body must be an object"}, 400)
            return None
        return body

    def _file(self, fp: Path, ctype: str | None = None):
        fp = fp.resolve()
        if not fp.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = ctype or (mimetypes.guess_type(fp.name)[0] or "application/octet-stream")
        if ctype.startswith("text/") or ctype == "application/json":
            ctype += "; charset=utf-8"
        self._send(200, fp.read_bytes(), ctype)

    def _artifact(self, rel: str):
        target = (ROOT / rel).resolve()
        if not str(target).startswith(str(ROOT.resolve()) + os.sep):
            return self._json({"error": "outside workspace"}, 403)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        extra = {}
        if ctype.startswith("image/") or ctype == "application/pdf":
            extra["Content-Disposition"] = f'inline; filename="{target.name}"'
        return self._send(200, target.read_bytes(), ctype, extra)

    def log_message(self, fmt, *args):
        sys.stderr.write("[biofilm-web] " + (fmt % args) + "\n")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in {"/api/health", "/healthz"}:
            return self._json({"ok": True, "service": "biofilm-coverage-analyzer"})
        if path in {"/", "/index.html", "/biofilm", "/biofilm.html"}:
            return self._file(STATIC / "biofilm.html", "text/html; charset=utf-8")
        if path == "/api/biofilm/state":
            sample_dir = ROOT / "samples"
            has_samples = sample_dir.is_dir() and any(
                p.is_file() and p.suffix.lower() in biofilm_api.IMAGE_EXTS
                for p in sample_dir.iterdir()
            )
            return self._json({
                "images": biofilm_api.list_images(),
                "methods": list(biofilm.THRESHOLD_METHODS),
                "channels": list(biofilm.CHANNEL_MODES),
                "has_session_samples": has_samples,
                "max_upload_mb": MAX_BODY_BYTES // (1024 * 1024),
            })
        if path.startswith("/artifact/"):
            return self._artifact(path[len("/artifact/"):])
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_json()
        if body is None:
            return None

        if path == "/api/biofilm/upload":
            res, err = biofilm_api.save_upload(
                body.get("name") or "", body.get("content_b64") or "", body.get("group")
            )
            return self._json({"error": err}, 400) if err else self._json(res)
        if path == "/api/biofilm/delete":
            return self._json(biofilm_api.delete_image(body.get("name") or ""))
        if path == "/api/biofilm/groups":
            mapping = body.get("groups") if isinstance(body.get("groups"), dict) else {}
            return self._json(biofilm_api.set_groups(mapping))
        if path == "/api/biofilm/load_samples":
            n = biofilm_api.load_samples_from(ROOT / "samples")
            return self._json({"ok": True, "loaded": n, "images": biofilm_api.list_images()})
        if path == "/api/biofilm/analyze":
            try:
                return self._json(biofilm_api.run_analysis(body))
            except Exception as exc:
                return self._json({"error": str(exc)}, 400)
        if path == "/api/biofilm/export_csv":
            csv_bytes = biofilm_api.export_csv_bytes(body)
            return self._send(
                200, csv_bytes, "text/csv; charset=utf-8",
                {"Content-Disposition": 'attachment; filename="biofilm_coverage.csv"'},
            )
        if path == "/api/biofilm/export_figure":
            res, err = biofilm_api.export_figure_artifacts(body)
            return self._json({"error": err}, 400) if err else self._json(res)
        return self._json({"error": "not found"}, 404)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Biofilm Coverage Analyzer web UI")
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = p.parse_args(argv)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Biofilm Coverage Analyzer running on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
