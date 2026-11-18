"""
biofilm/api.py
--------------
Web API helpers for the standalone Biofilm Coverage Analyzer.
"""

from __future__ import annotations

import base64
import io
import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

import analyzer as biofilm

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
BIOFILM_DIR = OUTPUT_DIR / "biofilm_uploads"
BIOFILM_OUT = OUTPUT_DIR / "biofilm_exports"
META_FILE = BIOFILM_DIR / "_groups.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._ -]{1,120}$")
_LOCK = threading.Lock()


def sanitize_filename(raw_name: str) -> str:
    """Turn any user filename (with (), +, #, spaces, unicode, etc.) into a safe basename."""
    base = Path((raw_name or "").replace("\\", "/")).name.strip()
    if not base:
        return ""
    p = Path(base)
    ext = p.suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", p.stem).strip(" .") or "image"
    stem = stem[:100]
    return f"{stem}{ext}"


def _ensure_dirs() -> None:
    BIOFILM_DIR.mkdir(parents=True, exist_ok=True)
    BIOFILM_OUT.mkdir(parents=True, exist_ok=True)


def _infer_group(filename: str) -> str:
    stem = Path(filename).stem.split(".")[0]
    prefix = "".join(ch for ch in stem if not ch.isdigit()).strip("_- ")
    return prefix or "Sample"


def _load_meta() -> dict[str, str]:
    if not META_FILE.is_file():
        return {}
    try:
        data = json.loads(META_FILE.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_meta(meta: dict[str, str]) -> None:
    _ensure_dirs()
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_images() -> list[dict[str, Any]]:
    with _LOCK:
        _ensure_dirs()
        meta = _load_meta()
        out = []
        for p in sorted(BIOFILM_DIR.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            w = h = 0
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                pass
            grp = meta.get(p.name) or _infer_group(p.name)
            out.append({
                "name": p.name,
                "group": grp,
                "size_kb": round(p.stat().st_size / 1024.0, 1),
                "width": w,
                "height": h,
            })
        return out


def save_upload(name: str, b64: str, group: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    clean_name = sanitize_filename(name)
    if not clean_name:
        sys.stderr.write(f"[biofilm] upload rejected: empty filename ({name!r})\n")
        return None, "bad filename"
    if Path(clean_name).suffix.lower() not in IMAGE_EXTS:
        msg = f"unsupported format {Path(clean_name).suffix!r}; allowed: {', '.join(sorted(IMAGE_EXTS))}"
        sys.stderr.write(f"[biofilm] upload rejected ({clean_name}): {msg}\n")
        return None, msg
    if not b64 or not isinstance(b64, str):
        sys.stderr.write(f"[biofilm] upload rejected ({clean_name}): empty content\n")
        return None, "empty image content"
    if "," in b64[:80] and "base64" in b64[:80]:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64.strip())
    except Exception as exc:
        sys.stderr.write(f"[biofilm] upload rejected ({clean_name}): bad base64: {exc}\n")
        return None, f"bad base64: {exc}"
    if not raw:
        sys.stderr.write(f"[biofilm] upload rejected ({clean_name}): 0-byte payload\n")
        return None, "empty file"
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except Exception as exc:
        sys.stderr.write(f"[biofilm] upload rejected ({clean_name}): unreadable image: {exc}\n")
        return None, f"unreadable image file: {exc}"

    with _LOCK:
        _ensure_dirs()
        dest = (BIOFILM_DIR / clean_name).resolve()
        if not str(dest).startswith(str(BIOFILM_DIR.resolve())):
            return None, "invalid path"
        dest.write_bytes(raw)
        meta = _load_meta()
        meta[clean_name] = (group or "").strip() or _infer_group(clean_name)
        _save_meta(meta)
    return {"ok": True, "name": clean_name, "group": meta[clean_name]}, None


def set_groups(mapping: dict[str, str]) -> dict[str, Any]:
    with _LOCK:
        _ensure_dirs()
        meta = _load_meta()
        for k, v in mapping.items():
            k_clean = (k or "").strip()
            v_clean = (v or "").strip() or "Default"
            if k_clean and (BIOFILM_DIR / k_clean).is_file():
                meta[k_clean] = v_clean
        _save_meta(meta)
    return {"ok": True, "images": list_images()}


def delete_image(name: str) -> dict[str, Any]:
    name = (name or "").strip()
    with _LOCK:
        _ensure_dirs()
        meta = _load_meta()
        if name == "__all__":
            for p in BIOFILM_DIR.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    p.unlink()
            meta.clear()
            _save_meta(meta)
            return {"ok": True}
        target = (BIOFILM_DIR / name).resolve()
        if str(target).startswith(str(BIOFILM_DIR.resolve())) and target.is_file():
            target.unlink()
        meta.pop(name, None)
        _save_meta(meta)
    return {"ok": True}


def load_samples_from(src_dir: Path) -> int:
    if not src_dir.is_dir():
        return 0
    count = 0
    with _LOCK:
        _ensure_dirs()
        meta = _load_meta()
        for p in sorted(src_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and SAFE_NAME.match(p.name):
                dest = BIOFILM_DIR / p.name
                shutil.copy2(p, dest)
                meta.setdefault(p.name, _infer_group(p.name))
                count += 1
        _save_meta(meta)
    return count


def parse_params(body: dict[str, Any]) -> biofilm.AnalysisParams:
    def _int(k: str, d: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(body.get(k, d))))
        except Exception:
            return d

    def _flt(k: str, d: float, lo: float, hi: float) -> float:
        try:
            return max(lo, min(hi, float(body.get(k, d))))
        except Exception:
            return d

    channel = str(body.get("channel") or "auto").lower()
    if channel not in biofilm.CHANNEL_MODES:
        channel = "auto"
    method = str(body.get("method") or "otsu").lower()
    if method not in biofilm.THRESHOLD_METHODS:
        method = "otsu"

    roi_raw = body.get("roi")
    roi = None
    if isinstance(roi_raw, (list, tuple)) and len(roi_raw) == 4:
        try:
            vals = tuple(max(0.0, min(100.0, float(x))) for x in roi_raw)
            if abs(vals[2] - vals[0]) >= 2.0 and abs(vals[3] - vals[1]) >= 2.0:
                roi = vals  # type: ignore[assignment]
        except Exception:
            roi = None

    cluster_mode = str(body.get("cluster_mode") or "separate").lower()
    if cluster_mode not in biofilm.CLUSTER_MODES:
        cluster_mode = "separate"

    return biofilm.AnalysisParams(
        channel=channel,
        method=method,
        unmix_dual=bool(body.get("unmix_dual", True)),
        manual_threshold=_int("manual_threshold", 100, 0, 255),
        manual_threshold_green=_int("manual_threshold_green", 45, 0, 255),
        manual_threshold_orange=_int("manual_threshold_orange", 60, 0, 255),
        threshold_offset=_int("threshold_offset", 0, -100, 100),
        bg_radius=_int("bg_radius", 35, 0, 150),
        smooth_sigma=_flt("smooth_sigma", 0.0, 0.0, 5.0),
        min_particle_px=_int("min_particle_px", 15, 0, 10000),
        fill_holes=bool(body.get("fill_holes", False)),
        invert=bool(body.get("invert", False)),
        exclude_scalebar=bool(body.get("exclude_scalebar", True)),
        scalebar_w_pct=_flt("scalebar_w_pct", 15.0, 0.0, 50.0),
        scalebar_h_pct=_flt("scalebar_h_pct", 12.0, 0.0, 50.0),
        roi=roi,
        separate_morphotypes=bool(body.get("separate_morphotypes", True)),
        rod_min_area_px=_int("rod_min_area_px", 230, 30, 3000),
        rod_min_length_px=_flt("rod_min_length_px", 28.0, 8.0, 200.0),
        cluster_mode=cluster_mode,
        cluster_min_area_px=_int("cluster_min_area_px", 1800, 300, 20000),
    )


def _gather_items(selected_names: list[str] | None = None) -> list[tuple[Path, str, str]]:
    imgs = list_images()
    if selected_names:
        allow = set(selected_names)
        imgs = [x for x in imgs if x["name"] in allow]
    return [
        (BIOFILM_DIR / x["name"], x["name"], x["group"])
        for x in imgs
        if (BIOFILM_DIR / x["name"]).is_file()
    ]


def run_analysis(body: dict[str, Any]) -> dict[str, Any]:
    params = parse_params(body)
    view = str(body.get("view") or "morphotype")
    if view not in {"morphotype", "overlay", "mask", "original"}:
        view = "morphotype"
    selected = body.get("names")
    items = _gather_items(selected if isinstance(selected, list) and selected else None)
    if not items:
        return {"params": {}, "images": [], "groups": [], "overall": biofilm.summarize_group([])}

    preview_name = body.get("preview_name")
    render_all = bool(body.get("render_all", False))

    if render_all:
        return biofilm.analyze_batch(items, params, render_view=view, max_preview_width=720)

    batch = biofilm.analyze_batch(items, params, render_view=None)
    target_name = preview_name or items[0][1]
    for src, nm, grp in items:
        if nm == target_name:
            res = biofilm.analyze_image(
                src, params, name=nm, group=grp, render_view=view, max_preview_width=960
            )
            for idx, r in enumerate(batch["images"]):
                if r["name"] == nm:
                    batch["images"][idx] = res
                    break
            break
    return batch


def export_csv_bytes(body: dict[str, Any]) -> bytes:
    params = parse_params(body)
    items = _gather_items(body.get("names") if isinstance(body.get("names"), list) else None)
    batch = biofilm.analyze_batch(items, params, render_view=None)
    return biofilm.results_to_csv(batch).encode("utf-8")


def export_figure_artifacts(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    params = parse_params(body)
    items = _gather_items(body.get("names") if isinstance(body.get("names"), list) else None)
    if not items:
        return None, "no images to plot"
    batch = biofilm.analyze_batch(items, params, render_view=None)
    title = str(body.get("title") or "Biofilm Surface Coverage").strip() or "Biofilm Surface Coverage"
    _ensure_dirs()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    png_path = BIOFILM_OUT / f"biofilm_coverage_{stamp}.png"
    pdf_path = BIOFILM_OUT / f"biofilm_coverage_{stamp}.pdf"
    biofilm.export_summary_figure(batch, png_path, pdf_path, title=title)
    rel_png = str(png_path.relative_to(ROOT))
    rel_pdf = str(pdf_path.relative_to(ROOT))
    return {"ok": True, "png": rel_png, "pdf": rel_pdf, "batch": batch}, None
