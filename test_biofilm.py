#!/usr/bin/env python3
"""
test_biofilm.py
-----------------------
Verification suite for the ImageJ-like SYTO 9 (Green) & OYP (Orange) biofilm
coverage analyzer (core + exports + web API).
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import analyzer as biofilm  # noqa: E402

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
results: list[tuple[bool, str]] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    results.append((bool(cond), label))
    print(f"{PASS if cond else FAIL}  {label}" + (f"\n        {detail}" if detail and not cond else ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_core() -> None:
    print("\n[1] core biofilm thresholding, SYTO 9 / OYP detection & known-coverage truth")
    # 1) Single-channel grayscale/white synthetic 200x200 image with exact 25.00% bright block
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:100, :100, :] = 220
    img[180:198, 172:198, :] = 255  # fake bottom-right scale bar

    r_keep = biofilm.analyze_image(
        img, biofilm.AnalysisParams(bg_radius=0, min_particle_px=0, exclude_scalebar=False)
    )
    check(r_keep["coverage_pct"] > 26.0, "keeping scalebar inflates coverage above 25%", str(r_keep))

    r_excl = biofilm.analyze_image(
        img, biofilm.AnalysisParams(bg_radius=0, min_particle_px=0, exclude_scalebar=True)
    )
    expected = 100.0 * 10000 / (200 * 200 - 30 * 24)
    check(abs(r_excl["coverage_pct"] - expected) < 0.05,
          f"scale-bar exclusion gives exact expected {expected:.2f}%", f"got {r_excl['coverage_pct']}")

    for method in biofilm.THRESHOLD_METHODS:
        if method == "manual":
            continue
        rm = biofilm.analyze_image(
            img, biofilm.AnalysisParams(method=method, bg_radius=0, min_particle_px=0, exclude_scalebar=True)
        )
        check(abs(rm["coverage_pct"] - expected) < 0.5, f"method={method} recovers ground truth", str(rm["coverage_pct"]))

    # 2) Single-channel Green (SYTO 9 only)
    g_only = np.zeros((200, 200, 3), dtype=np.uint8)
    g_only[20:70, 20:70, 1] = 210  # 2500 px = 6.25%
    rg = biofilm.analyze_image(
        g_only, biofilm.AnalysisParams(channel="auto", bg_radius=0, min_particle_px=0, exclude_scalebar=False)
    )
    check(rg["channel_used"] == "green_syto9", "auto-detects green_syto9 when only green present", rg["channel_used"])
    check(abs(rg["green_syto9_pct"] - 6.25) < 0.2 and rg["orange_oyp_pct"] is None,
          "green-only reports Green=6.25% and Orange=None", str(rg))

    # 3) Single-channel Orange (OYP only)
    o_only = np.zeros((200, 200, 3), dtype=np.uint8)
    o_only[20:70, 20:70, 0] = 240
    o_only[20:70, 20:70, 1] = 110
    ro = biofilm.analyze_image(
        o_only, biofilm.AnalysisParams(channel="auto", bg_radius=0, min_particle_px=0, exclude_scalebar=False)
    )
    check(ro["channel_used"] == "orange_oyp", "auto-detects orange_oyp when only orange present", ro["channel_used"])
    check(abs(ro["orange_oyp_pct"] - 6.25) < 0.2 and ro["green_syto9_pct"] is None,
          "orange-only reports Orange=6.25% and Green=None", str(ro))

    # 4) Dual-channel SYTO 9 (Green) + OYP (Orange)
    dual = np.zeros((200, 200, 3), dtype=np.uint8)
    dual[10:60, 10:60, 1] = 210                          # 6.25% Green (SYTO 9)
    dual[10:60, 80:130, 0] = 240; dual[10:60, 80:130, 1] = 100  # 6.25% Orange (OYP)
    r_dual = biofilm.analyze_image(
        dual, biofilm.AnalysisParams(channel="auto", method="otsu", bg_radius=0, min_particle_px=0, exclude_scalebar=False)
    )
    check(r_dual["channel_used"] == "dual_syto9_oyp", "auto-detects dual_syto9_oyp when both present", r_dual["channel_used"])
    check(abs(r_dual["coverage_pct"] - 12.5) < 0.2, "dual total coverage = 12.5%", str(r_dual))
    check(abs(r_dual["green_syto9_pct"] - 6.25) < 0.2 and abs(r_dual["orange_oyp_pct"] - 6.25) < 0.2,
          "unmixed dual splits Green=6.25% and Orange=6.25% cleanly", str(r_dual))

    # 5) Group Mean ± SD calculation (ddof=1)
    s = biofilm.summarize_group([10.0, 20.0, 30.0])
    check(abs(s["mean_pct"] - 20.0) < 1e-4 and abs(s["sd_pct"] - 10.0) < 1e-4,
          "group Mean=20.0 ± SD=10.0 (ddof=1)", str(s))
    check(abs(s["sem_pct"] - (10.0 / math.sqrt(3))) < 1e-3, "SEM = SD/sqrt(n)", str(s))

    # 6) Morphotype differentiation (Large Rods vs Small Coccobacilli) in single & dual channel
    morph = np.zeros((200, 200, 3), dtype=np.uint8)
    # Green large rod: 12 x 60 = 720 px (1.80%)
    morph[20:32, 20:80, 1] = 220
    # Green small coccobacillus: 8 x 10 = 80 px (0.20%)
    morph[100:108, 30:40, 1] = 220
    # Orange large rod: 12 x 60 = 720 px (1.80%)
    morph[50:62, 100:160, 0] = 240; morph[50:62, 100:160, 1] = 90
    # Orange small coccobacillus: 8 x 10 = 80 px (0.20%)
    morph[140:148, 120:130, 0] = 240; morph[140:148, 120:130, 1] = 90
    rmorph = biofilm.analyze_image(
        morph,
        biofilm.AnalysisParams(
            channel="auto", method="otsu", bg_radius=0, min_particle_px=10,
            exclude_scalebar=False, separate_morphotypes=True,
            rod_min_area_px=230, cluster_mode="separate",
        ),
        render_view="morphotype",
    )
    check(rmorph["channel_used"] == "dual_syto9_oyp", "dual morphotype image detected as dual_syto9_oyp")
    check(abs(rmorph["green_rods_pct"] - 1.80) < 0.05 and abs(rmorph["green_cocci_pct"] - 0.20) < 0.05,
          "dual morphotype splits Green Rods (1.80%) and Green Cocci (0.20%)", str(rmorph))
    check(abs(rmorph["orange_rods_pct"] - 1.80) < 0.05 and abs(rmorph["orange_cocci_pct"] - 0.20) < 0.05,
          "dual morphotype splits Orange Rods (1.80%) and Orange Cocci (0.20%)", str(rmorph))
    check(abs(rmorph["rods_pct"] - 3.60) < 0.05 and abs(rmorph["cocci_pct"] - 0.40) < 0.05,
          "combined Rods (3.60%) and Cocci (0.40%) match", str(rmorph))


def test_exports() -> None:
    print("\n[2] batch, CSV, and 300 DPI PNG + PDF export")
    imgs = []
    for i, frac in enumerate([20, 30, 40]):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:frac, :50, 1] = 220                 # Green SYTO 9
        arr[:frac, 50:, 0] = 240; arr[:frac, 50:, 1] = 90  # Orange OYP
        imgs.append((Image.fromarray(arr), f"field_{i}.png", "Biofilm_A"))

    batch = biofilm.analyze_batch(
        imgs, biofilm.AnalysisParams(channel="auto", bg_radius=0, min_particle_px=0, exclude_scalebar=False),
        render_view="overlay",
    )
    check(len(batch["images"]) == 3, "batch analyzed 3 dual-channel images")
    g = batch["groups"][0]
    check(g["has_both_channels"], "batch group tracks both Green (SYTO 9) and Orange (OYP)")
    check(abs(g["mean_pct"] - 30.0) < 0.2 and abs(g["sd_pct"] - 10.0) < 0.2, "batch total = 30.0 ± 10.0%", str(g))
    check(abs(g["green_syto9_summary"]["mean_pct"] - 15.0) < 0.2, "batch Green = 15.0%", str(g["green_syto9_summary"]))
    check(abs(g["orange_oyp_summary"]["mean_pct"] - 15.0) < 0.2, "batch Orange = 15.0%", str(g["orange_oyp_summary"]))

    csv_text = biofilm.results_to_csv(batch)
    check("Green_SYTO9_Pct" in csv_text and "Orange_OYP_Pct" in csv_text and "30.00 ± 10.00%" in csv_text,
          "CSV includes Green/Orange columns & Mean ± SD summaries")

    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "fig.png"
        pdf = Path(td) / "fig.pdf"
        biofilm.export_summary_figure(batch, png, pdf)
        check(png.is_file() and png.stat().st_size > 10_000, "exported 300 DPI PNG")
        check(pdf.is_file() and pdf.stat().st_size > 2_000, "exported vector PDF")


def test_http() -> None:
    print("\n[3] web server /biofilm & /api/biofilm/* endpoints")
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(30):
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.25)

        with urllib.request.urlopen(base + "/biofilm", timeout=5) as r:
            html = r.read().decode("utf-8")
            check(r.status == 200 and "SYTO 9" in html and "OYP" in html, "GET / & /biofilm return SYTO 9 & OYP HTML UI")

        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:25, :, 1] = 220
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        def post_json(path: str, payload: dict) -> tuple[int, dict, bytes]:
            req = urllib.request.Request(
                base + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                ct = resp.headers.get("Content-Type", "")
                return resp.status, (json.loads(raw) if "json" in ct else {}), raw

        st, res, _ = post_json("/api/biofilm/upload", {"name": "__test_bf.png", "content_b64": b64, "group": "TestGrp"})
        check(st == 200 and res.get("ok"), "POST /api/biofilm/upload succeeds")

        # Test filename with (), +, # (common from microscopes/downloads)
        st2, res2, _ = post_json("/api/biofilm/upload", {"name": "Control (1)+SYTO9 #2.png", "content_b64": b64})
        check(st2 == 200 and res2.get("ok") and bool(res2.get("name")),
              "POST /api/biofilm/upload sanitizes () + # filenames", str(res2))
        if res2.get("name"):
            post_json("/api/biofilm/delete", {"name": res2["name"]})

        # Test 16-bit grayscale TIFF handling
        arr16 = np.zeros((100, 100), dtype=np.uint16)
        arr16[:25, :] = 50000
        r16 = biofilm.analyze_image(arr16, biofilm.AnalysisParams(bg_radius=0, min_particle_px=0, exclude_scalebar=False))
        check(abs(r16["coverage_pct"] - 25.0) < 0.2, "16-bit image scales cleanly to 25.0%", str(r16))

        st, res, _ = post_json("/api/biofilm/analyze", {
            "channel": "auto", "method": "otsu", "bg_radius": 0, "min_particle_px": 0,
            "exclude_scalebar": False, "names": ["__test_bf.png"], "preview_name": "__test_bf.png",
        })
        check(st == 200 and abs(res["images"][0]["green_syto9_pct"] - 25.0) < 0.2,
              "POST /api/biofilm/analyze returns Green (SYTO 9) 25.0%", str(res.get("images")))

        st, _, csv_raw = post_json("/api/biofilm/export_csv", {
            "channel": "auto", "bg_radius": 0, "min_particle_px": 0, "exclude_scalebar": False,
            "names": ["__test_bf.png"],
        })
        check(st == 200 and b"__test_bf.png" in csv_raw, "POST /api/biofilm/export_csv downloads CSV")

        st, fig_res, _ = post_json("/api/biofilm/export_figure", {
            "channel": "auto", "bg_radius": 0, "min_particle_px": 0, "exclude_scalebar": False,
            "names": ["__test_bf.png"],
        })
        check(st == 200 and fig_res.get("ok") and fig_res.get("png"), "POST /api/biofilm/export_figure produces PNG + PDF")

        post_json("/api/biofilm/delete", {"name": "__test_bf.png"})
        for rel in (fig_res.get("png"), fig_res.get("pdf")):
            if rel:
                p = ROOT / rel
                if p.is_file():
                    p.unlink()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main() -> int:
    test_core()
    test_exports()
    test_http()
    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 62}\n{passed}/{total} biofilm checks passed\n{'=' * 62}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
