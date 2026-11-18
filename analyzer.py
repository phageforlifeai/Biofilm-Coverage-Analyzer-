#!/usr/bin/env python3
"""
agent/biofilm.py
----------------
ImageJ-compatible biofilm coverage area analyzer (% area + Mean ± SD)
for SYTO 9 (Green) and OYP (Orange) fluorescence microscopy, including
morphotype differentiation between:
  - Large rod-shaped bacteria (Bacilli)
  - Smaller coccobacilli
  - Optional dense unresolved microcolony clusters

Supports single-channel (SYTO 9 Green OR OYP Orange OR Grayscale) and
dual-channel (SYTO 9 + OYP) images with automatic 1-vs-2 channel detection,
spectral unmixing, rolling-ball background subtraction, 7 ImageJ Auto-Threshold
methods, scale-bar exclusion, CSV export, and 300 DPI PNG + PDF plotting.

CLI usage:
    python -m agent.biofilm /path/to/images/*.jpg --group "Sample A" --morphotypes --csv out.csv --plot out.png
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Channel overlay colors (SYTO 9 green & OYP orange)
COLOR_GREEN = (0, 230, 100)       # SYTO 9
COLOR_ORANGE = (255, 140, 20)     # OYP
COLOR_BOTH = (255, 235, 70)       # Overlap
COLOR_GRAY = (0, 210, 255)        # Generic grayscale

# Morphotype overlay colors
COLOR_ROD = (0, 215, 255)           # Cyan: Large rods (single-channel)
COLOR_COCCI = (255, 195, 45)        # Gold: Small coccobacilli (single-channel)
COLOR_CLUSTER = (235, 80, 210)      # Magenta: Dense unresolved clusters

COLOR_GREEN_ROD = (0, 230, 120)     # Vivid emerald: Green (SYTO 9) Rods
COLOR_GREEN_COCCI = (165, 255, 90)  # Bright lime: Green (SYTO 9) Coccobacilli
COLOR_ORANGE_ROD = (255, 110, 20)   # Vivid orange-red: Orange (OYP) Rods
COLOR_ORANGE_COCCI = (255, 210, 70) # Warm amber: Orange (OYP) Coccobacilli

THRESHOLD_METHODS = (
    "otsu",
    "li",
    "isodata",
    "triangle",
    "moments",
    "yen",
    "huang",
    "manual",
)

CHANNEL_MODES = (
    "auto",
    "dual_syto9_oyp",
    "green_syto9",
    "orange_oyp",
    "luminance",
)

CLUSTER_MODES = (
    "separate",  # 3rd class: report dense unresolved clusters separately
    "cocci",     # assign dense clusters to coccobacilli
    "rods",      # assign dense clusters to rods
    "off",       # strict 2-class split by size/length only
)


@dataclass
class AnalysisParams:
    channel: str = "auto"                # auto | dual_syto9_oyp | green_syto9 | orange_oyp | luminance
    method: str = "li"                   # otsu | li | isodata | triangle | moments | yen | huang | manual
    unmix_dual: bool = True              # reduce orange RGB bleed-through into green & vice versa
    manual_threshold: int = 100          # 0..255 (single-channel manual)
    manual_threshold_green: int = 45     # 0..255 (dual SYTO 9 green manual)
    manual_threshold_orange: int = 60    # 0..255 (dual OYP orange manual)
    threshold_offset: int = 0            # -100..+100 added to auto threshold
    bg_radius: int = 35                  # rolling-ball radius in px (0 = disabled)
    smooth_sigma: float = 0.0            # Gaussian blur sigma before thresholding
    min_particle_px: int = 15            # minimum connected-component area in px (0 = keep all)
    fill_holes: bool = False             # fill internal holes in segmented objects
    invert: bool = False                 # dark biofilm on bright background (brightfield)
    exclude_scalebar: bool = True        # exclude bottom-right scale bar corner
    scalebar_w_pct: float = 15.0         # width % from right edge to exclude
    scalebar_h_pct: float = 12.0         # height % from bottom edge to exclude
    roi: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) in 0..100 %
    # Morphotype differentiation (Large Rods vs Small Coccobacilli)
    separate_morphotypes: bool = True
    rod_min_area_px: int = 230           # min pixel area for a large rod
    rod_min_length_px: float = 28.0      # min major axis length (px) for elongated rod
    cluster_mode: str = "separate"       # separate | cocci | rods | off
    cluster_min_area_px: int = 1800      # large blob area cutoff for unresolved clusters


# ---------------------------------------------------------------------------
# ImageJ Auto-Threshold implementations (operating on uint8 0..255 arrays)
# ---------------------------------------------------------------------------

def _histogram(img: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    vals = img[valid_mask] if valid_mask is not None else img.ravel()
    if vals.size == 0:
        return np.zeros(256, dtype=np.float64)
    return np.bincount(vals.astype(np.uint8), minlength=256).astype(np.float64)


def threshold_otsu(hist: np.ndarray) -> int:
    total = hist.sum()
    if total <= 0:
        return 127
    bins = np.arange(256, dtype=np.float64)
    sum_total = np.dot(bins, hist)
    w_b = 0.0
    sum_b = 0.0
    max_var = -1.0
    best = 0
    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            best = t
    return int(best)


def threshold_isodata(hist: np.ndarray) -> int:
    """Ridler-Calvard / ImageJ 'Default' iterative intermeans."""
    total = hist.sum()
    if total <= 0:
        return 127
    bins = np.arange(256, dtype=np.float64)
    t = int(round(np.dot(bins, hist) / total))
    for _ in range(100):
        lo = hist[: t + 1]
        hi = hist[t + 1 :]
        w_lo = lo.sum()
        w_hi = hi.sum()
        if w_lo == 0 or w_hi == 0:
            break
        m_lo = np.dot(bins[: t + 1], lo) / w_lo
        m_hi = np.dot(bins[t + 1 :], hi) / w_hi
        new_t = int(round(0.5 * (m_lo + m_hi)))
        if new_t == t:
            break
        t = new_t
    return int(np.clip(t, 0, 255))


def threshold_triangle(hist: np.ndarray) -> int:
    """Zack et al. (1977) Triangle method."""
    nz = np.nonzero(hist > 0)[0]
    if nz.size < 2:
        return int(nz[0]) if nz.size == 1 else 127
    min_b, max_b = int(nz[0]), int(nz[-1])
    peak = int(np.argmax(hist))
    if (peak - min_b) < (max_b - peak):
        x1, y1 = float(peak), float(hist[peak])
        x2, y2 = float(max_b), 0.0
        xs = np.arange(peak, max_b + 1, dtype=np.float64)
        ys = hist[peak : max_b + 1]
        dx, dy = x2 - x1, y2 - y1
        denom = math.hypot(dx, dy) or 1.0
        dists = np.abs(dy * (xs - x1) - dx * (ys - y1)) / denom
        best = peak + int(np.argmax(dists))
    else:
        x1, y1 = float(min_b), 0.0
        x2, y2 = float(peak), float(hist[peak])
        xs = np.arange(min_b, peak + 1, dtype=np.float64)
        ys = hist[min_b : peak + 1]
        dx, dy = x2 - x1, y2 - y1
        denom = math.hypot(dx, dy) or 1.0
        dists = np.abs(dy * (xs - x1) - dx * (ys - y1)) / denom
        best = min_b + int(np.argmax(dists))
    return int(np.clip(best, 0, 255))


def threshold_li(hist: np.ndarray) -> int:
    """Li & Tam (1998) minimum cross-entropy thresholding."""
    total = hist.sum()
    if total <= 0:
        return 127
    bins = np.arange(256, dtype=np.float64)
    mean_all = np.dot(bins, hist) / total
    t_old = mean_all
    for _ in range(200):
        idx = int(np.clip(math.floor(t_old), 0, 254))
        w_lo = hist[: idx + 1].sum()
        w_hi = total - w_lo
        if w_lo <= 0 or w_hi <= 0:
            break
        m_lo = max(np.dot(bins[: idx + 1], hist[: idx + 1]) / w_lo, 1e-6)
        m_hi = max(np.dot(bins[idx + 1 :], hist[idx + 1 :]) / w_hi, 1e-6)
        if abs(m_hi - m_lo) < 1e-9:
            break
        t_new = (m_hi - m_lo) / (math.log(m_hi) - math.log(m_lo))
        if abs(t_new - t_old) < 0.1:
            t_old = t_new
            break
        t_old = t_new
    return int(np.clip(round(t_old), 0, 255))


def threshold_moments(hist: np.ndarray) -> int:
    """Tsai (1985) moment-preserving thresholding."""
    total = hist.sum()
    if total <= 0:
        return 127
    p = hist / total
    bins = np.arange(256, dtype=np.float64)
    m0 = 1.0
    m1 = np.dot(bins, p)
    m2 = np.dot(bins ** 2, p)
    m3 = np.dot(bins ** 3, p)
    cd = m0 * m2 - m1 * m1
    if abs(cd) < 1e-12:
        return int(round(m1))
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (m0 * -m3 + m2 * m1) / cd
    disc = c1 * c1 - 4.0 * c0
    if disc < 0:
        return int(round(m1))
    z0 = 0.5 * (-c1 - math.sqrt(disc))
    z1 = 0.5 * (-c1 + math.sqrt(disc))
    if abs(z1 - z0) < 1e-9:
        return int(round(m1))
    p0 = float(np.clip((z1 - m1) / (z1 - z0), 0.0, 1.0))
    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, p0))
    return int(np.clip(idx, 0, 255))


def threshold_yen(hist: np.ndarray) -> int:
    """Yen et al. (1995) maximum correlation criterion."""
    total = hist.sum()
    if total <= 0:
        return 127
    p = hist / total
    p1 = np.cumsum(p)
    p2 = 1.0 - p1
    p_sq = np.cumsum(p * p)
    p_sq_rev = np.cumsum((p * p)[::-1])[::-1]
    p_sq_hi = np.empty(256, dtype=np.float64)
    p_sq_hi[:-1] = p_sq_rev[1:]
    p_sq_hi[-1] = 0.0

    best_t = 0
    best_val = -1e18
    for t in range(255):
        if p1[t] <= 1e-12 or p2[t] <= 1e-12 or p_sq[t] <= 1e-18 or p_sq_hi[t] <= 1e-18:
            continue
        crit = math.log((p1[t] * p2[t]) ** 2) - math.log(p_sq[t] * p_sq_hi[t])
        if crit > best_val:
            best_val = crit
            best_t = t
    return int(best_t)


def threshold_huang(hist: np.ndarray) -> int:
    """Huang & Wang (1995) fuzzy thresholding (ImageJ 'Huang')."""
    nz = np.nonzero(hist > 0)[0]
    if nz.size < 2:
        return int(nz[0]) if nz.size == 1 else 127
    first, last = int(nz[0]), int(nz[-1])
    c = float(last - first) or 1.0
    bins = np.arange(256, dtype=np.float64)
    w_cum = np.cumsum(hist)
    s_cum = np.cumsum(bins * hist)
    total_w = w_cum[last]
    total_s = s_cum[last]

    best_t = first
    best_ent = 1e18
    for t in range(first, last):
        w0 = w_cum[t]
        w1 = total_w - w0
        if w0 <= 0 or w1 <= 0:
            continue
        mu0 = s_cum[t] / w0
        mu1 = (total_s - s_cum[t]) / w1

        idx0 = np.arange(first, t + 1, dtype=np.float64)
        u0 = np.clip(1.0 / (1.0 + np.abs(idx0 - mu0) / c), 1e-9, 1.0 - 1e-9)
        s0 = -np.sum(hist[first : t + 1] * (u0 * np.log(u0) + (1.0 - u0) * np.log(1.0 - u0)))

        idx1 = np.arange(t + 1, last + 1, dtype=np.float64)
        u1 = np.clip(1.0 / (1.0 + np.abs(idx1 - mu1) / c), 1e-9, 1.0 - 1e-9)
        s1 = -np.sum(hist[t + 1 : last + 1] * (u1 * np.log(u1) + (1.0 - u1) * np.log(1.0 - u1)))

        ent = s0 + s1
        if ent < best_ent:
            best_ent = ent
            best_t = t
    return int(np.clip(best_t, 0, 255))


def compute_threshold(
    plane: np.ndarray,
    method: str = "li",
    valid_mask: np.ndarray | None = None,
    manual_value: int = 100,
    offset: int = 0,
) -> int:
    m = method.lower()
    if m == "manual":
        return int(np.clip(manual_value, 0, 255))
    hist = _histogram(plane, valid_mask)
    fn = {
        "otsu": threshold_otsu,
        "isodata": threshold_isodata,
        "triangle": threshold_triangle,
        "li": threshold_li,
        "moments": threshold_moments,
        "yen": threshold_yen,
        "huang": threshold_huang,
    }.get(m)
    if fn is None:
        raise ValueError(f"Unknown threshold method: {method!r}")
    base = fn(hist)
    return int(np.clip(base + int(offset), 0, 255))


# ---------------------------------------------------------------------------
# Channel detection (SYTO 9 Green vs OYP Orange vs Dual)
# ---------------------------------------------------------------------------

def detect_channel_mode(rgb: np.ndarray, valid_mask: np.ndarray | None = None) -> str:
    """
    Detect whether an image contains:
      - 'dual_syto9_oyp': both Green (SYTO 9) and Orange (OYP) populations
      - 'green_syto9':    only Green (SYTO 9)
      - 'orange_oyp':     only Orange (OYP)
      - 'luminance':      grayscale / brightfield
    """
    pts = rgb[valid_mask] if valid_mask is not None else rgb.reshape(-1, 3)
    if pts.size == 0:
        return "luminance"

    r = pts[:, 0].astype(np.int16)
    g = pts[:, 1].astype(np.int16)
    b = pts[:, 2].astype(np.int16)

    if float(np.mean(np.abs(r - g))) < 5.0 and float(np.mean(np.abs(g - b))) < 5.0:
        return "luminance"

    n_px = float(r.size)
    green_dom = float(np.sum(((g - r) > 20) & (g > 35))) / n_px
    orange_dom = float(np.sum(((r - g) > 20) & (r > 40))) / n_px

    if green_dom > 0.012 and orange_dom > 0.012:
        return "dual_syto9_oyp"
    if green_dom > 0.012:
        return "green_syto9"
    if orange_dom > 0.012:
        return "orange_oyp"

    if float(np.percentile(r, 95)) > float(np.percentile(g, 95)) + 15:
        return "orange_oyp"
    if float(np.percentile(g, 95)) > float(np.percentile(r, 95)) + 15:
        return "green_syto9"
    return "luminance"


def _orange_oyp_plane(rgb: np.ndarray) -> np.ndarray:
    """Extract cleanest contrast plane for single-channel Orange (OYP) micrographs."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    if float(np.median(r)) > 120.0 and float(np.percentile(g, 95)) > 110.0:
        return g
    return np.maximum(r, g)


def subtract_background_rolling_ball(plane: np.ndarray, radius: int) -> np.ndarray:
    """
    Fast rolling-ball background subtraction via morphological grey opening,
    capped at the 65th percentile so dense saturated biofilm colonies are not
    hollowed out when radius is smaller than the colony diameter.
    """
    if radius <= 0:
        return plane
    h, w = plane.shape
    scale = 4 if radius >= 16 and min(h, w) >= 200 else (2 if radius >= 8 and min(h, w) >= 100 else 1)
    r_scaled = max(2, int(round(radius / scale)))
    yy, xx = np.ogrid[-r_scaled : r_scaled + 1, -r_scaled : r_scaled + 1]
    footprint = (xx * xx + yy * yy) <= (r_scaled * r_scaled)

    if scale > 1:
        small = Image.fromarray(plane).resize(
            (max(8, w // scale), max(8, h // scale)), Image.Resampling.BILINEAR
        )
        bg_small = ndimage.grey_opening(np.asarray(small, dtype=np.uint8), footprint=footprint)
        bg = np.asarray(
            Image.fromarray(bg_small).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.int16,
        )
    else:
        bg = ndimage.grey_opening(plane, footprint=footprint).astype(np.int16)

    cap = int(np.percentile(plane, 65))
    bg = np.minimum(bg, cap)
    return np.clip(plane.astype(np.int16) - bg, 0, 255).astype(np.uint8)


def build_valid_mask(h: int, w: int, params: AnalysisParams) -> tuple[np.ndarray, dict[str, int] | None]:
    valid = np.ones((h, w), dtype=bool)
    if params.roi is not None:
        x0p, y0p, x1p, y1p = params.roi
        x0 = int(np.clip(round(min(x0p, x1p) / 100.0 * w), 0, w))
        x1 = int(np.clip(round(max(x0p, x1p) / 100.0 * w), 0, w))
        y0 = int(np.clip(round(min(y0p, y1p) / 100.0 * h), 0, h))
        y1 = int(np.clip(round(max(y0p, y1p) / 100.0 * h), 0, h))
        roi_mask = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            roi_mask[y0:y1, x0:x1] = True
            valid &= roi_mask

    sb_box: dict[str, int] | None = None
    if params.exclude_scalebar:
        sw = int(np.clip(round(params.scalebar_w_pct / 100.0 * w), 0, w))
        sh = int(np.clip(round(params.scalebar_h_pct / 100.0 * h), 0, h))
        if sw > 0 and sh > 0:
            valid[h - sh : h, w - sw : w] = False
            sb_box = {"x0": w - sw, "y0": h - sh, "x1": w, "y1": h}
    return valid, sb_box


def _postprocess_mask(mask: np.ndarray, valid_mask: np.ndarray, min_area: int, fill_holes: bool) -> tuple[np.ndarray, int]:
    m = mask & valid_mask
    if fill_holes:
        m = ndimage.binary_fill_holes(m) & valid_mask
    labeled, n_comp = ndimage.label(m)
    if n_comp == 0:
        return m, 0
    if min_area > 1:
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_area
        keep[0] = False
        m = keep[labeled]
        _, n_comp = ndimage.label(m)
    return m, int(n_comp)


def _preprocess_plane(plane: np.ndarray, params: AnalysisParams) -> np.ndarray:
    out = plane.astype(np.uint8, copy=False)
    if params.invert:
        out = (255 - out).astype(np.uint8)
    if params.smooth_sigma and params.smooth_sigma > 0.05:
        out = np.clip(
            ndimage.gaussian_filter(out.astype(np.float32), sigma=float(params.smooth_sigma)),
            0, 255,
        ).astype(np.uint8)
    if params.bg_radius and params.bg_radius > 0:
        out = subtract_background_rolling_ball(out, int(params.bg_radius))
    return out


# ---------------------------------------------------------------------------
# Morphotype differentiation (Large Rods vs Small Coccobacilli)
# ---------------------------------------------------------------------------

def classify_morphotypes(
    mask: np.ndarray,
    *,
    rod_min_area_px: int = 230,
    rod_min_length_px: float = 28.0,
    cluster_mode: str = "separate",
    cluster_min_area_px: int = 1800,
) -> dict[str, Any]:
    """
    Split touching cells using smoothed distance-transform Voronoi seeding and
    classify each object into:
      - rods:     large elongated rod-shaped bacteria
      - cocci:    smaller coccobacilli
      - clusters: dense unresolved microcolony aggregates (handled per cluster_mode)
    """
    rods = np.zeros_like(mask, dtype=bool)
    cocci = np.zeros_like(mask, dtype=bool)
    clusters = np.zeros_like(mask, dtype=bool)

    if not np.any(mask):
        return {
            "rods": rods, "cocci": cocci, "clusters": clusters,
            "n_rods": 0, "n_cocci": 0, "n_clusters": 0,
        }

    dist = ndimage.distance_transform_edt(mask)
    dist_sm = ndimage.gaussian_filter(dist, sigma=1.4)
    peaks = (dist_sm == ndimage.maximum_filter(dist_sm, size=9)) & mask & (dist >= 1.5)
    markers, n_m = ndimage.label(peaks)

    comp, n_c = ndimage.label(mask)
    if n_c > 0:
        has_m = np.bincount(comp[markers > 0], minlength=n_c + 1) > 0
        missing = np.where(~has_m)[0]
        missing = missing[missing > 0]
        if len(missing):
            slices_c = ndimage.find_objects(comp)
            nid = n_m + 1
            for cid in missing:
                sl = slices_c[cid - 1]
                if sl is None:
                    continue
                sub = comp[sl] == cid
                iy, ix = np.unravel_index(
                    np.argmax(np.where(sub, dist_sm[sl], -1.0)), sub.shape
                )
                markers[sl][iy, ix] = nid
                nid += 1

    _, idx = ndimage.distance_transform_edt(markers == 0, return_indices=True)
    ws = np.where(mask, markers[idx[0], idx[1]], 0)

    n_rods = n_cocci = n_clusters = 0
    slices = ndimage.find_objects(ws)
    for cid, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub = ws[sl] == cid
        area = int(sub.sum())
        if area == 0:
            continue

        if area >= 4:
            ys, xs = np.nonzero(sub)
            yc = ys - ys.mean()
            xc = xs - xs.mean()
            cxx = float(np.mean(xc * xc)) + 1.0 / 12.0
            cyy = float(np.mean(yc * yc)) + 1.0 / 12.0
            cxy = float(np.mean(xc * yc))
            tr = cxx + cyy
            det = cxx * cyy - cxy * cxy
            disc = max(0.0, tr * tr / 4.0 - det)
            major = 4.0 * math.sqrt(tr / 2.0 + math.sqrt(disc))
            minor = 4.0 * math.sqrt(max(0.0, tr / 2.0 - math.sqrt(disc)))
        else:
            major = minor = 2.0
        aspect = major / max(minor, 1.0)

        is_cluster = (cluster_mode != "off") and (
            area >= cluster_min_area_px or (area >= max(650, rod_min_area_px * 2) and aspect < 1.9)
        )
        is_rod = (area >= rod_min_area_px) or (major >= rod_min_length_px and minor >= 7.5)

        if is_cluster:
            if cluster_mode == "separate":
                clusters[sl] |= sub
                n_clusters += 1
            elif cluster_mode == "rods":
                rods[sl] |= sub
                n_rods += 1
            else:  # "cocci"
                cocci[sl] |= sub
                n_cocci += 1
        elif is_rod:
            rods[sl] |= sub
            n_rods += 1
        else:
            cocci[sl] |= sub
            n_cocci += 1

    return {
        "rods": rods,
        "cocci": cocci,
        "clusters": clusters,
        "n_rods": n_rods,
        "n_cocci": n_cocci,
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# Core single-image analysis & overlay rendering
# ---------------------------------------------------------------------------

def _pil_to_rgb_u8(im: Image.Image) -> np.ndarray:
    """Convert any PIL image (including 16-bit/float TIFFs) cleanly to uint8 RGB."""
    if im.mode.startswith("I") or im.mode == "F":
        arr = np.asarray(im, dtype=np.float32)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi <= 255.0 and lo >= 0.0:
            u8 = arr.astype(np.uint8)
        elif hi > lo:
            u8 = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            u8 = np.zeros(arr.shape, dtype=np.uint8)
        return np.stack([u8, u8, u8], axis=-1)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def analyze_image(
    image_source: str | Path | Image.Image | np.ndarray,
    params: AnalysisParams | None = None,
    *,
    name: str | None = None,
    group: str = "Default",
    render_view: str | None = None,  # None | 'morphotype' | 'overlay' | 'mask' | 'original'
    max_preview_width: int = 900,
) -> dict[str, Any]:
    params = params or AnalysisParams()

    if isinstance(image_source, (str, Path)):
        p = Path(image_source)
        name = name or p.name
        with Image.open(p) as im:
            rgb = _pil_to_rgb_u8(im)
    elif isinstance(image_source, Image.Image):
        name = name or "image"
        rgb = _pil_to_rgb_u8(image_source)
    else:
        name = name or "image"
        arr = np.asarray(image_source)
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            hi = float(np.max(arr_f)) or 1.0
            arr = np.clip(arr_f / (65535.0 if hi > 255.0 else 1.0) * (255.0 if hi > 255.0 else 1.0), 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
        else:
            rgb = arr[:, :, :3].astype(np.uint8)

    h, w, _ = rgb.shape
    valid_mask, sb_box = build_valid_mask(h, w, params)
    analyzed_px = int(valid_mask.sum())
    if analyzed_px == 0:
        raise ValueError("ROI/exclusion left 0 pixels to analyze.")

    channel_used = detect_channel_mode(rgb, valid_mask) if params.channel == "auto" else params.channel

    green_mask: np.ndarray | None = None
    orange_mask: np.ndarray | None = None
    thresh_info: dict[str, int] = {}

    if channel_used == "dual_syto9_oyp":
        r_bg = _preprocess_plane(rgb[:, :, 0], params)
        g_bg = _preprocess_plane(rgb[:, :, 1], params)
        if params.unmix_dual:
            g_plane = np.clip(g_bg.astype(np.float32) - 0.45 * r_bg.astype(np.float32), 0, 255).astype(np.uint8)
            r_plane = np.clip(r_bg.astype(np.float32) - 0.35 * g_bg.astype(np.float32), 0, 255).astype(np.uint8)
        else:
            g_plane, r_plane = g_bg, r_bg

        t_g = compute_threshold(
            g_plane, params.method, valid_mask,
            manual_value=params.manual_threshold_green, offset=params.threshold_offset,
        )
        t_o = compute_threshold(
            r_plane, params.method, valid_mask,
            manual_value=params.manual_threshold_orange, offset=params.threshold_offset,
        )
        green_mask, _ = _postprocess_mask(g_plane > t_g, valid_mask, params.min_particle_px, params.fill_holes)
        orange_mask, _ = _postprocess_mask(r_plane > t_o, valid_mask, params.min_particle_px, params.fill_holes)
        final_mask, particle_count = _postprocess_mask(
            green_mask | orange_mask, valid_mask, params.min_particle_px, params.fill_holes
        )
        thresh_info = {"green": t_g, "orange": t_o}
        hist_plane = np.maximum(g_plane, r_plane)

    elif channel_used == "green_syto9":
        plane = _preprocess_plane(rgb[:, :, 1], params)
        t_val = compute_threshold(
            plane, params.method, valid_mask,
            manual_value=params.manual_threshold, offset=params.threshold_offset,
        )
        final_mask, particle_count = _postprocess_mask(
            plane > t_val, valid_mask, params.min_particle_px, params.fill_holes
        )
        green_mask = final_mask
        thresh_info = {"green": t_val}
        hist_plane = plane

    elif channel_used == "orange_oyp":
        raw_plane = _orange_oyp_plane(rgb)
        plane = _preprocess_plane(raw_plane, params)
        t_val = compute_threshold(
            plane, params.method, valid_mask,
            manual_value=params.manual_threshold, offset=params.threshold_offset,
        )
        final_mask, particle_count = _postprocess_mask(
            plane > t_val, valid_mask, params.min_particle_px, params.fill_holes
        )
        orange_mask = final_mask
        thresh_info = {"orange": t_val}
        hist_plane = plane

    else:  # luminance
        raw_plane = np.clip(
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2], 0, 255
        ).astype(np.uint8)
        plane = _preprocess_plane(raw_plane, params)
        t_val = compute_threshold(
            plane, params.method, valid_mask,
            manual_value=params.manual_threshold, offset=params.threshold_offset,
        )
        final_mask, particle_count = _postprocess_mask(
            plane > t_val, valid_mask, params.min_particle_px, params.fill_holes
        )
        thresh_info = {"value": t_val}
        hist_plane = plane

    fg_px = int(final_mask.sum())
    coverage_pct = 100.0 * fg_px / analyzed_px

    def _pct(m: np.ndarray | None) -> float | None:
        return round(100.0 * int(m.sum()) / analyzed_px, 4) if m is not None else None

    result: dict[str, Any] = {
        "name": name,
        "group": group,
        "width": w,
        "height": h,
        "channel_requested": params.channel,
        "channel_used": channel_used,
        "method": params.method,
        "thresholds": thresh_info,
        "analyzed_px": analyzed_px,
        "total_px": int(h * w),
        "foreground_px": fg_px,
        "coverage_pct": round(coverage_pct, 4),
        "green_syto9_pct": _pct(green_mask),
        "orange_oyp_pct": _pct(orange_mask),
        "overlap_pct": (
            _pct(green_mask & orange_mask)
            if (green_mask is not None and orange_mask is not None)
            else None
        ),
        "particle_count": particle_count,
        "rods_pct": None,
        "cocci_pct": None,
        "clusters_pct": None,
        "green_rods_pct": None,
        "green_cocci_pct": None,
        "green_clusters_pct": None,
        "orange_rods_pct": None,
        "orange_cocci_pct": None,
        "orange_clusters_pct": None,
        "n_rods": 0,
        "n_cocci": 0,
        "n_clusters": 0,
    }

    green_morph: dict[str, Any] | None = None
    orange_morph: dict[str, Any] | None = None
    single_morph: dict[str, Any] | None = None

    if params.separate_morphotypes:
        kw = dict(
            rod_min_area_px=int(params.rod_min_area_px),
            rod_min_length_px=float(params.rod_min_length_px),
            cluster_mode=str(params.cluster_mode),
            cluster_min_area_px=int(params.cluster_min_area_px),
        )
        if channel_used == "dual_syto9_oyp" and green_mask is not None and orange_mask is not None:
            green_morph = classify_morphotypes(green_mask, **kw)
            orange_morph = classify_morphotypes(orange_mask, **kw)
            result["green_rods_pct"] = _pct(green_morph["rods"])
            result["green_cocci_pct"] = _pct(green_morph["cocci"])
            result["green_clusters_pct"] = _pct(green_morph["clusters"])
            result["orange_rods_pct"] = _pct(orange_morph["rods"])
            result["orange_cocci_pct"] = _pct(orange_morph["cocci"])
            result["orange_clusters_pct"] = _pct(orange_morph["clusters"])
            all_rods = green_morph["rods"] | orange_morph["rods"]
            all_cocci = green_morph["cocci"] | orange_morph["cocci"]
            all_clusters = green_morph["clusters"] | orange_morph["clusters"]
            result["rods_pct"] = _pct(all_rods)
            result["cocci_pct"] = _pct(all_cocci)
            result["clusters_pct"] = _pct(all_clusters)
            result["n_rods"] = green_morph["n_rods"] + orange_morph["n_rods"]
            result["n_cocci"] = green_morph["n_cocci"] + orange_morph["n_cocci"]
            result["n_clusters"] = green_morph["n_clusters"] + orange_morph["n_clusters"]
        else:
            single_morph = classify_morphotypes(final_mask, **kw)
            result["rods_pct"] = _pct(single_morph["rods"])
            result["cocci_pct"] = _pct(single_morph["cocci"])
            result["clusters_pct"] = _pct(single_morph["clusters"])
            result["n_rods"] = single_morph["n_rods"]
            result["n_cocci"] = single_morph["n_cocci"]
            result["n_clusters"] = single_morph["n_clusters"]
            if channel_used == "green_syto9":
                result["green_rods_pct"] = result["rods_pct"]
                result["green_cocci_pct"] = result["cocci_pct"]
                result["green_clusters_pct"] = result["clusters_pct"]
            elif channel_used == "orange_oyp":
                result["orange_rods_pct"] = result["rods_pct"]
                result["orange_cocci_pct"] = result["cocci_pct"]
                result["orange_clusters_pct"] = result["clusters_pct"]

    h256 = _histogram(hist_plane, valid_mask)
    result["histogram_32"] = [int(h256[i * 8 : (i + 1) * 8].sum()) for i in range(32)]

    if render_view:
        result["preview_b64"] = render_preview_b64(
            rgb, final_mask, valid_mask, sb_box,
            green_mask=green_mask, orange_mask=orange_mask,
            single_morph=single_morph, green_morph=green_morph, orange_morph=orange_morph,
            view=render_view, max_width=max_preview_width,
        )

    return result


def render_preview_b64(
    rgb: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    sb_box: dict[str, int] | None,
    *,
    green_mask: np.ndarray | None = None,
    orange_mask: np.ndarray | None = None,
    single_morph: dict[str, Any] | None = None,
    green_morph: dict[str, Any] | None = None,
    orange_morph: dict[str, Any] | None = None,
    view: str = "overlay",
    max_width: int = 900,
) -> str:
    h, w, _ = rgb.shape
    if view == "original":
        out = rgb.copy()
    elif view == "morphotype" and (single_morph is not None or (green_morph is not None and orange_morph is not None)):
        out = (rgb.astype(np.float32) * 0.36).astype(np.uint8)
        if green_morph is not None and orange_morph is not None:
            layers = (
                (green_morph["cocci"], COLOR_GREEN_COCCI),
                (green_morph["rods"], COLOR_GREEN_ROD),
                (orange_morph["cocci"], COLOR_ORANGE_COCCI),
                (orange_morph["rods"], COLOR_ORANGE_ROD),
                (green_morph["clusters"] | orange_morph["clusters"], COLOR_CLUSTER),
            )
        else:
            assert single_morph is not None
            layers = (
                (single_morph["cocci"], COLOR_COCCI),
                (single_morph["rods"], COLOR_ROD),
                (single_morph["clusters"], COLOR_CLUSTER),
            )
        for m_sub, col in layers:
            if np.any(m_sub):
                out[m_sub] = (
                    0.25 * rgb[m_sub].astype(np.float32) + 0.75 * np.array(col, dtype=np.float32)
                ).astype(np.uint8)
        out[~valid_mask] = (rgb[~valid_mask].astype(np.float32) * 0.25).astype(np.uint8)
    elif view == "mask":
        out = np.zeros_like(rgb)
        if green_morph is not None and orange_morph is not None:
            out[green_morph["cocci"]] = COLOR_GREEN_COCCI
            out[green_morph["rods"]] = COLOR_GREEN_ROD
            out[orange_morph["cocci"]] = COLOR_ORANGE_COCCI
            out[orange_morph["rods"]] = COLOR_ORANGE_ROD
            out[green_morph["clusters"] | orange_morph["clusters"]] = COLOR_CLUSTER
        elif single_morph is not None:
            out[single_morph["cocci"]] = COLOR_COCCI
            out[single_morph["rods"]] = COLOR_ROD
            out[single_morph["clusters"]] = COLOR_CLUSTER
        elif green_mask is not None and orange_mask is not None:
            out[green_mask & ~orange_mask] = COLOR_GREEN
            out[orange_mask & ~green_mask] = COLOR_ORANGE
            out[green_mask & orange_mask] = COLOR_BOTH
        elif green_mask is not None:
            out[green_mask] = COLOR_GREEN
        elif orange_mask is not None:
            out[orange_mask] = COLOR_ORANGE
        else:
            out[mask] = (255, 255, 255)
        out[~valid_mask] = (45, 48, 54)
    else:  # overlay (channel colors)
        out = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
        if green_mask is not None and orange_mask is not None:
            layers = (
                (green_mask & ~orange_mask, COLOR_GREEN),
                (orange_mask & ~green_mask, COLOR_ORANGE),
                (green_mask & orange_mask, COLOR_BOTH),
            )
            for m_sub, col in layers:
                if np.any(m_sub):
                    out[m_sub] = (
                        0.30 * rgb[m_sub].astype(np.float32) + 0.70 * np.array(col, dtype=np.float32)
                    ).astype(np.uint8)
        elif green_mask is not None:
            if np.any(green_mask):
                out[green_mask] = (
                    0.30 * rgb[green_mask].astype(np.float32) + 0.70 * np.array(COLOR_GREEN, dtype=np.float32)
                ).astype(np.uint8)
        elif orange_mask is not None:
            if np.any(orange_mask):
                out[orange_mask] = (
                    0.30 * rgb[orange_mask].astype(np.float32) + 0.70 * np.array(COLOR_ORANGE, dtype=np.float32)
                ).astype(np.uint8)
        elif np.any(mask):
            out[mask] = (
                0.30 * rgb[mask].astype(np.float32) + 0.70 * np.array(COLOR_GRAY, dtype=np.float32)
            ).astype(np.uint8)
        out[~valid_mask] = (rgb[~valid_mask].astype(np.float32) * 0.25).astype(np.uint8)

    pil_img = Image.fromarray(out)
    if sb_box is not None and view != "original":
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle(
            [sb_box["x0"], sb_box["y0"], sb_box["x1"] - 1, sb_box["y1"] - 1],
            outline=(255, 190, 60), width=2,
        )

    if w > max_width:
        nh = max(1, int(round(h * max_width / w)))
        pil_img = pil_img.resize((max_width, nh), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Batch & Group Summary Statistics (% ± SD)
# ---------------------------------------------------------------------------

def summarize_group(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean_pct": 0.0, "sd_pct": 0.0, "sd_pop_pct": 0.0,
            "sem_pct": 0.0, "median_pct": 0.0, "iqr_pct": 0.0,
            "min_pct": 0.0, "max_pct": 0.0, "formatted": "—",
        }
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd_sample = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    sd_pop = float(np.std(arr, ddof=0))
    sem = sd_sample / math.sqrt(n) if n >= 2 else 0.0
    q25, med, q75 = [float(x) for x in np.percentile(arr, [25, 50, 75])]
    return {
        "n": n,
        "mean_pct": round(mean, 4),
        "sd_pct": round(sd_sample, 4),
        "sd_pop_pct": round(sd_pop, 4),
        "sem_pct": round(sem, 4),
        "median_pct": round(med, 4),
        "iqr_pct": round(q75 - q25, 4),
        "min_pct": round(float(np.min(arr)), 4),
        "max_pct": round(float(np.max(arr)), 4),
        "formatted": f"{mean:.2f} ± {sd_sample:.2f}%" if n >= 2 else f"{mean:.2f}% (n=1)",
    }


def _summarize_rows(rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    s = summarize_group([r["coverage_pct"] for r in rows])
    s["group"] = group_name

    def _opt_sum(key: str) -> dict[str, Any] | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return summarize_group(vals) if vals else None

    for k in (
        "green_syto9", "orange_oyp", "rods", "cocci", "clusters",
        "green_rods", "green_cocci", "green_clusters",
        "orange_rods", "orange_cocci", "orange_clusters",
    ):
        row_k = f"{k}_pct"
        sub = _opt_sum(row_k)
        if sub is not None:
            s[f"{k}_summary"] = sub

    s["has_both_channels"] = bool(s.get("green_syto9_summary") and s.get("orange_oyp_summary"))
    s["has_morphotypes"] = bool(s.get("rods_summary") and s.get("cocci_summary"))
    return s


def analyze_batch(
    items: list[tuple[str | Path | Image.Image, str, str]],
    params: AnalysisParams | None = None,
    *,
    render_view: str | None = "morphotype",
    max_preview_width: int = 900,
) -> dict[str, Any]:
    params = params or AnalysisParams()
    results: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = {}

    for src, name, grp in items:
        grp_clean = (grp or "Default").strip() or "Default"
        res = analyze_image(
            src, params, name=name, group=grp_clean,
            render_view=render_view, max_preview_width=max_preview_width,
        )
        results.append(res)
        by_group.setdefault(grp_clean, []).append(res)

    groups_summary = [_summarize_rows(rows, grp_name) for grp_name, rows in by_group.items()]
    overall = _summarize_rows(results, "ALL_COMBINED")
    return {
        "params": asdict(params),
        "images": results,
        "groups": groups_summary,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# CSV & Publication Figure Export
# ---------------------------------------------------------------------------

def results_to_csv(batch: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "Image", "Group", "Channel_Mode", "Threshold_Method", "Thresholds",
        "Total_Coverage_Pct", "Green_SYTO9_Pct", "Orange_OYP_Pct", "Overlap_Pct",
        "Rods_Pct", "Coccobacilli_Pct", "Clusters_Pct",
        "Green_Rods_Pct", "Green_Coccobacilli_Pct", "Green_Clusters_Pct",
        "Orange_Rods_Pct", "Orange_Coccobacilli_Pct", "Orange_Clusters_Pct",
        "N_Rods", "N_Coccobacilli", "N_Clusters",
        "Foreground_Px", "Analyzed_Px", "Particle_Count", "Width", "Height",
    ])

    def _f(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else ""

    for r in batch.get("images", []):
        t_str = ";".join(f"{k}={v}" for k, v in r.get("thresholds", {}).items())
        w.writerow([
            r["name"], r["group"], r["channel_used"], r["method"], t_str,
            f"{r['coverage_pct']:.4f}",
            _f(r.get("green_syto9_pct")), _f(r.get("orange_oyp_pct")), _f(r.get("overlap_pct")),
            _f(r.get("rods_pct")), _f(r.get("cocci_pct")), _f(r.get("clusters_pct")),
            _f(r.get("green_rods_pct")), _f(r.get("green_cocci_pct")), _f(r.get("green_clusters_pct")),
            _f(r.get("orange_rods_pct")), _f(r.get("orange_cocci_pct")), _f(r.get("orange_clusters_pct")),
            r.get("n_rods", 0), r.get("n_cocci", 0), r.get("n_clusters", 0),
            r["foreground_px"], r["analyzed_px"], r["particle_count"],
            r["width"], r["height"],
        ])

    w.writerow([])
    w.writerow([
        "Group_Summary", "Metric", "N", "Mean_Pct", "SD_Sample_Pct",
        "SD_Pop_Pct", "SEM_Pct", "Median_Pct", "IQR_Pct", "Min_Pct", "Max_Pct", "Mean_plus_SD",
    ])

    def _write_sum(label: str, metric: str, g: dict[str, Any]) -> None:
        w.writerow([
            label, metric, g["n"], f"{g['mean_pct']:.4f}", f"{g['sd_pct']:.4f}",
            f"{g['sd_pop_pct']:.4f}", f"{g['sem_pct']:.4f}",
            f"{g['median_pct']:.4f}", f"{g['iqr_pct']:.4f}",
            f"{g['min_pct']:.4f}", f"{g['max_pct']:.4f}", g["formatted"],
        ])

    summary_keys = [
        ("Total_Biofilm", None),
        ("Green_SYTO9", "green_syto9_summary"),
        ("Orange_OYP", "orange_oyp_summary"),
        ("Large_Rods", "rods_summary"),
        ("Small_Coccobacilli", "cocci_summary"),
        ("Dense_Clusters", "clusters_summary"),
        ("Green_Rods", "green_rods_summary"),
        ("Green_Coccobacilli", "green_cocci_summary"),
        ("Orange_Rods", "orange_rods_summary"),
        ("Orange_Coccobacilli", "orange_cocci_summary"),
    ]
    for g in batch.get("groups", []):
        for metric, skey in summary_keys:
            sub = g if skey is None else g.get(skey)
            if sub is not None and (skey != "clusters_summary" or sub["mean_pct"] > 0):
                _write_sum(g["group"], metric, sub)

    return buf.getvalue()


def export_summary_figure(
    batch: dict[str, Any],
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    title: str = "Biofilm Surface Coverage (SYTO 9 / OYP & Morphotypes)",
) -> None:
    """Export publication-ready Mean ± SD bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = batch.get("groups", [])
    rows = batch.get("images", [])
    if not groups:
        raise ValueError("No groups to plot.")

    has_morph = any(g.get("has_morphotypes") for g in groups)
    has_dual = any(g.get("has_both_channels") for g in groups)

    if has_dual and has_morph:
        series = [
            ("Green Rods", "green_rods_pct", "green_rods_summary", "#059669"),
            ("Green Cocci", "green_cocci_pct", "green_cocci_summary", "#6EE7B7"),
            ("Orange Rods", "orange_rods_pct", "orange_rods_summary", "#EA580C"),
            ("Orange Cocci", "orange_cocci_pct", "orange_cocci_summary", "#FCD34D"),
            ("Total", "coverage_pct", None, "#2563EB"),
        ]
    elif has_morph:
        series = [
            ("Large Rods", "rods_pct", "rods_summary", "#06B6D4"),
            ("Coccobacilli", "cocci_pct", "cocci_summary", "#F59E0B"),
            ("Total", "coverage_pct", None, "#2563EB"),
        ]
        if any((g.get("clusters_summary") or {}).get("mean_pct", 0) > 0.05 for g in groups):
            series.insert(2, ("Clusters", "clusters_pct", "clusters_summary", "#D946EF"))
    elif has_dual:
        series = [
            ("Green (SYTO 9)", "green_syto9_pct", "green_syto9_summary", "#10B981"),
            ("Orange (OYP)", "orange_oyp_pct", "orange_oyp_summary", "#F59E0B"),
            ("Total", "coverage_pct", None, "#2563EB"),
        ]
    else:
        series = [("Coverage", "coverage_pct", None, "#2563EB")]

    n_ser = len(series)
    fig_w = max(5.6, (0.75 * n_ser + 0.8) * len(groups) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.9), dpi=300)
    rng = np.random.default_rng(42)

    xs = np.arange(len(groups), dtype=float)
    width = min(0.55, 0.78 / max(1, n_ser))
    ymax_seen = 10.0

    for s_idx, (label, row_key, sum_key, color) in enumerate(series):
        offset = (s_idx - (n_ser - 1) / 2.0) * width
        means, sds, bar_xs = [], [], []
        for i, g in enumerate(groups):
            sub = g.get(sum_key) if sum_key else g
            if sub is None:
                continue
            bar_xs.append(i + offset)
            means.append(sub["mean_pct"])
            sds.append(sub["sd_pct"])
            pts = [r[row_key] for r in rows if r["group"] == g["group"] and r.get(row_key) is not None]
            jitter = rng.uniform(-width * 0.16, width * 0.16, size=len(pts)) if len(pts) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(pts), i + offset) + jitter, pts,
                s=24, color="#FFFFFF", edgecolor="#111827", linewidth=0.9, zorder=4,
            )
            ymax_seen = max(ymax_seen, sub["mean_pct"] + sub["sd_pct"], *(pts or [0.0]))
            ax.text(
                i + offset, sub["mean_pct"] + sub["sd_pct"] + 1.6,
                f"{sub['mean_pct']:.1f}±{sub['sd_pct']:.1f}%",
                ha="center", va="bottom", fontsize=7.2 if n_ser >= 3 else 9.0,
                fontweight="bold", color="#111827",
            )
        if bar_xs:
            ax.bar(
                bar_xs, means, yerr=sds, width=width * 0.9, label=label,
                color=color, alpha=0.86, edgecolor="#111827", linewidth=1.0,
                capsize=3.5 if n_ser >= 3 else 5,
                error_kw={"elinewidth": 1.2, "ecolor": "#111827", "capthick": 1.1},
                zorder=2,
            )

    if n_ser > 1:
        ax.legend(frameon=False, loc="upper left", ncol=min(n_ser, 3), fontsize=8.5)

    ax.set_ylim(0, min(118.0, max(ymax_seen * 1.28, ymax_seen + 14.0)))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g['group']} (n={g['n']})" for g in groups], fontsize=10.5)
    ax.set_ylabel("Biofilm Coverage Area (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="ImageJ-like Biofilm Coverage Area (% + SD) for SYTO 9, OYP, and Rod/Coccobacilli morphotypes"
    )
    ap.add_argument("images", nargs="+", help="Image paths or directories")
    ap.add_argument("--group", default="Sample", help="Group name for all images (or 'auto' by prefix)")
    ap.add_argument("--channel", choices=CHANNEL_MODES, default="auto")
    ap.add_argument("--method", choices=THRESHOLD_METHODS, default="li")
    ap.add_argument("--no-unmix", action="store_true", help="Disable RGB spectral unmixing for dual SYTO9+OYP")
    ap.add_argument("--no-morphotypes", action="store_true", help="Disable Rod vs Coccobacilli differentiation")
    ap.add_argument("--rod-area", type=int, default=230, help="Min pixel area for large rods (default 230)")
    ap.add_argument("--rod-length", type=float, default=28.0, help="Min major axis length px for rods (default 28)")
    ap.add_argument("--cluster-mode", choices=CLUSTER_MODES, default="separate")
    ap.add_argument("--manual", type=int, default=100, help="Manual threshold (0-255)")
    ap.add_argument("--offset", type=int, default=0, help="Threshold offset (-100..100)")
    ap.add_argument("--bg-radius", type=int, default=35, help="Rolling-ball background radius px (0=off)")
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian sigma (0=off)")
    ap.add_argument("--min-area", type=int, default=15, help="Min particle area in px")
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--invert", action="store_true", help="Invert (for brightfield dark-on-light)")
    ap.add_argument("--keep-scalebar", action="store_true", help="Do not exclude bottom-right scalebar")
    ap.add_argument("--csv", help="Write CSV report to path")
    ap.add_argument("--plot", help="Write Mean ± SD PNG chart to path")
    ap.add_argument("--pdf", help="Also write vector PDF chart to path")
    args = ap.parse_args(argv)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths: list[Path] = []
    for item in args.images:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in exts))
        elif p.is_file():
            paths.append(p)
    if not paths:
        print("No valid image files found.")
        return 1

    def infer_group(p: Path) -> str:
        if args.group != "auto":
            return args.group
        stem = p.stem.split(".")[0]
        prefix = "".join(ch for ch in stem if not ch.isdigit()).strip("_- ")
        return prefix or "Sample"

    params = AnalysisParams(
        channel=args.channel,
        method=args.method,
        unmix_dual=not args.no_unmix,
        separate_morphotypes=not args.no_morphotypes,
        rod_min_area_px=args.rod_area,
        rod_min_length_px=args.rod_length,
        cluster_mode=args.cluster_mode,
        manual_threshold=args.manual,
        threshold_offset=args.offset,
        bg_radius=args.bg_radius,
        smooth_sigma=args.smooth,
        min_particle_px=args.min_area,
        fill_holes=args.fill_holes,
        invert=args.invert,
        exclude_scalebar=not args.keep_scalebar,
    )

    items = [(p, p.name, infer_group(p)) for p in paths]
    batch = analyze_batch(items, params, render_view=None)

    print(f"\n{'Image':18s} {'Mode':14s} {'Green%':>8s} {'Orange%':>8s} {'Rods%':>8s} {'Cocci%':>8s} {'Clust%':>8s} {'Total%':>8s}")
    print("-" * 88)
    for r in batch["images"]:
        gs = f"{r['green_syto9_pct']:7.2f}%" if r["green_syto9_pct"] is not None else "       —"
        os_ = f"{r['orange_oyp_pct']:7.2f}%" if r["orange_oyp_pct"] is not None else "       —"
        rs = f"{r['rods_pct']:7.2f}%" if r["rods_pct"] is not None else "       —"
        cs = f"{r['cocci_pct']:7.2f}%" if r["cocci_pct"] is not None else "       —"
        cls = f"{r['clusters_pct']:7.2f}%" if r["clusters_pct"] is not None else "       —"
        print(f"{r['name']:18s} {r['channel_used']:14s} {gs} {os_} {rs} {cs} {cls} {r['coverage_pct']:7.2f}%")

    print("\nGroup Summary (Mean ± SD):")
    for g in batch["groups"]:
        print(f"  {g['group']} (n={g['n']}):")
        for lbl, skey in (
            ("Total Coverage", None),
            ("Green (SYTO 9)", "green_syto9_summary"),
            ("Orange (OYP)", "orange_oyp_summary"),
            ("Large Rods", "rods_summary"),
            ("Coccobacilli", "cocci_summary"),
            ("Dense Clusters", "clusters_summary"),
            ("  Green Rods", "green_rods_summary"),
            ("  Green Cocci", "green_cocci_summary"),
            ("  Orange Rods", "orange_rods_summary"),
            ("  Orange Cocci", "orange_cocci_summary"),
        ):
            sub = g if skey is None else g.get(skey)
            if sub is not None and (skey != "clusters_summary" or sub["mean_pct"] > 0):
                print(f"    {lbl:16s}: {sub['formatted']}")

    if args.csv:
        Path(args.csv).write_text(results_to_csv(batch), encoding="utf-8")
        print(f"\nSaved CSV -> {args.csv}")
    if args.plot:
        export_summary_figure(batch, args.plot, args.pdf)
        print(f"Saved plot -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
