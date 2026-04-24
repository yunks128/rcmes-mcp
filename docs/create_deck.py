"""Generate RCMES-MCP slide deck using the NASA JPL external-facing template.

Also regenerates illustrative PNG figures (under docs/figures/) for the advanced
spatiotemporal-analysis slides. Figures use synthetic data — they document what
each tool produces, not real model output.
"""

import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import copy
from lxml import etree

# Headless matplotlib for figure generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

TEMPLATE = "/home/ks/science-model-dashboard/NASAjpl_Template_16x9_vA9.pptx"
OUTPUT = "/home/ks/rcmes-mcp/docs/RCMES-MCP_Introduction.pptx"
FIG_DIR = Path("/home/ks/rcmes-mcp/docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

prs = Presentation(TEMPLATE)

# Remove existing slides (iterate in reverse)
while len(prs.slides) > 0:
    rId = prs.slides._sldIdLst[-1].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
    if rId is None:
        # try without namespace
        rId_elem = prs.slides._sldIdLst[-1]
        for rel in prs.part.rels.values():
            if rel.target_part == prs.slides[-1].part:
                rId = rel.rId
                break
    prs.part.drop_rel(rId)
    prs.slides._sldIdLst.remove(prs.slides._sldIdLst[-1])

# Layout references
COVER_NO_IMAGE = prs.slide_layouts[0]    # Cover_No Image
COVER_SQ_IMAGE = prs.slide_layouts[1]    # Cover_Sq Image
TITLE_CONTENT = prs.slide_layouts[5]     # Title and Content
TITLE_SUBTITLE = prs.slide_layouts[6]    # Title and Subtitle
TITLE_SUB_CONTENT = prs.slide_layouts[7] # Title, Subtitle and Content
TWO_CONTENT = prs.slide_layouts[8]       # Two Content
TITLE_ONLY = prs.slide_layouts[4]        # Title Only
BLANK = prs.slide_layouts[17]            # Blank
CLOSING = prs.slide_layouts[18]          # Closing Slide

NAVY = RGBColor(0x00, 0x32, 0x5B)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
FONT = "Arial"

FOOTER_TEXT = "For required markings, please visit https://mh.jpl.nasa.gov"


def set_placeholder_text(slide, idx, text):
    """Set text in a placeholder by index."""
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            ph.text = text
            return ph
    return None


def add_textbox(slide, left, top, width, height, text, font_size=Pt(14),
                bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT, font_name=FONT):
    """Add a simple textbox."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = font_size
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return box


def set_footer(slide):
    """Set footer placeholder if it exists."""
    for ph in slide.placeholders:
        pf = ph.placeholder_format
        if pf.type is not None and pf.type.name == 'FOOTER':
            ph.text = FOOTER_TEXT


def add_bullet_content(tf, items, font_size=Pt(14), color=DARK_GRAY, spacing=Pt(6)):
    """Add bullet items to an existing text frame."""
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = font_size
        p.font.color.rgb = color
        p.font.name = FONT
        p.space_after = spacing
        p.level = 0


# ============================================================
# FIGURE GENERATORS — synthetic but representative outputs of each new tool
# Saved to docs/figures/, embedded into the new slides below.
# ============================================================

_RNG = np.random.default_rng(42)


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------
# Real-data path: load NEX-GDDP-CMIP6 via RCMES tools, run actual analyses.
# Falls back to synthetic figures on any failure so deck regen never blocks.
# Toggle with RCMES_DECK_REAL_DATA=0 to force synthetic.
# ----------------------------------------------------------------------
os.environ.setdefault("RCMES_CACHE_DIR", "/home/ks/rcmes-mcp/rcmes_mcp_cache")
USE_REAL_DATA = os.environ.get("RCMES_DECK_REAL_DATA", "1") != "0"
_REAL_AVAILABLE = False
if USE_REAL_DATA:
    try:
        import xarray as xr
        from scipy import ndimage
        from rcmes_mcp.tools.data_access import load_climate_data, wait_for_materialization
        from rcmes_mcp.utils.session import session_manager
        _REAL_AVAILABLE = True
        print("[deck] Real-data mode ON — figures will use NEX-GDDP-CMIP6 from S3")
    except Exception as e:
        print(f"[deck] Real-data deps unavailable ({e}); using synthetic figures")
else:
    print("[deck] Synthetic figures requested (RCMES_DECK_REAL_DATA=0)")

CALIFORNIA = dict(lat_min=32.0, lat_max=42.0, lon_min=-125.0, lon_max=-114.0)
LA_BASIN = dict(lat_min=32.0, lat_max=36.0, lon_min=-120.0, lon_max=-116.0)
WESTERN_US = dict(lat_min=30.0, lat_max=50.0, lon_min=-125.0, lon_max=-110.0)


def _load(variable, model, scenario, start_date, end_date, **bbox):
    """Load NEX-GDDP-CMIP6 via the RCMES load_climate_data tool. Returns DataArray."""
    print(f"[deck] _load {variable} {model}/{scenario} {start_date}..{end_date} bbox={bbox}")
    r = load_climate_data(
        variable=variable, model=model, scenario=scenario,
        start_date=start_date, end_date=end_date, **bbox,
    )
    if not r.get("success"):
        raise RuntimeError(f"load_climate_data failed: {r.get('error')}")
    ds_id = r["dataset_id"]
    wait_for_materialization(ds_id, timeout=1800)
    ds = session_manager.get(ds_id)
    if isinstance(ds, xr.DataArray):
        return ds
    var = variable if variable in ds.data_vars else list(ds.data_vars)[0]
    return ds[var]


def _normalize_lons_arr(arr_2d, lons):
    """Convert 0-360 lons to -180:180 and re-sort columns of arr_2d to match."""
    lons = np.asarray(lons)
    if lons.max() > 180:
        lons = ((lons + 180) % 360) - 180
        sort = np.argsort(lons)
        return arr_2d[..., sort], lons[sort]
    return arr_2d, lons


def _to_year_index(da):
    """Replace time coord with integer year so xr.concat works across CMIP6 calendars."""
    years = np.array([
        t.year if hasattr(t, "year") else int(str(t)[:4])
        for t in da.time.values
    ])
    return da.assign_coords(time=years)


# ============================================================
# REAL-DATA FIGURE GENERATORS
# ============================================================

def _real_anomaly_hovmoller():
    # Near-future window so individual extreme blobs stand out against
    # a near-zero background rather than uniformly-red late-century forcing.
    fut = _load("tasmax", "ACCESS-CM2", "ssp585",
                "2025-01-01", "2029-12-31", **CALIFORNIA)
    base = _load("tasmax", "ACCESS-CM2", "historical",
                 "1980-01-01", "2009-12-31", **CALIFORNIA)
    print("[deck] Computing day-of-year z-score climatology...")
    clim_mean = base.groupby("time.dayofyear").mean()
    clim_std = base.groupby("time.dayofyear").std().where(lambda s: s > 0)
    z = (fut.groupby("time.dayofyear") - clim_mean).groupby("time.dayofyear") / clim_std
    z = z.resample(time="7D").mean()  # weekly for cleaner Hovmöller
    hov = z.mean(dim="lat").compute()

    arr, lons = _normalize_lons_arr(hov.values, hov.lon.values)
    times = hov.time.values

    # Find blob events on the (time × lon) hovmöller
    mask = np.abs(arr) > 1.5
    labels, n = ndimage.label(mask)
    boxes = []
    for i in range(1, n + 1):
        idx = np.where(labels == i)
        if idx[0].size < 8:
            continue
        boxes.append((int(idx[1].min()), int(idx[1].max()),
                      int(idx[0].min()), int(idx[0].max())))

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    vlim = 3.5
    im = ax.pcolormesh(lons, times, arr, cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim, shading="auto")
    for x0, x1, t0, t1 in boxes:
        ax.plot(
            [lons[x0], lons[x1], lons[x1], lons[x0], lons[x0]],
            [times[t0], times[t0], times[t1], times[t1], times[t0]],
            color="black", linewidth=1.2, linestyle="--",
        )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Time")
    ax.set_title("California tasmax z-score — zonal mean Hovmöller\n"
                 "ACCESS-CM2 SSP5-8.5 (2025–29) vs 1980–2009 baseline")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("z-score (σ)")
    return _save(fig, "fig_anomaly_hovmoller.png")


def _real_eof():
    base = _load("tasmax", "ACCESS-CM2", "historical",
                 "1980-01-01", "2009-12-31", **CALIFORNIA)
    print("[deck] Monthly anomaly + EOF SVD...")
    monthly = base.resample(time="MS").mean()
    clim = monthly.groupby("time.month").mean()
    anom = (monthly.groupby("time.month") - clim).compute().transpose("time", "lat", "lon")

    arr = anom.values.astype(np.float64)
    n_t, n_lat, n_lon = arr.shape
    lat_w = np.sqrt(np.cos(np.deg2rad(anom.lat.values))).clip(0)
    weighted = arr * lat_w[np.newaxis, :, np.newaxis]
    flat = weighted.reshape(n_t, -1)
    col_finite = np.isfinite(flat).all(axis=0)
    flat_clean = np.nan_to_num(flat[:, col_finite])

    U, S, Vt = np.linalg.svd(flat_clean, full_matrices=False)
    var_frac = (S ** 2) / np.sum(S ** 2)

    eof1_flat = np.full(n_lat * n_lon, np.nan)
    eof1_flat[col_finite] = Vt[0, :]
    eof1 = (eof1_flat.reshape(n_lat, n_lon) / lat_w[:, np.newaxis])
    pc1 = U[:, 0] * S[0]

    eof1, lons = _normalize_lons_arr(eof1, anom.lon.values)
    lats = anom.lat.values

    fig = plt.figure(figsize=(11, 4.5))
    ax_map = fig.add_subplot(1, 2, 1)
    vlim = float(np.nanmax(np.abs(eof1)))
    im = ax_map.pcolormesh(lons, lats, eof1, cmap="RdBu_r",
                           vmin=-vlim, vmax=vlim, shading="auto")
    ax_map.set_xlabel("Longitude (°)")
    ax_map.set_ylabel("Latitude (°)")
    ax_map.set_title(f"EOF1 — California tasmax monthly anomaly ({var_frac[0]*100:.0f}% var)\nACCESS-CM2 historical 1980–2009")
    plt.colorbar(im, ax=ax_map, pad=0.02, label="loading")

    ax_ts = fig.add_subplot(1, 2, 2)
    months = anom.time.values
    ax_ts.plot(months, pc1, color="#0f4c81", linewidth=1.0)
    ax_ts.axhline(0, color="grey", linewidth=0.6)
    ax_ts.fill_between(months, pc1, 0, where=pc1 > 0, color="#c4321c", alpha=0.3)
    ax_ts.fill_between(months, pc1, 0, where=pc1 < 0, color="#1a73c2", alpha=0.3)
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("PC1 amplitude")
    ax_ts.set_title("Principal component 1")
    ax_ts.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_eof.png")


def _real_ensemble_spread():
    models = ["ACCESS-CM2", "GFDL-ESM4", "MRI-ESM2-0"]
    series = []
    used = []
    for m in models:
        try:
            data = _load("tas", m, "ssp585",
                         "2050-01-01", "2099-12-31", **LA_BASIN)
            ts = data.mean(dim=["lat", "lon"]).resample(time="1YS").mean().compute()
            ts = _to_year_index(ts)  # normalize across model calendars
            series.append(ts)
            used.append(m)
        except Exception as e:
            print(f"[deck]   {m} skipped: {e}")
    if not series:
        raise RuntimeError("no ensemble members loaded")
    stack = xr.concat(series, dim=xr.Variable("model", used), join="outer")
    if float(stack.mean()) > 200:
        stack = stack - 273.15
    ref = float(stack.median(dim="model").isel(time=0))
    stack_anom = stack - ref
    p05 = stack_anom.quantile(0.05, dim="model")
    p25 = stack_anom.quantile(0.25, dim="model")
    p50 = stack_anom.median(dim="model")
    p75 = stack_anom.quantile(0.75, dim="model")
    p95 = stack_anom.quantile(0.95, dim="model")
    times = stack_anom.time.values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(times, p05, p95, color="#1f77b4", alpha=0.18, label="5–95% across models")
    ax.fill_between(times, p25, p75, color="#1f77b4", alpha=0.3, label="25–75%")
    for ts in series:
        vals = ts.values
        if float(vals.mean()) > 200:
            vals = vals - 273.15
        ax.plot(ts.time.values, vals - ref, color="grey", linewidth=0.5, alpha=0.5)
    ax.plot(times, p50, color="#1f77b4", linewidth=2, label="Median")
    ax.set_xlabel("Year")
    ax.set_ylabel("Δ Temperature vs 2050 (°C)")
    ax.set_title(f"LA-area tas SSP5-8.5 — {len(used)}-model ensemble spread")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_ensemble_spread.png")


def _real_scenario_fan():
    scenarios = [("ssp126", "#1a9850"),
                 ("ssp245", "#fdae61"),
                 ("ssp585", "#7f0000")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ref = None
    plotted = 0
    for sc, color in scenarios:
        try:
            data = _load("tas", "ACCESS-CM2", sc,
                         "2050-01-01", "2099-12-31", **LA_BASIN)
            ts = data.mean(dim=["lat", "lon"]).resample(time="1YS").mean().compute()
            vals = ts.values
            if float(vals.mean()) > 200:
                vals = vals - 273.15
            if ref is None:
                ref = float(vals[0])
            anom = vals - ref
            # 5-yr rolling mean for smoother visual
            kernel = np.ones(5) / 5
            smoothed = np.convolve(anom, kernel, mode="same")
            ax.plot(ts.time.values, anom, color=color, linewidth=0.6, alpha=0.35)
            ax.plot(ts.time.values, smoothed, color=color, linewidth=2.0, label=sc)
            plotted += 1
        except Exception as e:
            print(f"[deck]   scenario {sc} skipped: {e}")
    if plotted == 0:
        raise RuntimeError("no scenarios plotted")
    ax.set_xlabel("Year")
    ax.set_ylabel("Δ Temperature vs 2050 (°C)")
    ax.set_title("Scenario divergence — ACCESS-CM2 LA-area tas")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_scenario_fan.png")


def _real_time_of_emergence():
    hist = _load("tas", "ACCESS-CM2", "historical",
                 "1980-01-01", "1999-12-31", **WESTERN_US)
    # ssp245 (Middle of the Road) gives a slower, more pedagogical emergence pattern —
    # under ssp585 the signal exceeds even 2σ across the entire Western US by ~2025.
    fut = _load("tas", "ACCESS-CM2", "ssp245",
                "2015-01-01", "2099-12-31", **WESTERN_US)
    print("[deck] Computing time-of-emergence map...")
    ah = hist.resample(time="1YS").mean().compute()
    af = fut.resample(time="1YS").mean().compute()
    mu0 = ah.mean(dim="time")
    sigma0 = ah.std(dim="time")
    # Require full 20-yr window (no edge bias) and use a stricter 2σ threshold so
    # the emergence year actually varies across cells instead of pegging at the
    # first valid year everywhere.
    rolled = af.rolling(time=20, center=True, min_periods=20).mean()
    deviation = np.abs(rolled - mu0)
    threshold = 2.0 * sigma0
    emerged = deviation > threshold
    years_da = xr.DataArray(rolled["time"].dt.year.values, dims=["time"], coords={"time": rolled.time})
    emergence_year = years_da.where(emerged).min(dim="time", skipna=True).compute()

    arr, lons = _normalize_lons_arr(emergence_year.values, emergence_year.lon.values)
    lats = emergence_year.lat.values
    # Replace any non-finite (never-emerged or oddities) with NaN, set explicit vmin/vmax
    arr = np.where(np.isfinite(arr), arr, np.nan)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        vmin = float(np.nanpercentile(finite, 5))
        vmax = float(np.nanpercentile(finite, 95))
        if vmax - vmin < 5:
            vmax = vmin + 10
    else:
        vmin, vmax = 2030, 2090
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(lons, lats, arr, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Western US ToE — ACCESS-CM2 SSP2-4.5 (2015–2099)\n"
                 "year |Δtas| > 2σ baseline (1980–1999)")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Emergence year")
    return _save(fig, "fig_time_of_emergence.png")


def _real_change_and_agreement():
    models = ["ACCESS-CM2", "GFDL-ESM4", "MRI-ESM2-0"]
    changes = []
    used = []
    for m in models:
        try:
            h = _load("tas", m, "historical", "1990-01-01", "1999-12-31", **WESTERN_US)
            f = _load("tas", m, "ssp585", "2090-01-01", "2099-12-31", **WESTERN_US)
            change = (f.mean(dim="time") - h.mean(dim="time")).compute()
            # Drop time coord (already reduced); keep only spatial
            change = change.reset_coords(drop=True) if hasattr(change, "reset_coords") else change
            changes.append(change)
            used.append(m)
        except Exception as e:
            print(f"[deck]   change member {m} skipped: {e}")
    if len(changes) < 2:
        raise RuntimeError("need ≥2 models for agreement")
    stack = xr.concat(changes, dim=xr.Variable("model", used), join="outer")
    ens_mean = stack.mean(dim="model")
    ens_sign = np.sign(ens_mean)
    same_sign = (np.sign(stack) == ens_sign)
    agreement = same_sign.sum(dim="model") / len(used)
    # Mask cells where any model is NaN (ocean / outside coverage)
    valid = stack.notnull().all(dim="model")
    agreement = agreement.where(valid)

    arr_mean, lons = _normalize_lons_arr(ens_mean.values, ens_mean.lon.values)
    arr_agree, _ = _normalize_lons_arr(agreement.values, agreement.lon.values)
    lats = ens_mean.lat.values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    vlim = float(np.nanmax(np.abs(arr_mean)))
    im1 = axes[0].pcolormesh(lons, lats, arr_mean, cmap="RdBu_r",
                             vmin=-vlim, vmax=vlim, shading="auto")
    axes[0].set_title(f"Ensemble-mean ΔT — Western US ({len(used)} models)\n"
                      "2090-99 SSP5-8.5 vs 1990-99 historical")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], pad=0.02, label="K")

    im2 = axes[1].pcolormesh(lons, lats, arr_agree, cmap="YlGnBu", vmin=0, vmax=1, shading="auto")
    axes[1].set_title("Model agreement on sign")
    axes[1].set_xlabel("Longitude")
    plt.colorbar(im2, ax=axes[1], pad=0.02, label="fraction agreeing")
    fig.tight_layout()
    return _save(fig, "fig_change_and_agreement.png")


# ----------------------------------------------------------------------
# Synthetic fallback generators (used when real data is unavailable)
# ----------------------------------------------------------------------


def _synth_anomaly_hovmoller():
    """Standardized-anomaly Hovmöller (time × longitude) with detected events boxed."""
    n_t, n_lon = 240, 120
    times = np.arange("2050-01", "2070-01", dtype="datetime64[M]")[:n_t]
    lons = np.linspace(-130, -110, n_lon)  # California-ish
    base = 0.4 * _RNG.standard_normal((n_t, n_lon))

    # Inject coherent extreme events (positive and negative blobs)
    events = [
        (40, 60, 3.4, 18, 30),   # (t0, lon_idx0, peak, dt, dlon)
        (90, 25, -2.8, 12, 25),
        (150, 80, 3.0, 22, 35),
        (200, 40, -2.2, 10, 20),
    ]
    for t0, x0, peak, dt, dx in events:
        for ti in range(max(0, t0), min(n_t, t0 + dt)):
            for xi in range(max(0, x0), min(n_lon, x0 + dx)):
                rt = (ti - t0 - dt / 2) / (dt / 2)
                rx = (xi - x0 - dx / 2) / (dx / 2)
                base[ti, xi] += peak * np.exp(-(rt ** 2 + rx ** 2))

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    vlim = 3.5
    im = ax.pcolormesh(lons, times, base, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    # Draw event boxes
    for t0, x0, peak, dt, dx in events:
        x_left, x_right = lons[x0], lons[min(n_lon - 1, x0 + dx - 1)]
        t_top, t_bot = times[t0], times[min(n_t - 1, t0 + dt - 1)]
        ax.plot(
            [x_left, x_right, x_right, x_left, x_left],
            [t_top, t_top, t_bot, t_bot, t_top],
            color="black", linewidth=1.3, linestyle="--",
        )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Time")
    ax.set_title("Standardized anomaly Hovmöller — extreme events outlined")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("z-score (σ)")
    return _save(fig, "fig_anomaly_hovmoller.png")


def _synth_eof():
    """Two-panel EOF: leading spatial mode (dipole) + PC1 time series."""
    lats = np.linspace(-20, 20, 50)
    lons = np.linspace(150, 270, 130)
    LON, LAT = np.meshgrid(lons, lats)
    pattern = -np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(2 * (LON - 200)))

    n_t = 360
    months = np.arange("1980-01", "2010-01", dtype="datetime64[M]")[:n_t]
    pc1 = (
        1.5 * np.sin(2 * np.pi * np.arange(n_t) / 60.0)  # 5-yr-ish ENSO-like
        + 0.6 * _RNG.standard_normal(n_t)
    )

    fig = plt.figure(figsize=(11, 4.5))
    ax_map = fig.add_subplot(1, 2, 1)
    vlim = float(np.nanmax(np.abs(pattern)))
    im = ax_map.pcolormesh(lons, lats, pattern, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    ax_map.set_xlabel("Longitude (°)")
    ax_map.set_ylabel("Latitude (°)")
    ax_map.set_title("EOF1 — leading spatial mode (42% variance)")
    plt.colorbar(im, ax=ax_map, pad=0.02, label="loading")

    ax_ts = fig.add_subplot(1, 2, 2)
    ax_ts.plot(months, pc1, color="#0f4c81", linewidth=1.2)
    ax_ts.axhline(0, color="grey", linewidth=0.6)
    ax_ts.fill_between(months, pc1, 0, where=pc1 > 0, color="#c4321c", alpha=0.3)
    ax_ts.fill_between(months, pc1, 0, where=pc1 < 0, color="#1a73c2", alpha=0.3)
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("PC1 amplitude")
    ax_ts.set_title("Principal component time series")
    ax_ts.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_eof.png")


def _synth_ensemble_spread():
    """Ensemble spread plot: median + 25-75 + 5-95 bands across 8 model trajectories."""
    n_t = 100
    years = np.arange(2000, 2100)
    # Shared signal (warming) + per-model spread that grows over time
    warming = np.linspace(0, 4.5, n_t)
    n_models = 8
    members = []
    for k in range(n_models):
        m_offset = 0.4 * _RNG.standard_normal()
        slope_var = 1 + 0.18 * _RNG.standard_normal()
        noise = 0.25 * _RNG.standard_normal(n_t)
        members.append(m_offset + slope_var * warming + noise)
    members = np.array(members)
    p05, p25, p50, p75, p95 = np.percentile(members, [5, 25, 50, 75, 95], axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(years, p05, p95, color="#1f77b4", alpha=0.18, label="5–95% across models")
    ax.fill_between(years, p25, p75, color="#1f77b4", alpha=0.3, label="25–75%")
    for m in members:
        ax.plot(years, m, color="grey", linewidth=0.45, alpha=0.5)
    ax.plot(years, p50, color="#1f77b4", linewidth=2.0, label="Median")
    ax.set_xlabel("Year")
    ax.set_ylabel("Δ Temperature (°C)")
    ax.set_title("Multi-model ensemble spread — SSP5-8.5 (8 models)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_ensemble_spread.png")


def _synth_scenario_fan():
    """Scenario fan chart: SSP1-2.6 / 2-4.5 / 5-8.5 with 5-95% bands."""
    n_t = 100
    years = np.arange(2000, 2100)
    fig, ax = plt.subplots(figsize=(9, 5))

    scenarios = [
        ("ssp126", 1.4, "#1a9850"),
        ("ssp245", 2.7, "#fdae61"),
        ("ssp585", 4.6, "#7f0000"),
    ]
    n_models = 6
    for label, end_warming, color in scenarios:
        signal = np.linspace(0, end_warming, n_t)
        members = np.array([
            (1 + 0.15 * _RNG.standard_normal()) * signal
            + 0.35 * _RNG.standard_normal() * np.linspace(0, 1, n_t)
            + 0.18 * _RNG.standard_normal(n_t)
            for _ in range(n_models)
        ])
        med = np.median(members, axis=0)
        lo = np.percentile(members, 5, axis=0)
        hi = np.percentile(members, 95, axis=0)
        ax.fill_between(years, lo, hi, color=color, alpha=0.18)
        ax.plot(years, med, color=color, linewidth=1.8, label=f"{label} (median, 5–95%)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Δ Temperature vs 2000 (°C)")
    ax.set_title("Scenario fan chart — divergence across SSP pathways")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_scenario_fan.png")


def _synth_emergence():
    """Time-of-emergence map (year when |signal| > σ_threshold * baseline std)."""
    lats = np.linspace(-60, 75, 70)
    lons = np.linspace(-180, 180, 140)
    LON, LAT = np.meshgrid(lons, lats)
    # Tropics emerge first; arctic emerges later but strongly; mid-latitudes mixed
    base = 2030 + 30 * np.abs(LAT) / 75 + 4 * _RNG.standard_normal(LAT.shape)
    base = np.clip(base, 2025, 2095)
    # Add some never-emerged speckle
    mask = _RNG.random(LAT.shape) < 0.04
    base = np.where(mask, np.nan, base)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(lons, lats, base, cmap="viridis", vmin=2030, vmax=2090, shading="auto")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Time of emergence — year |signal| > 1σ baseline natural variability")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Emergence year")
    return _save(fig, "fig_time_of_emergence.png")


def _synth_change_and_agreement():
    """Two-panel: ensemble-mean change (left) + model agreement on sign (right)."""
    lats = np.linspace(-60, 75, 70)
    lons = np.linspace(-180, 180, 140)
    LON, LAT = np.meshgrid(lons, lats)
    # Change: positive everywhere, larger at high lats (polar amplification)
    change = 1.0 + 4.5 * np.abs(LAT) / 75 + 0.6 * _RNG.standard_normal(LAT.shape)
    # Agreement: high in tropics & arctic, lower over mid-latitudes & oceans
    agree = 0.6 + 0.35 * (1 - np.exp(-((LAT - 30) / 25) ** 2))
    agree += 0.05 * _RNG.standard_normal(LAT.shape)
    agree = np.clip(agree, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    vlim = float(np.nanmax(np.abs(change)))
    im1 = axes[0].pcolormesh(lons, lats, change, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    axes[0].set_title("Ensemble-mean ΔT (°C, late-century vs baseline)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], pad=0.02, label="°C")

    im2 = axes[1].pcolormesh(lons, lats, agree, cmap="YlGnBu", vmin=0, vmax=1, shading="auto")
    axes[1].set_title("Model agreement on sign (IPCC stippling)")
    axes[1].set_xlabel("Longitude")
    plt.colorbar(im2, ax=axes[1], pad=0.02, label="fraction agreeing")

    fig.tight_layout()
    return _save(fig, "fig_change_and_agreement.png")


# ----------------------------------------------------------------------
# Dispatchers: try real data, fall back to synthetic on any failure
# ----------------------------------------------------------------------
def _try_real(real_fn, synth_fn, label):
    if _REAL_AVAILABLE:
        try:
            print(f"\n[deck] === Building '{label}' from real RCMES data ===")
            t0 = __import__("time").perf_counter()
            path = real_fn()
            elapsed = round(__import__("time").perf_counter() - t0, 1)
            print(f"[deck]   ✓ {label} done in {elapsed}s → {path}")
            return path
        except Exception as e:
            print(f"[deck]   ✗ {label} real-data path failed: {e}")
            print(f"[deck]   falling back to synthetic for {label}")
    return synth_fn()


def make_anomaly_hovmoller_figure():
    return _try_real(_real_anomaly_hovmoller, _synth_anomaly_hovmoller, "anomaly Hovmöller")


def make_eof_figure():
    return _try_real(_real_eof, _synth_eof, "EOF")


def make_ensemble_spread_figure():
    return _try_real(_real_ensemble_spread, _synth_ensemble_spread, "ensemble spread")


def make_scenario_fan_figure():
    return _try_real(_real_scenario_fan, _synth_scenario_fan, "scenario fan")


def make_emergence_figure():
    return _try_real(_real_time_of_emergence, _synth_emergence, "time-of-emergence")


def make_change_and_agreement_figure():
    return _try_real(_real_change_and_agreement, _synth_change_and_agreement, "change + agreement")


# Generate all figures up front so slide code can reference paths
FIG_ANOMALY = make_anomaly_hovmoller_figure()
FIG_EOF = make_eof_figure()
FIG_ENSEMBLE = make_ensemble_spread_figure()
FIG_SCENARIO = make_scenario_fan_figure()
FIG_TOE = make_emergence_figure()
FIG_AGREEMENT = make_change_and_agreement_figure()


# ============================================================
# SLIDE 1: Cover — Cover_No Image layout
# ph[15]=section label, ph[16]=title, ph[17]=subtitle, ph[18]=author, ph[20]=footer
# ============================================================
slide = prs.slides.add_slide(COVER_NO_IMAGE)
set_placeholder_text(slide, 15, "Project Update")
set_placeholder_text(slide, 16, "RCMES-MCP")
set_placeholder_text(slide, 17, "Enabling AI-Driven Climate Analysis\nwith NASA's NEX-GDDP-CMIP6 Data")
set_placeholder_text(slide, 18, "March 2026")
set_placeholder_text(slide, 20, FOOTER_TEXT)


# ============================================================
# SLIDE 2: What is RCMES-MCP? — Title and Content
# ph[17]=section, ph[0]=title, ph[19]=content, ph[21]=footer
# ============================================================
slide = prs.slides.add_slide(TITLE_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "What is RCMES-MCP?")
set_footer(slide)

ph = None
for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        ph = p
        break
tf = ph.text_frame
add_bullet_content(tf, [
    "An MCP (Model Context Protocol) server that exposes NASA's climate analysis tools to AI agents",
    "Provides direct access to the NEX-GDDP-CMIP6 dataset — 38 TB of downscaled global climate projections at 0.25° resolution",
    "Enables conversational climate analysis: users ask questions in natural language, AI executes the science",
    "Built on NASA JPL's Regional Climate Model Evaluation System (RCMES) heritage",
    "No data downloads, no coding required — AI handles data access, processing, analysis, and visualization",
], font_size=Pt(14))


# ============================================================
# SLIDE 3: The Data — Two Content layout
# ph[17]=section, ph[0]=title, ph[14]=subtitle, ph[19]=left content, ph[20]=right content
# ============================================================
slide = prs.slides.add_slide(TWO_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "The Data: NEX-GDDP-CMIP6")
set_placeholder_text(slide, 14, "38 TB of bias-corrected, downscaled CMIP6 climate projections hosted on AWS Open Data")
set_footer(slide)

# Left content - dataset details
for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        add_bullet_content(tf, [
            "0.25° resolution (~25 km), daily timestep",
            "Global coverage, 1950–2100",
            "35 CMIP6 climate models",
            "Historical + 4 SSP scenarios",
            "Accessed via S3 in Zarr format — no downloads",
        ], font_size=Pt(13))

# Right content - variables
for p in slide.placeholders:
    if p.placeholder_format.idx == 20:
        tf = p.text_frame
        items = [
            "tas / tasmax / tasmin — Temperature",
            "pr — Precipitation",
            "hurs / huss — Humidity",
            "sfcWind — Surface wind speed",
            "rsds / rlds — Radiation (SW & LW)",
        ]
        # Add header
        p0 = tf.paragraphs[0]
        p0.text = "9 Climate Variables:"
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = FONT
        p0.space_after = Pt(4)
        for item in items:
            pp = tf.add_paragraph()
            pp.text = item
            pp.font.size = Pt(12)
            pp.font.color.rgb = DARK_GRAY
            pp.font.name = FONT
            pp.space_after = Pt(3)


# ============================================================
# SLIDE 4: Capabilities — Title and Content
# ============================================================
slide = prs.slides.add_slide(TITLE_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "42+ Climate Analysis Tools")
set_footer(slide)

for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        items = [
            "Data Access — Load climate data by model, variable, scenario; region/country selection; multi-model ensemble loading",
            "Processing — Temporal & spatial subsetting, regridding, unit conversion, anomaly & standardized-anomaly, country masking",
            "Statistical Analysis — Climatology, trend with significance, bias, RMSE, correlation, EOF/PCA decomposition",
            "Spatiotemporal Detection — 3D connected-component extreme-event cataloguing, time of emergence",
            "Ensemble & Scenario — Multi-model statistics with IPCC-style agreement maps, scenario-comparison fan charts",
            "Extreme Indices — 22 ETCCDI indices, heatwave & drought analysis",
            "Visualization — Spatial maps, time series, Hovmöller diagrams, fan charts, ensemble spread plots, Taylor diagrams",
        ]
        add_bullet_content(tf, items, font_size=Pt(13))


# ============================================================
# SLIDE 5: How It Works — Title and Content
# ============================================================
slide = prs.slides.add_slide(TITLE_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "How It Works")
set_footer(slide)

for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        items = [
            "User asks a climate question in natural language (e.g., \"What is the heatwave trend in California?\")",
            "AI agent selects and chains appropriate tools via the MCP protocol",
            "Tools access NEX-GDDP-CMIP6 data directly from AWS S3 — no downloads needed",
            "Analysis engine (xarray, Dask, xclim, SciPy) processes the data",
            "Results returned as statistics, maps, time series plots, or derived datasets",
            "Chainable operations: each tool returns a dataset_id for multi-step workflows",
        ]
        add_bullet_content(tf, items, font_size=Pt(13))

# Add flow diagram below using textboxes
# User → MCP Server → AWS S3 → Analysis → Results
flow_labels = ["User / AI Agent", "MCP Server", "AWS S3 Data", "Analysis Engine", "Results"]
flow_colors = [
    RGBColor(0x00, 0x7D, 0xBA),  # blue
    NAVY,
    RGBColor(0x23, 0x72, 0x3B),  # green
    RGBColor(0xE6, 0x7E, 0x22),  # orange
    RGBColor(0xC0, 0x39, 0x2B),  # red
]
box_w = Inches(1.65)
box_h = Inches(0.45)
gap = Inches(0.18)
total = len(flow_labels) * box_w + (len(flow_labels) - 1) * gap
start_x = (prs.slide_width - total) // 2
y = Inches(4.55)

for i, (label, color) in enumerate(zip(flow_labels, flow_colors)):
    x = start_x + i * (box_w + gap)
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    stf = shape.text_frame
    stf.vertical_anchor = MSO_ANCHOR.MIDDLE
    sp = stf.paragraphs[0]
    sp.text = label
    sp.font.size = Pt(9)
    sp.font.bold = True
    sp.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    sp.font.name = FONT
    sp.alignment = PP_ALIGN.CENTER

    if i < len(flow_labels) - 1:
        ax = x + box_w + Emu(10000)
        ay = y + box_h // 2 - Inches(0.08)
        arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, ax, ay, gap - Emu(20000), Inches(0.16))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = RGBColor(0x99, 0x99, 0x99)
        arrow.line.fill.background()


# ============================================================
# SLIDE 6: Example Use Case — Title, Subtitle and Content
# ============================================================
slide = prs.slides.add_slide(TITLE_SUB_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "Example: Heatwave Trend Analysis")
set_placeholder_text(slide, 14, "\"What is the heatwave trend in California under the high-emissions scenario?\"")
set_footer(slide)

for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        steps = [
            "1. AI loads data: load_climate_data(tasmax, ACCESS-CM2, ssp585, California, 2015–2100)",
            "2. AI analyzes heatwaves: analyze_heatwaves(dataset_id, threshold_percentile=90)",
            "3. AI computes trend: calculate_trend(heatwave_id) → significant warming trend with p-value",
            "4. AI visualizes: generate_timeseries_plot(dataset_id, show_trend=True) → plot returned to user",
            "",
            "All steps executed automatically — the user only asks the question",
        ]
        add_bullet_content(tf, steps, font_size=Pt(13))
        # Make last item italic
        last_p = tf.paragraphs[-1]
        last_p.font.italic = True
        last_p.font.color.rgb = RGBColor(0x00, 0x7D, 0xBA)


# ============================================================
# Helper for figure-backed analytics slides (TITLE_ONLY layout)
# Layout: figure on the left half, bullet text on the right.
# ============================================================
def add_figure_slide(title_text, image_path, bullet_items,
                     img_left=Inches(0.35), img_top=Inches(1.55),
                     img_width=Inches(7.6),
                     text_left=Inches(8.2), text_top=Inches(1.55),
                     text_width=Inches(4.85), text_height=Inches(5.4)):
    s = prs.slides.add_slide(TITLE_ONLY)
    set_placeholder_text(s, 17, "RCMES-MCP")
    set_placeholder_text(s, 0, title_text)
    set_footer(s)
    s.shapes.add_picture(image_path, img_left, img_top, width=img_width)

    txt_box = s.shapes.add_textbox(text_left, text_top, text_width, text_height)
    tf = txt_box.text_frame
    tf.word_wrap = True
    add_bullet_content(tf, bullet_items, font_size=Pt(13), spacing=Pt(8))
    return s


# ============================================================
# SLIDE 7: Advanced Spatiotemporal Analysis — overview
# ============================================================
slide = prs.slides.add_slide(TITLE_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "Advanced Spatiotemporal Analysis")
set_footer(slide)

for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        items = [
            "Pillar 1 — Anomaly detection & dominant patterns: standardized z-score anomalies, "
            "3D connected-component extreme-event cataloguing, EOF/PCA decomposition, Hovmöller visualization",
            "Pillar 2 — Multi-model ensembles & scenario comparison: parallel multi-model loading, "
            "ensemble statistics with IPCC-style agreement maps, pairwise scenario differences, "
            "fan charts and ensemble spread plots, time-of-emergence mapping",
            "10 new tools added to the agent toolkit — chainable through the same dataset_id pattern",
            "Surfaces standard climate-research diagnostics that previously required custom scripting",
        ]
        add_bullet_content(tf, items, font_size=Pt(13))


# ============================================================
# SLIDE 8: Spatiotemporal Extreme-Event Detection
# ============================================================
add_figure_slide(
    "Spatiotemporal Extreme-Event Detection",
    FIG_ANOMALY,
    [
        "calculate_standardized_anomaly — z-scores per day-of-year (or month/season) "
        "vs a baseline window, comparable across regions and seasons",
        "detect_extreme_events — 3D connected-component labeling on |z| > N·σ; "
        "returns an event catalogue (start/end, peak σ, location, footprint, duration)",
        "Use case: heatwave / cold-snap / wet-spell / drought hunting under SSP scenarios",
        "generate_hovmoller — time × lat (or × lon) view that reveals propagating "
        "anomalies and persistent regimes at a glance",
    ],
)


# ============================================================
# SLIDE 9: Dominant Variability Patterns (EOF / PCA)
# ============================================================
add_figure_slide(
    "Dominant Variability Patterns (EOF / PCA)",
    FIG_EOF,
    [
        "calculate_eof — cosine-latitude-weighted SVD that returns the leading "
        "spatial modes plus their principal-component time series",
        "Surfaces dominant patterns of variability "
        "(teleconnection-like dipoles, seasonal cycles, regime shifts)",
        "Two chained dataset_ids: spatial modes (mode, lat, lon) and PCs (time, mode) — "
        "feed straight into generate_map and generate_timeseries_plot",
        "Reports variance explained per mode and cumulative variance",
    ],
    img_width=Inches(7.6),
)


# ============================================================
# SLIDE 10: Multi-Model Ensemble Loading & Spread
# ============================================================
add_figure_slide(
    "Multi-Model Ensembles — Quantifying Uncertainty",
    FIG_ENSEMBLE,
    [
        "load_multi_model_ensemble — parallel-loads up to 10 CMIP6 models for the "
        "same variable/scenario/region into one dataset with a model dimension",
        "generate_ensemble_spread_plot — median + 25–75% inner band + 5–95% outer band "
        "across models over time (with optional per-model overlays)",
        "Spread is the answer to \"how much do models disagree?\" — wide bands signal "
        "low confidence, tight bands signal robust signal",
        "Cap of 10 protects RAM; pre-subset region & time before loading",
    ],
)


# ============================================================
# SLIDE 11: Ensemble Statistics & IPCC-style Agreement
# ============================================================
add_figure_slide(
    "Ensemble Statistics & Model Agreement",
    FIG_AGREEMENT,
    [
        "calculate_ensemble_statistics — reduces ensemble across the model dimension "
        "into ensemble mean, spread (σ), and min/max envelope dataset_ids",
        "With a baseline window, also returns an agreement map: per-cell fraction of "
        "models agreeing with the ensemble-mean SIGN of change — IPCC stippling",
        "Left: ensemble-mean ΔT (late-century vs baseline). Right: model agreement "
        "(0 = full disagreement, 1 = all models agree)",
        "Standard inputs for confidence-aware climate assessments",
    ],
)


# ============================================================
# SLIDE 12: Scenario Comparison
# ============================================================
add_figure_slide(
    "Scenario Comparison Across SSP Pathways",
    FIG_SCENARIO,
    [
        "compare_scenarios — loads the same variable/model/region across multiple "
        "SSP scenarios (max 5) and computes pairwise time-mean differences",
        "generate_scenario_fan_chart — multi-line projection plot with shaded "
        "5–95% bands per scenario; the canonical IPCC-style figure",
        "Reveals when low- vs high-emission pathways diverge — central to mitigation messaging",
        "Smoothing window option for decadal-scale signals; pairs well with "
        "generate_map on each pairwise difference",
    ],
)


# ============================================================
# SLIDE 13: Time of Emergence
# ============================================================
add_figure_slide(
    "Time of Emergence — When Does Climate Change Arrive?",
    FIG_TOE,
    [
        "calculate_time_of_emergence — per grid cell, the first year that the "
        "rolling-mean signal exceeds N·σ of baseline natural variability",
        "Returns a 2D (lat, lon) map of emergence years (NaN where the signal "
        "never emerges within the window)",
        "Tropics typically emerge first (low natural variability); high latitudes "
        "later but with larger absolute change — the colour gradient tells the story",
        "Useful for impact-assessment timelines and adaptation planning",
    ],
)


# ============================================================
# SLIDE 14: Current Status — Two Content
# (was SLIDE 7 — renumbered)
# ============================================================
slide = prs.slides.add_slide(TWO_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "Current Status")
set_placeholder_text(slide, 14, "")
set_footer(slide)

# Left - accomplishments
for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        p0 = tf.paragraphs[0]
        p0.text = "Accomplishments"
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = FONT
        for item in [
            "32+ MCP tools: full climate analysis pipeline",
            "Operational 38 TB data access via AWS S3",
            "22 ETCCDI extreme climate indices",
            "Interactive web UI (React + FastAPI)",
            "Multiple deployment modes (MCP, Web, API, Docker)",
            "Session management with chainable operations",
            "Country-level masking & geographic selection",
        ]:
            pp = tf.add_paragraph()
            pp.text = item
            pp.font.size = Pt(11)
            pp.font.color.rgb = DARK_GRAY
            pp.font.name = FONT
            pp.space_after = Pt(3)

# Right - tech stack
for p in slide.placeholders:
    if p.placeholder_format.idx == 20:
        tf = p.text_frame
        p0 = tf.paragraphs[0]
        p0.text = "Technology Stack"
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = FONT
        stack = [
            ("Core:", "Python 3.10+, xarray, Dask, zarr"),
            ("Climate:", "xclim, SciPy, NetCDF4"),
            ("Geospatial:", "Shapely, GeoPandas, Cartopy"),
            ("Server:", "FastMCP, FastAPI, Uvicorn"),
            ("Frontend:", "React, TypeScript"),
            ("Data:", "AWS S3, NEX-GDDP-CMIP6"),
            ("AI:", "MCP Protocol, Azure OpenAI"),
        ]
        for label, tech in stack:
            pp = tf.add_paragraph()
            run1 = pp.add_run()
            run1.text = f"{label} "
            run1.font.size = Pt(11)
            run1.font.bold = True
            run1.font.color.rgb = RGBColor(0x00, 0x7D, 0xBA)
            run1.font.name = FONT
            run2 = pp.add_run()
            run2.text = tech
            run2.font.size = Pt(11)
            run2.font.color.rgb = DARK_GRAY
            run2.font.name = FONT
            pp.space_after = Pt(3)


# ============================================================
# SLIDE 15: Path Forward — Two Content
# ============================================================
slide = prs.slides.add_slide(TWO_CONTENT)
set_placeholder_text(slide, 17, "RCMES-MCP")
set_placeholder_text(slide, 0, "Path Forward & Continued Support")
set_placeholder_text(slide, 14, "")
set_footer(slide)

# Left - growth opportunities
for p in slide.placeholders:
    if p.placeholder_format.idx == 19:
        tf = p.text_frame
        p0 = tf.paragraphs[0]
        p0.text = "Growth Opportunities"
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = FONT
        for item in [
            "Expand to additional NASA datasets (MERRA-2, ECCO, GLDAS)",
            "Model evaluation with observational datasets",
            "NASA Earthdata integration for authenticated access",
            "Multi-model ensemble analysis & weighting",
            "Community tool for broader climate science use",
            "Reproducible climate assessments for policy support",
        ]:
            pp = tf.add_paragraph()
            pp.text = item
            pp.font.size = Pt(11)
            pp.font.color.rgb = DARK_GRAY
            pp.font.name = FONT
            pp.space_after = Pt(3)

# Right - impact
for p in slide.placeholders:
    if p.placeholder_format.idx == 20:
        tf = p.text_frame
        p0 = tf.paragraphs[0]
        p0.text = "What Continued Support Enables"
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = FONT
        for item in [
            "Democratize access to NASA climate data for non-experts",
            "Accelerate climate research by removing data engineering bottlenecks",
            "Enable rapid AI-assisted climate impact assessments",
            "Lower the barrier to entry across disciplines",
            "Position NASA JPL at the forefront of AI-enabled science",
            "Support decision-makers with actionable climate intelligence",
        ]:
            pp = tf.add_paragraph()
            pp.text = item
            pp.font.size = Pt(11)
            pp.font.color.rgb = DARK_GRAY
            pp.font.name = FONT
            pp.space_after = Pt(3)


# ============================================================
# SLIDE 16: Closing slide
# ============================================================
slide = prs.slides.add_slide(CLOSING)
set_footer(slide)


# ============================================================
# Save
# ============================================================
prs.save(OUTPUT)
print(f"Saved: {OUTPUT}")
print(f"Slides: {len(prs.slides)}")
