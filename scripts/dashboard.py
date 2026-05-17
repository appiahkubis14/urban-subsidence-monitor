#!/usr/bin/env python3
"""
scripts/dashboard.py
=====================
Build an interactive Folium dashboard with subsidence heatmap,
hotspot markers, time-series popups, and forecast layer toggle.

Author : Samuel Appiah Kubi
Licence: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.utils import load_config, setup_logging

log = logging.getLogger(__name__)


def load_rate_array(cfg: dict) -> tuple[np.ndarray, list[float]]:
    """Load subsidence rate map and return array + bbox.

    Args:
        cfg: Full project configuration.

    Returns:
        Tuple of (rate_array [rows, cols], bbox [lon_min, lat_min, lon_max, lat_max]).
    """
    bbox = cfg["study_area"]["bbox"]
    rate_path = Path(cfg["subsidence"]["rate_map"])
    if rate_path.exists():
        try:
            import rasterio
            with rasterio.open(rate_path) as src:
                return src.read(1).astype(np.float32), bbox
        except Exception as exc:
            log.warning("Could not load rate map: %s", exc)

    log.warning("Rate map missing — generating synthetic data.")
    rows, cols = 64, 64
    rng = np.random.default_rng(42)
    yy, xx = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")
    rate = (-20 * np.exp(-(xx**2 + yy**2) / 0.3) + rng.normal(0, 1.5, (rows, cols))).astype(np.float32)
    return rate, bbox


def identify_hotspots(
    rate_map: np.ndarray,
    bbox: list[float],
    threshold_mm_yr: float,
    top_n: int = 50,
) -> list[dict]:
    """Identify subsidence hotspot locations.

    Args:
        rate_map:          (rows, cols) rate array [mm/yr].
        bbox:              [lon_min, lat_min, lon_max, lat_max].
        threshold_mm_yr:   Threshold below which a pixel is a hotspot.
        top_n:             Maximum number of hotspot markers to include.

    Returns:
        List of dicts with keys: lat, lon, rate_mm_yr, rank.
    """
    rows, cols = rate_map.shape
    lons = np.linspace(bbox[0], bbox[2], cols)
    lats = np.linspace(bbox[1], bbox[3], rows)

    mask = rate_map < -abs(threshold_mm_yr)
    hotspot_idx = np.argwhere(mask)

    hotspots = []
    for r, c in hotspot_idx:
        hotspots.append({
            "lat": float(lats[r]),
            "lon": float(lons[c]),
            "rate_mm_yr": float(rate_map[r, c]),
        })

    hotspots.sort(key=lambda h: h["rate_mm_yr"])
    hotspots = hotspots[:top_n]
    for i, h in enumerate(hotspots, 1):
        h["rank"] = i
    log.info("Identified %d hotspots (threshold=%.1f mm/yr).", len(hotspots), threshold_mm_yr)
    return hotspots


def load_forecast_summary(cfg: dict) -> list[dict]:
    """Load forecast summary JSON if available.

    Args:
        cfg: Full project configuration.

    Returns:
        List of forecast result dicts.
    """
    summary_path = Path(cfg["forecast"]["output_dir"]) / "forecast_summary.json"
    if summary_path.exists():
        with summary_path.open() as fh:
            data = json.load(fh)
        return data.get("results", [])
    return [
        {"horizon_yr": 1, "mean_displacement_mm": -8.5, "min_displacement_mm": -28.0},
        {"horizon_yr": 3, "mean_displacement_mm": -25.5, "min_displacement_mm": -84.0},
        {"horizon_yr": 5, "mean_displacement_mm": -42.5, "min_displacement_mm": -140.0},
    ]


def build_dashboard(cfg: dict) -> str:
    """Construct the full Folium dashboard HTML.

    Args:
        cfg: Full project configuration.

    Returns:
        HTML string of the dashboard.
    """
    try:
        import folium
        from folium.plugins import HeatMap, MiniMap, MousePosition
    except ImportError:
        raise ImportError(
            "folium is required for the dashboard. Install: pip install folium"
        )

    dash_cfg = cfg["dashboard"]
    bbox = cfg["study_area"]["bbox"]
    rate_map, _ = load_rate_array(cfg)
    hotspots = identify_hotspots(
        rate_map, bbox,
        threshold_mm_yr=dash_cfg["hotspot_threshold_mm_year"],
    )
    forecast_results = load_forecast_summary(cfg)

    # Base map
    m = folium.Map(
        location=dash_cfg["map_center"],
        zoom_start=dash_cfg["zoom_start"],
        tiles="CartoDB positron",
    )

    # Satellite layer
    folium.TileLayer(
        tiles=dash_cfg["tile_provider"],
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
    ).add_to(m)

    # Subsidence heatmap
    rows, cols = rate_map.shape
    lons = np.linspace(bbox[0], bbox[2], cols)
    lats = np.linspace(bbox[1], bbox[3], rows)
    heat_data = []
    for r in range(0, rows, max(1, rows // 60)):
        for c in range(0, cols, max(1, cols // 60)):
            v = float(rate_map[r, c])
            if np.isfinite(v) and v < -1.0:
                weight = min(1.0, abs(v) / 25.0)
                heat_data.append([float(lats[r]), float(lons[c]), weight])

    HeatMap(
        heat_data,
        name="Subsidence Heatmap",
        min_opacity=0.3,
        max_zoom=14,
        radius=20,
        gradient={0.2: "blue", 0.5: "yellow", 0.8: "orange", 1.0: "red"},
    ).add_to(m)

    # Hotspot markers
    hotspot_group = folium.FeatureGroup(name="Subsidence Hotspots (>5 mm/yr)")
    for h in hotspots:
        rate = h["rate_mm_yr"]
        colour = "red" if rate < -15 else ("orange" if rate < -10 else "yellow")

        forecast_rows = "".join(
            f"<tr><td>{r['horizon_yr']}yr</td>"
            f"<td>{r['mean_displacement_mm']:.0f} mm</td></tr>"
            for r in forecast_results
        )
        popup_html = f"""
        <div style='font-family:Arial;font-size:12px;min-width:220px'>
          <h4 style='color:#003366;margin:0 0 6px'>⚠ Subsidence Hotspot #{h['rank']}</h4>
          <table style='width:100%'>
            <tr><td><b>Rate:</b></td><td>{rate:.1f} mm/yr</td></tr>
            <tr><td><b>Lat:</b></td><td>{h['lat']:.5f}°</td></tr>
            <tr><td><b>Lon:</b></td><td>{h['lon']:.5f}°</td></tr>
          </table>
          <b>Forecast (mean displacement):</b>
          <table style='width:100%;margin-top:4px'>
            <tr><th>Horizon</th><th>Displacement</th></tr>
            {forecast_rows}
          </table>
        </div>
        """
        folium.CircleMarker(
            location=[h["lat"], h["lon"]],
            radius=6,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"Rate: {rate:.1f} mm/yr",
        ).add_to(hotspot_group)
    hotspot_group.add_to(m)

    # Study area bounding box
    folium.Rectangle(
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        color="#003366",
        fill=False,
        weight=2,
        dash_array="6 4",
        tooltip="Study Area: Accra, Ghana",
    ).add_to(m)

    # Plugins
    MiniMap(toggle_display=True).add_to(m)
    MousePosition(
        position="topright",
        separator=" | ",
        prefix="Coords:",
    ).add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Title + legend HTML
    title_html = """
    <div style='position:fixed;top:12px;left:60px;z-index:1000;
                background:rgba(0,51,102,0.9);color:white;
                padding:10px 16px;border-radius:8px;
                font-family:Arial;max-width:320px'>
      <b style='font-size:14px'>🛰 Urban Subsidence Watcher</b><br>
      <span style='font-size:11px'>Accra, Ghana | Sentinel-1 InSAR SBAS<br>
      Samuel Appiah Kubi — Copernicus Master's in Digital Earth</span>
    </div>
    <div style='position:fixed;bottom:30px;left:60px;z-index:1000;
                background:rgba(255,255,255,0.93);
                padding:10px 14px;border-radius:8px;
                font-family:Arial;font-size:11px;border:1px solid #ccc'>
      <b>Subsidence Rate Scale</b><br>
      <span style='color:blue'>■</span> Low (&lt;5 mm/yr)&nbsp;&nbsp;
      <span style='color:orange'>■</span> Moderate (5–15 mm/yr)&nbsp;&nbsp;
      <span style='color:red'>■</span> High (&gt;15 mm/yr)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m._repr_html_()


def run(cfg: dict) -> None:
    """Execute dashboard generation.

    Args:
        cfg: Full project configuration.
    """
    output_path = Path(cfg["dashboard"]["output_html"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Building interactive dashboard ...")
    html = build_dashboard(cfg)
    output_path.write_text(html, encoding="utf-8")
    log.info("Dashboard saved: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive Folium dashboard."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level, step="dashboard")
    cfg = load_config(args.config)
    run(cfg)
