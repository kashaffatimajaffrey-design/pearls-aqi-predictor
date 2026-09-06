"""US EPA Air Quality Index computation from raw pollutant concentrations.

APIs hand us concentrations in ug/m3. The EPA AQI is defined on a per-pollutant
sub-index basis; the reported AQI is the maximum sub-index and the "dominant
pollutant" is the one that produced it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Molecular weights (g/mol) for ug/m3 -> ppb/ppm conversion at 25 C, 1 atm.
_MOLAR_VOLUME = 24.45
_MW = {"o3": 48.00, "no2": 46.0055, "so2": 64.066, "co": 28.010}

# (C_low, C_high, I_low, I_high) -- EPA 2024 revision for PM2.5.
BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    # PM2.5, 24h avg, ug/m3
    "pm2_5": [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500),
    ],
    # PM10, 24h avg, ug/m3
    "pm10": [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500),
    ],
    # O3, 8h avg, ppb
    "o3": [
        (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
        (86, 105, 151, 200), (106, 200, 201, 300),
    ],
    # NO2, 1h, ppb
    "no2": [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500),
    ],
    # SO2, 1h, ppb
    "so2": [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500),
    ],
    # CO, 8h, ppm
    "co": [
        (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500),
    ],
}

CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activity.",
    "Moderate": "Unusually sensitive people should consider limiting prolonged exertion outdoors.",
    "Unhealthy for Sensitive Groups": "Children, older adults and people with heart/lung disease should reduce prolonged outdoor exertion.",
    "Unhealthy": "Everyone should reduce prolonged outdoor exertion. Sensitive groups should stay indoors.",
    "Very Unhealthy": "Health alert -- avoid outdoor activity. Use air purifiers indoors.",
    "Hazardous": "Emergency conditions. Everyone should remain indoors with windows closed.",
}


def ugm3_to_ppb(value: float | np.ndarray, pollutant: str):
    """Convert ug/m3 to ppb (or ppm for CO) at 25 C and 1 atm."""
    mw = _MW[pollutant]
    ppb = np.asarray(value, dtype=float) * _MOLAR_VOLUME / mw
    return ppb / 1000.0 if pollutant == "co" else ppb  # CO index uses ppm


def _truncate(value: float, pollutant: str) -> float:
    """EPA requires truncating the concentration before table lookup."""
    if pollutant in ("pm2_5", "co"):
        return float(np.floor(value * 10) / 10)
    return float(np.floor(value))


def sub_index(concentration: float, pollutant: str) -> float:
    """Piecewise-linear EPA sub-index. Returns NaN for missing input."""
    if concentration is None or (isinstance(concentration, float) and np.isnan(concentration)):
        return float("nan")
    if pollutant in _MW:
        concentration = float(ugm3_to_ppb(concentration, pollutant))
    c = _truncate(max(concentration, 0.0), pollutant)
    table = BREAKPOINTS[pollutant]
    for c_lo, c_hi, i_lo, i_hi in table:
        if c <= c_hi:
            c = max(c, c_lo)
            return round((i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo)
    return float(table[-1][3])  # above the table -> cap at 500


def compute_aqi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-pollutant sub-indices, overall `aqi` and `dominant_pollutant`.

    Rolling averages match the EPA averaging windows where we have the history
    (24h for particulates, 8h for O3/CO); at the head of the series we fall back
    to the instantaneous reading so early rows are still usable.
    """
    out = df.copy().sort_values("ts").reset_index(drop=True)

    averaged = {}
    for p in ("pm2_5", "pm10"):
        if p in out:
            averaged[p] = out[p].rolling(24, min_periods=1).mean()
    for p in ("o3", "co"):
        if p in out:
            averaged[p] = out[p].rolling(8, min_periods=1).mean()
    for p in ("no2", "so2"):
        if p in out:
            averaged[p] = out[p]

    sub_cols = []
    for p, series in averaged.items():
        col = f"aqi_{p}"
        out[col] = [sub_index(v, p) for v in series]
        sub_cols.append(col)

    if not sub_cols:
        raise ValueError("no pollutant columns available to compute AQI")

    out["aqi"] = out[sub_cols].max(axis=1)
    out["dominant_pollutant"] = (
        out[sub_cols].idxmax(axis=1).str.replace("aqi_", "", regex=False)
    )
    return out


def categorize(aqi: float) -> str:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown"
    for _lo, hi, name, _ in CATEGORIES:
        if aqi <= hi:
            return name
    return "Hazardous"


def category_color(aqi: float) -> str:
    for _lo, hi, _, color in CATEGORIES:
        if aqi <= hi:
            return color
    return "#7e0023"


def advice(aqi: float) -> str:
    return HEALTH_ADVICE.get(categorize(aqi), "No advice available.")
