"""
utils/weather_fetcher.py
Live weather integration for Flight Delay Prediction System
Uses Open-Meteo API — 100% FREE, no API key, no signup required
https://open-meteo.com/
"""

import requests
import streamlit as st
from datetime import datetime


# ── Airport coordinates ────────────────────────────────────────────────────────
AIRPORT_COORDS = {
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "LAX": {"lat": 33.9425, "lon": -118.4081},
    "ORD": {"lat": 41.9742, "lon": -87.9073},
    "DFW": {"lat": 32.8998, "lon": -97.0403},
    "DEN": {"lat": 39.8561, "lon": -104.6737},
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "SFO": {"lat": 37.6213, "lon": -122.3790},
    "SEA": {"lat": 47.4502, "lon": -122.3088},
    "LAS": {"lat": 36.0840, "lon": -115.1537},
    "MCO": {"lat": 28.4312, "lon": -81.3081},
    "EWR": {"lat": 40.6895, "lon": -74.1745},
    "MIA": {"lat": 25.7959, "lon": -80.2870},
    "PHX": {"lat": 33.4373, "lon": -112.0078},
    "IAH": {"lat": 29.9902, "lon": -95.3368},
    "BOS": {"lat": 42.3656, "lon": -71.0096},
    "MSP": {"lat": 44.8848, "lon": -93.2223},
    "DTW": {"lat": 42.2162, "lon": -83.3554},
    "FLL": {"lat": 26.0726, "lon": -80.1527},
    "PHL": {"lat": 39.8719, "lon": -75.2411},
    "LGA": {"lat": 40.7772, "lon": -73.8726},
    "BWI": {"lat": 39.1754, "lon": -76.6682},
    "SLC": {"lat": 40.7884, "lon": -111.9778},
    "DCA": {"lat": 38.8521, "lon": -77.0377},
    "SAN": {"lat": 32.7338, "lon": -117.1933},
    "MDW": {"lat": 41.7868, "lon": -87.7522},
    "TPA": {"lat": 27.9755, "lon": -82.5332},
    "PDX": {"lat": 45.5898, "lon": -122.5951},
    "HNL": {"lat": 21.3187, "lon": -157.9225},
    "DAL": {"lat": 32.8471, "lon": -96.8518},
    "STL": {"lat": 38.7487, "lon": -90.3700},
}

# ── WMO weather code → human label + icon ─────────────────────────────────────
# Full list: https://open-meteo.com/en/docs#weathervariables
WMO_CODES = {
    0:  ("Clear sky",           "☀️",  "Clear"),
    1:  ("Mainly clear",        "🌤️", "Clear"),
    2:  ("Partly cloudy",       "⛅",  "Clouds"),
    3:  ("Overcast",            "☁️",  "Clouds"),
    45: ("Fog",                 "🌫️", "Fog"),
    48: ("Icy fog",             "🌫️", "Fog"),
    51: ("Light drizzle",       "🌦️", "Drizzle"),
    53: ("Moderate drizzle",    "🌦️", "Drizzle"),
    55: ("Dense drizzle",       "🌧️", "Drizzle"),
    61: ("Slight rain",         "🌧️", "Rain"),
    63: ("Moderate rain",       "🌧️", "Rain"),
    65: ("Heavy rain",          "🌧️", "Rain"),
    71: ("Slight snow",         "❄️",  "Snow"),
    73: ("Moderate snow",       "❄️",  "Snow"),
    75: ("Heavy snow",          "❄️",  "Snow"),
    77: ("Snow grains",         "❄️",  "Snow"),
    80: ("Slight showers",      "🌧️", "Rain"),
    81: ("Moderate showers",    "🌧️", "Rain"),
    82: ("Violent showers",     "🌧️", "Rain"),
    85: ("Slight snow showers", "❄️",  "Snow"),
    86: ("Heavy snow showers",  "❄️",  "Snow"),
    95: ("Thunderstorm",        "⛈️",  "Thunderstorm"),
    96: ("Thunderstorm w/ hail","⛈️",  "Thunderstorm"),
    99: ("Thunderstorm w/ hail","⛈️",  "Thunderstorm"),
}

WEATHER_ICON_MAP = {code: v[1] for code, v in WMO_CODES.items()}


def fetch_weather(iata_code: str) -> dict:
    """
    Fetch current weather for an airport using Open-Meteo (free, no key).
    Falls back to safe defaults if the airport isn't in our coords list or API fails.
    """
    coords = AIRPORT_COORDS.get(iata_code)
    if not coords:
        return _default_weather(iata_code, reason="no_coords")

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":              coords["lat"],
                "longitude":             coords["lon"],
                "current":               [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "surface_pressure",
                    "visibility",
                ],
                "wind_speed_unit":       "ms",   # metres per second
                "timezone":              "auto",
            },
            timeout=6,
        )
        resp.raise_for_status()
        c = resp.json()["current"]

        wmo      = int(c.get("weather_code", 0))
        desc, icon, main = WMO_CODES.get(wmo, ("Unknown", "🌡️", "Clouds"))
        vis_raw  = c.get("visibility", 10000)          # open-meteo gives metres
        # some versions return km — normalise
        vis_m    = vis_raw if vis_raw > 500 else vis_raw * 1000

        return {
            "airport":        iata_code,
            "live":           True,
            "temperature_c":  round(c["temperature_2m"], 1),
            "feels_like_c":   round(c["apparent_temperature"], 1),
            "humidity_pct":   int(c["relative_humidity_2m"]),
            "pressure_hpa":   round(c.get("surface_pressure", 1013), 1),
            "wind_speed_ms":  round(c["wind_speed_10m"], 2),
            "wind_deg":       int(c.get("wind_direction_10m", 0)),
            "visibility_m":   float(vis_m),
            "clouds_pct":     int(c.get("cloud_cover", 0)),
            "precipitation":  round(c.get("precipitation", 0.0), 2),
            "weather_code":   wmo,
            "weather_main":   main,
            "weather_desc":   desc,
            "fetched_at":     datetime.utcnow().strftime("%H:%M UTC"),
        }

    except requests.exceptions.Timeout:
        return _default_weather(iata_code, reason="timeout")
    except requests.exceptions.HTTPError as e:
        return _default_weather(iata_code, reason=f"http_{e.response.status_code}")
    except Exception as e:
        return _default_weather(iata_code, reason="error")


def _default_weather(iata_code: str, reason: str = "unavailable") -> dict:
    """Safe fallback — model still runs, UI shows OFFLINE badge."""
    return {
        "airport":       iata_code,
        "live":          False,
        "reason":        reason,
        "temperature_c": 20.0,
        "feels_like_c":  20.0,
        "humidity_pct":  60,
        "pressure_hpa":  1013.0,
        "wind_speed_ms": 5.0,
        "wind_deg":      0,
        "visibility_m":  10000.0,
        "clouds_pct":    20,
        "precipitation": 0.0,
        "weather_code":  0,
        "weather_main":  "Clear",
        "weather_desc":  "Weather data unavailable",
        "fetched_at":    "N/A",
    }


def get_route_weather(origin: str, dest: str) -> tuple[dict, dict]:
    """Fetch weather for both airports. Returns (origin_wx, dest_wx)."""
    return fetch_weather(origin), fetch_weather(dest)


# ── Feature engineering ────────────────────────────────────────────────────────

def _wind_severity(speed_ms: float) -> int:
    if speed_ms < 5:  return 0
    if speed_ms < 10: return 1
    if speed_ms < 17: return 2
    return 3

def _visibility_cat(vis_m: float) -> int:
    if vis_m < 200:  return 0
    if vis_m < 1000: return 1
    if vis_m < 5000: return 2
    return 3

def _precip_cat(mm: float) -> int:
    if mm == 0:   return 0
    if mm < 2.5:  return 1
    if mm < 7.5:  return 2
    return 3

WEATHER_CODE_MAP = {
    "Clear": 0, "Clouds": 1, "Fog": 3, "Drizzle": 4,
    "Rain": 5, "Snow": 6, "Thunderstorm": 7,
}


def compute_risk_score(origin_wx: dict, dest_wx: dict) -> float:
    """Composite 0–10 weather risk score for the full route."""
    score = 0.0
    for wx in [origin_wx, dest_wx]:
        spd  = wx["wind_speed_ms"]
        vis  = wx["visibility_m"]
        prec = wx["precipitation"]
        cond = wx["weather_main"]

        if spd > 10:               score += 1.5
        if spd > 17:               score += 2.0
        if vis < 5000:             score += 0.5
        if vis < 1000:             score += 1.5
        if vis < 200:              score += 2.5
        if prec > 0:               score += 0.5
        if prec > 2.5:             score += 1.0
        if prec > 7.5:             score += 1.5
        if cond == "Thunderstorm": score += 3.0
        elif cond in ("Snow", "Fog"): score += 2.0
        elif cond in ("Rain", "Drizzle"): score += 1.0

    return min(round(score, 1), 10.0)


def build_weather_features(origin_wx: dict, dest_wx: dict) -> dict:
    """25 model-ready weather features merged into input_data before prediction."""
    return {
        "ORIGIN_TEMP_C":      origin_wx["temperature_c"],
        "ORIGIN_HUMIDITY":    origin_wx["humidity_pct"],
        "ORIGIN_WIND_MS":     origin_wx["wind_speed_ms"],
        "ORIGIN_WIND_SEV":    _wind_severity(origin_wx["wind_speed_ms"]),
        "ORIGIN_VIS_CAT":     _visibility_cat(origin_wx["visibility_m"]),
        "ORIGIN_PRECIP_MM":   origin_wx["precipitation"],
        "ORIGIN_PRECIP_CAT":  _precip_cat(origin_wx["precipitation"]),
        "ORIGIN_WX_CODE":     WEATHER_CODE_MAP.get(origin_wx["weather_main"], 1),
        "ORIGIN_PRESSURE":    origin_wx["pressure_hpa"],
        "ORIGIN_CLOUDS":      origin_wx["clouds_pct"],
        "DEST_TEMP_C":        dest_wx["temperature_c"],
        "DEST_HUMIDITY":      dest_wx["humidity_pct"],
        "DEST_WIND_MS":       dest_wx["wind_speed_ms"],
        "DEST_WIND_SEV":      _wind_severity(dest_wx["wind_speed_ms"]),
        "DEST_VIS_CAT":       _visibility_cat(dest_wx["visibility_m"]),
        "DEST_PRECIP_MM":     dest_wx["precipitation"],
        "DEST_PRECIP_CAT":    _precip_cat(dest_wx["precipitation"]),
        "DEST_WX_CODE":       WEATHER_CODE_MAP.get(dest_wx["weather_main"], 1),
        "DEST_PRESSURE":      dest_wx["pressure_hpa"],
        "DEST_CLOUDS":        dest_wx["clouds_pct"],
        "TEMP_DIFF":          abs(origin_wx["temperature_c"] - dest_wx["temperature_c"]),
        "MAX_WIND_MS":        max(origin_wx["wind_speed_ms"], dest_wx["wind_speed_ms"]),
        "ANY_PRECIP":         int(origin_wx["precipitation"] > 0 or dest_wx["precipitation"] > 0),
        "WORST_VIS_M":        min(origin_wx["visibility_m"], dest_wx["visibility_m"]),
        "ROUTE_WEATHER_RISK": compute_risk_score(origin_wx, dest_wx),
    }
