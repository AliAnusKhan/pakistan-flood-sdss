"""
config.py
---------
Central configuration for the Pakistan Agricultural Flood SDSS dashboard.

Keeping district lists, coordinates, and styling tokens here (instead of
inline inside app.py) means a new district, province, or color scheme can
be added by editing plain data in ONE place, without touching any UI or
GEE logic.
"""

import os

CURRENT_YEAR = 2026

# ----------------------------------------------------------------------------
# Region data
# ----------------------------------------------------------------------------
PROVINCE_DISTRICT_MAP = {
    "Sindh": ["Larkana", "Dadu", "Jacobabad", "Shikarpur", "Kashmore", "Badin", "Sukkur"],
    "Punjab": ["Rajanpur", "Muzaffargarh"],
    "Khyber Pakhtunkhwa": ["Charsadda", "Nowshera"],
    "Balochistan": ["Jaffarabad", "Nasirabad"],
    "All Pakistan": ["All Pakistan"],
}

DISTRICT_COORDS = {
    "Larkana": [27.5580, 68.2120], "Dadu": [26.7303, 67.7769],
    "Jacobabad": [28.2835, 68.4388], "Shikarpur": [27.9571, 68.6381],
    "Kashmore": [28.4332, 69.5852], "Rajanpur": [29.1044, 70.3301],
    "Muzaffargarh": [30.0703, 71.1933], "Jaffarabad": [28.3421, 68.1408],
    "Nasirabad": [28.5800, 68.1700], "Badin": [24.6558, 68.8383],
    "Sukkur": [27.7052, 68.8574], "Charsadda": [34.1482, 71.7406],
    "Nowshera": [34.0153, 71.9747], "All Pakistan": [30.3753, 69.3451],
}

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
SEVERITY_STYLE = {
    "Low":      {"badge": "background:#E7EFE2; color:#3F5A34; border:1px solid #C3D4B4;", "hex": "#4C7A54"},
    "Moderate": {"badge": "background:#F3E7CC; color:#8A5E17; border:1px solid #E3C88C;", "hex": "#B8792A"},
    "High":     {"badge": "background:#F0DAC9; color:#8A3E17; border:1px solid #E0AE85;", "hex": "#A6431C"},
    "Extreme":  {"badge": "background:#EBD1C9; color:#6B1F10; border:1px solid #D69C8B;", "hex": "#7A2418"},
    "Unknown":  {"badge": "background:#E6E7E2; color:#4B5449; border:1px solid #CBCFC3;", "hex": "#6B756A"},
}

ACCENT_ENV = "#1B6E76"      # river teal  - flood & physical parameters
ACCENT_ECO = "#B3872F"      # wheat       - agricultural & economic parameters
ACCENT_NEUTRAL = "#5B6B62"  # slate ink   - statistical / neutral parameters

CHART_PALETTE = ["#1B6E76", "#B3872F", "#5C6B2F", "#A6431C", "#4C7A54", "#6E8FA3", "#8A5E17", "#3F5A34"]
CHART_SEQUENTIAL_TEAL = ["#DCE9E6", "#5B9AA0", "#0E3B36"]
CHART_SEQUENTIAL_WHEAT = ["#F0E4C8", "#C99A3E", "#7A5A17"]

# ----------------------------------------------------------------------------
# Friendly translations for raw backend error/status strings
# ----------------------------------------------------------------------------
FRIENDLY_ERROR_MAP = {
    "not found in GAUL boundary collection": (
        "We couldn't match that district name to a known administrative boundary. "
        "Try selecting it again from the dropdown rather than a manual entry."
    ),
    "Error resolving district geometry": (
        "We couldn't locate the boundary for this region. This is usually temporary — "
        "please try running the analysis again in a moment."
    ),
    "Could not extract cropland areas": (
        "No cropland could be measured for this region and time window. Try a wider "
        "date range or a neighboring district."
    ),
    "Flood detection engine failed": (
        "The satellite flood-detection layer had trouble loading for this exact window. "
        "Results shown may rely on a fallback data source — check the confidence badge."
    ),
    "CHIRPS rainfall baseline extraction failed": (
        "Historical rainfall comparison data was temporarily unavailable; rainfall Z-score "
        "may show as N/A for this run."
    ),
}


def friendly_error(raw_message: str) -> str:
    if not raw_message:
        return "An unknown error occurred while extracting satellite data."
    for pattern, friendly in FRIENDLY_ERROR_MAP.items():
        if pattern in raw_message:
            return friendly
    return raw_message

# ----------------------------------------------------------------------------
# AI Advisory (LLM-generated recommendations) settings — powered by Groq
# ----------------------------------------------------------------------------
# Requires GROQ_API_KEY in .env. If it's missing, ai_insights.py degrades
# gracefully - the AI Insights tab shows a "not configured" message instead
# of crashing the rest of the dashboard, which works fully without it.
#
# Get a free key at: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llama-3.3-70b-versatile is Groq's current flagship free-tier model as of
# writing; check https://console.groq.com/docs/models for the current list
# if this one is ever retired.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ----------------------------------------------------------------------------
# Data source registry - single source of truth shown in BOTH the UI and the
# PDF report, so the two can never describe the methodology differently.
# ----------------------------------------------------------------------------
DATA_SOURCES = [
    {
        "name": "District Boundaries",
        "collection_id": "FAO/GAUL/2015/level2",
        "provider": "FAO (Food and Agriculture Organization of the UN)",
        "purpose": "Administrative district geometry used to clip every other layer.",
    },
    {
        "name": "Cropland Mask",
        "collection_id": "ESA/WorldCover/v100",
        "provider": "European Space Agency",
        "purpose": "10m global land-cover map (2020 snapshot); cropland class used as a static proxy for all analysis years.",
    },
    {
        "name": "Flood Detection (primary, 2014+)",
        "collection_id": "COPERNICUS/S1_GRD",
        "provider": "Copernicus / ESA Sentinel-1",
        "purpose": "10m SAR imagery; low VV backscatter used to detect standing floodwater, all-weather/cloud-penetrating.",
    },
    {
        "name": "Flood Detection (fallback, pre-2014)",
        "collection_id": "MODIS/061/MOD09A1",
        "provider": "NASA / USGS MODIS",
        "purpose": "500m optical surface reflectance; NDWI used to detect water where Sentinel-1 has no coverage.",
    },
    {
        "name": "Vegetation Health (NDVI)",
        "collection_id": "MODIS/061/MOD13Q1",
        "provider": "NASA / USGS MODIS",
        "purpose": "250m 16-day NDVI composite; before/after window difference used for crop health drop.",
    },
    {
        "name": "Rainfall",
        "collection_id": "UCSB-CHG/CHIRPS/DAILY",
        "provider": "Climate Hazards Group, UC Santa Barbara",
        "purpose": "5km daily precipitation; summed over the analysis window and compared to a historical baseline.",
    },
    {
        "name": "Elevation",
        "collection_id": "USGS/SRTMGL1_003",
        "provider": "NASA / USGS SRTM",
        "purpose": "30m digital elevation model; average elevation as a flood-risk context indicator.",
    },
]