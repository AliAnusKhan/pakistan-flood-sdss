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
# Severity -> (badge css, solid hex accent used by charts/tiles/map markers)
SEVERITY_STYLE = {
    "Low":      {"badge": "background:#dcfce7; color:#15803d; border:1px solid #bbf7d0;", "hex": "#16a34a"},
    "Moderate": {"badge": "background:#fef9c3; color:#a16207; border:1px solid #fef08a;", "hex": "#ca8a04"},
    "High":     {"badge": "background:#ffedd5; color:#c2410c; border:1px solid #fed7aa;", "hex": "#ea580c"},
    "Extreme":  {"badge": "background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5;", "hex": "#dc2626"},
    "Unknown":  {"badge": "background:#f1f5f9; color:#475569; border:1px solid #e2e8f0;", "hex": "#64748b"},
}

ACCENT_ENV = "#0ea5e9"   # sky blue  - environmental / physical parameters (flood side)
ACCENT_ECO = "#16a34a"   # green     - economic / agricultural parameters (agriculture side)
ACCENT_NEUTRAL = "#64748b"

# ----------------------------------------------------------------------------
# Friendly translations for raw backend error/status strings
# ----------------------------------------------------------------------------
# Maps a substring found in a raw error/status message to a plain-language,
# non-technical explanation shown to the end user instead of the raw
# Python/GEE exception text.
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
    """Translate a raw backend error/status string into a plain-language
    message for end users, falling back to the original text if no known
    pattern matches."""
    if not raw_message:
        return "An unknown error occurred while extracting satellite data."
    for pattern, friendly in FRIENDLY_ERROR_MAP.items():
        if pattern in raw_message:
            return friendly
    return raw_message

# ----------------------------------------------------------------------------
# AI Advisory (LLM-generated recommendations) settings
# ----------------------------------------------------------------------------
# Requires GEMINI_API_KEY in .env. If it's missing, ai_insights.py degrades
# gracefully - the AI Insights tab shows a "not configured" message instead
# of crashing the rest of the dashboard, which works fully without it.
#
# Get a free key (no credit card needed) at: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# gemini-2.0-flash is on Google's free tier as of writing; check
# https://ai.google.dev/gemini-api/docs/models for the current free-tier model
# if this one is no longer available when you deploy.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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