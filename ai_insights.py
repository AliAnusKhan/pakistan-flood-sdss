"""
ai_insights.py
--------------
Generates plain-language flood-recovery recommendations and precautions for
future events, using the Groq API (Llama 3.3 70B - Fast & Free tier).

Design principle: the AI Insights feature is an ADD-ON, not a dependency.
If GROQ_API_KEY isn't set, or the `groq` package isn't installed,
or the API call fails for any reason, this module returns a clear status
message instead of raising - the rest of the dashboard (which doesn't
depend on this) keeps working normally.

Prepared by Ali Anus (Updated with Pakistan Cropping Calendar Rules)
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("ai_insights")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

try:
    from groq import Groq
    _GROQ_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    Groq = None
    _GROQ_IMPORT_ERROR = str(exc)


SYSTEM_PROMPT = (
    "You are an agricultural disaster-recovery advisor writing for district-level "
    "government officials and farmers in Pakistan. Given satellite-derived flood and "
    "crop-damage statistics for one district and time window, write a concise, "
    "practical advisory. Do not invent numbers beyond what is given.\n\n"
    "CRITICAL PAKISTAN CROPPING CALENDAR RULES:\n"
    "1. Kharif Season (May to October): Active standing crops are Rice, Cotton, Sugarcane, and Maize. "
    "WHEAT IS NEVER A STANDING CROP IN JULY/AUGUST/SEPTEMBER (it is harvested by April). Do not mention Wheat for summer/monsoon floods.\n"
    "2. Rabi Season (November to April): Active standing crops are Wheat, Gram, Mustard, and Barley.\n"
    "3. Only refer to the crops explicitly listed in the provided 'Top affected commodities' data or valid for the analysis window.\n\n"
    "Structure your response in exactly these four short sections, using plain language (no jargon):\n"
    "1. Situation Summary (2-3 sentences)\n"
    "2. Immediate Precautions (3-5 bullet points, actionable this season)\n"
    "3. Recovery Recommendations (3-5 bullet points, for the next 3-6 months)\n"
    "4. Future Prevention (2-4 bullet points, structural/long-term measures)\n\n"
    "Keep the entire response under 300 words. Do not add a preamble or disclaimer - "
    "start directly with '1. Situation Summary'."
)


def _build_user_prompt(data: dict) -> str:
    """Turns the computed 10-parameter result into a compact fact sheet for the model."""
    commodities = data.get("Commodity_Breakdown") or []
    commodities_text = (
        ", ".join(
            f"{c['Commodity']} (${c['Financial Loss ($ USD)']:,.0f})"
            for c in commodities[:4]
        )
        if commodities
        else "Not specifically itemized (general cropland)"
    )

    return (
        f"District: {data.get('District')}\n"
        f"Analysis window: {data.get('Start_Date')} to {data.get('End_Date')}\n"
        f"Flood severity (auto-classified): {data.get('Flood_Severity')}\n"
        f"Total cropland: {data.get('P1_Total_Cropland_SqKm')} sq km\n"
        f"Flooded cropland: {data.get('P2_Flooded_Cropland_SqKm')} sq km\n"
        f"Crop damage share: {data.get('P3_Crop_Damage_Percent')}%\n"
        f"NDVI health drop: {data.get('P4_NDVI_Health_Drop')}\n"
        f"Rainfall (CHIRPS): {data.get('P5_Rainfall_CHIRPS_mm')} mm\n"
        f"Soil moisture saturation: {data.get('P6_Soil_Moisture_Sat_Percent')}%\n"
        f"Average elevation: {data.get('P7_Elevation_Avg_Meters')} m\n"
        f"Estimated yield loss: {data.get('P8_Crop_Yield_Loss_Tons')} metric tons\n"
        f"Estimated financial loss: ${data.get('P9_Financial_Loss_USD')}\n"
        f"Vulnerability score: {data.get('P10_Vulnerability_Score')} / 100\n"
        f"Top affected commodities: {commodities_text}"
    )


def is_configured() -> bool:
    """True only if both the package and an API key are available."""
    return Groq is not None and bool(GROQ_API_KEY)


def generate_ai_recommendations(data: dict) -> tuple:
    """
    Returns (recommendation_text | None, status_message).
    Safe to cache with Streamlit.
    """
    if Groq is None:
        return None, (
            "The 'groq' package is not installed. Run: pip install groq"
        )
    if not GROQ_API_KEY:
        return None, (
            "GROQ_API_KEY is not set in your .env file. Get a free key "
            "at https://console.groq.com - AI Insights is optional, the rest of "
            "the dashboard works fully without it."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(data)},
            ],
            model=GROQ_MODEL,
            max_tokens=700,
            temperature=0.3,  # Lower temperature for more accurate & factual output
        )
        
        recommendation = (chat_completion.choices[0].message.content or "").strip()
        if not recommendation:
            return None, "The AI model returned an empty response. Try again."
            
        return recommendation, "Success"
        
    except Exception as exc:
        logger.error("AI recommendation generation failed: %s", exc)
        return None, f"AI Insights request failed: {exc}"