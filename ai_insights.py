"""
ai_insights.py
--------------
LLM advisory layer for the Pakistan Agricultural Flood SDSS — powered by
Groq (fast inference on open models like Llama 3.3), using Groq's
OpenAI-compatible REST API directly via `requests` (no extra SDK needed).

Takes the 10 computed satellite parameters (NOT raw imagery — just numbers)
for a district/time-window and asks the model to produce a plain-language
situation summary, immediate precautions, recovery steps, and longer-term
prevention measures.

Public interface expected by app.py:
    is_configured() -> bool
    generate_ai_recommendations(data: dict) -> tuple[str | None, str]
        Returns (ai_text, status). ai_text is None on failure; status holds
        either "Success" or a human-readable error message.

Requires GROQ_API_KEY in your .env file. Get a free key at:
https://console.groq.com/keys
"""

from __future__ import annotations

import logging

import requests

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("ai_insights")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def is_configured() -> bool:
    """True only if an API key is present in .env."""
    return bool(GROQ_API_KEY)


def _build_prompt(data: dict) -> str:
    commodities = data.get("Commodity_Breakdown", [])
    commodity_lines = "\n".join(
        f"- {c['Commodity']}: {c['Yield Loss (Metric Tons)']:,} tons lost, "
        f"${c['Financial Loss ($ USD)']:,.0f} financial loss"
        for c in commodities
    ) or "- No commodity data available"

    stats = data.get("Statistical_Metrics", {})

    return f"""You are an agricultural disaster-recovery advisor for Pakistan.
Based ONLY on the satellite-derived data below for {data.get('District')}
({data.get('Start_Date')} to {data.get('End_Date')}), write a concise,
plain-language advisory for local administrators and farmers. Do not invent
numbers not given below.

FLOOD SEVERITY: {data.get('Flood_Severity')}
Total Cropland: {data.get('P1_Total_Cropland_SqKm')} sq km
Flooded Cropland: {data.get('P2_Flooded_Cropland_SqKm')} sq km
Crop Damage Share: {data.get('P3_Crop_Damage_Percent')}%
NDVI Vegetation Health Drop: {data.get('P4_NDVI_Health_Drop')}
Monsoon Rainfall (CHIRPS): {data.get('P5_Rainfall_CHIRPS_mm')} mm
Soil Moisture Saturation: {data.get('P6_Soil_Moisture_Sat_Percent')}%
Average Elevation: {data.get('P7_Elevation_Avg_Meters')} m
Aggregate Yield Loss: {data.get('P8_Crop_Yield_Loss_Tons')} metric tons
Estimated Financial Loss: ${data.get('P9_Financial_Loss_USD'):,.0f}
Vulnerability Score: {data.get('P10_Vulnerability_Score')}/100
Flood Detection Method: {stats.get('Flood Detection Method')}
Rainfall Z-Score vs 20-yr norm: {stats.get('Monsoon Rain Z-Score')}

Commodity breakdown:
{commodity_lines}

Structure your response with these exact section headers:
1. Situation Summary
2. Immediate Precautions
3. Recovery Steps
4. Longer-Term Prevention

Keep it under 350 words total. Be specific to the data given, not generic."""


def generate_ai_recommendations(data: dict) -> tuple:
    """
    Calls Groq's chat completions endpoint with the computed parameters and
    returns (ai_text, status). On any failure, ai_text is None and status
    contains a friendly reason — app.py displays this via st.error() so the
    rest of the dashboard keeps working even if this call fails.
    """
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY is not set in your .env file."

    prompt = _build_prompt(data)

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 600,
            },
            timeout=30,
        )
    except requests.exceptions.Timeout:
        return None, "Groq API request timed out. Check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        return None, "Could not reach Groq's API. Check your internet connection and try again."
    except Exception as exc:
        logger.error("Groq request failed: %s", exc)
        return None, f"AI recommendation generation failed: {exc}"

    if response.status_code == 401:
        return None, (
            "Your GROQ_API_KEY appears to be invalid or expired. Get a new "
            "one at https://console.groq.com/keys"
        )
    if response.status_code == 404:
        return None, (
            f"Model '{GROQ_MODEL}' was not found or has been retired. Check "
            "https://console.groq.com/docs/models for a current model name "
            "and update GROQ_MODEL in your .env file."
        )
    if response.status_code == 429:
        return None, "Groq rate limit reached. Wait a moment and try again."
    if response.status_code != 200:
        try:
            err_detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            err_detail = response.text
        return None, f"Groq API error ({response.status_code}): {err_detail}"

    try:
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Unexpected Groq response shape: %s", exc)
        return None, "Groq returned an unexpected response format. Try again."

    if not text or not text.strip():
        return None, "Groq returned an empty response. Try again."

    return text.strip(), "Success"