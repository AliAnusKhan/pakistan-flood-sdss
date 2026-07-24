"""
Standalone sanity check - run this BEFORE wiring the key into the full
dashboard, to confirm the key + package actually work in isolation.
Usage:  python test_gemini_key.py
"""
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Key loaded from .env:", "YES" if key else "NO - check your .env file path/name")

from google import genai

client = genai.Client(api_key=key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Reply with exactly: Setup working.",
)
print("Response:", response.text)