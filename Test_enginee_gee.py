"""
tests/test_gee_engine.py
-------------------------
Unit tests for the pure, non-GEE business logic in gee_engine.py:
severity classification and district crop-profile lookup.

These do NOT call Earth Engine and do NOT need GEE_PROJECT_ID or network
access — they only exercise plain-Python functions, so they're safe to run
in CI on every commit.

Run with:  pytest tests/test_gee_engine.py -v
"""

import os
import sys

# Ensure the project root is importable when running pytest from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# GEE_PROJECT_ID is required at import time by gee_engine.py; set a dummy
# value before import so these pure-logic tests don't need real credentials.
os.environ.setdefault("GEE_PROJECT_ID", "test-project-placeholder")

from gee_engine import (  # noqa: E402
    classify_flood_severity,
    get_district_crop_profile,
    DISTRICT_CROP_PROFILES,
)


class TestClassifyFloodSeverity:
    def test_none_is_unknown(self):
        assert classify_flood_severity(None) == "Unknown"

    def test_zero_is_low(self):
        assert classify_flood_severity(0.0) == "Low"

    def test_boundary_low_moderate(self):
        assert classify_flood_severity(10.0) == "Low"
        assert classify_flood_severity(10.01) == "Moderate"

    def test_boundary_moderate_high(self):
        assert classify_flood_severity(25.0) == "Moderate"
        assert classify_flood_severity(25.01) == "High"

    def test_boundary_high_extreme(self):
        assert classify_flood_severity(50.0) == "High"
        assert classify_flood_severity(50.01) == "Extreme"

    def test_very_high_value_is_extreme(self):
        assert classify_flood_severity(95.5) == "Extreme"


class TestGetDistrictCropProfile:
    def test_sindh_rice_belt_district(self):
        profile = get_district_crop_profile("Larkana")
        assert profile == DISTRICT_CROP_PROFILES["SINDH_RICE_BELT"]

    def test_case_insensitive_match(self):
        # District names should match regardless of capitalization.
        assert get_district_crop_profile("LARKANA") == DISTRICT_CROP_PROFILES["SINDH_RICE_BELT"]
        assert get_district_crop_profile("dadu") == DISTRICT_CROP_PROFILES["SINDH_RICE_BELT"]

    def test_cotton_belt_district(self):
        profile = get_district_crop_profile("Muzaffargarh")
        assert profile == DISTRICT_CROP_PROFILES["PUNJAB_SINDH_COTTON_BELT"]

    def test_kpk_horticulture_district(self):
        profile = get_district_crop_profile("Nowshera")
        assert profile == DISTRICT_CROP_PROFILES["KPK_MIXED_HORTICULTURE"]

    def test_unknown_district_falls_back_to_national_default(self):
        profile = get_district_crop_profile("Some Unmapped District")
        assert profile == DISTRICT_CROP_PROFILES["DEFAULT_NATIONAL"]

    def test_all_profiles_have_shares_summing_close_to_one(self):
        # Sanity check: each district's commodity share weights should
        # sum to (approximately) 100% of agricultural output.
        for profile_name, commodities in DISTRICT_CROP_PROFILES.items():
            total_share = sum(cfg["share"] for cfg in commodities.values())
            assert abs(total_share - 1.0) < 0.01, (
                f"{profile_name} shares sum to {total_share}, expected ~1.0"
            )