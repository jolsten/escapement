from __future__ import annotations

import numpy as np
import pytest

from escapement import Clock, epoch

# ── Epoch constants ──────────────────────────────────────────────────


class TestEpochConstants:
    def test_ccsds_epoch(self):
        assert np.datetime64("1958-01-01T00:00:00", "ns") == epoch.CCSDS

    def test_unix_epoch(self):
        assert np.datetime64("1970-01-01T00:00:00", "ns") == epoch.UNIX

    def test_gps_epoch(self):
        assert np.datetime64("1980-01-06T00:00:00", "ns") == epoch.GPS

    def test_glonass_epoch(self):
        assert np.datetime64("1996-01-01T00:00:00", "ns") == epoch.GLONASS

    def test_galileo_epoch(self):
        assert np.datetime64("1999-08-22T00:00:00", "ns") == epoch.GALILEO

    def test_j2000_epoch(self):
        assert np.datetime64("2000-01-01T12:00:00", "ns") == epoch.J2000

    def test_beidou_epoch(self):
        assert np.datetime64("2006-01-01T00:00:00", "ns") == epoch.BEIDOU

    def test_epoch_ordering(self):
        assert epoch.CCSDS < epoch.UNIX < epoch.GPS < epoch.GLONASS
        assert epoch.GLONASS < epoch.GALILEO < epoch.J2000 < epoch.BEIDOU


# ── Epoch registry ──────────────────────────────────────────────────


class TestEpochRegistry:
    def test_builtins_registered(self):
        for name in ("CCSDS", "UNIX", "GPS", "GLONASS", "GALILEO", "J2000", "BEIDOU"):
            assert epoch.get(name) == getattr(epoch, name)

    def test_register_custom(self):
        epoch.register("MISSION_X", np.datetime64("2027-03-15"))
        assert epoch.get("MISSION_X") == np.datetime64("2027-03-15T00:00:00", "ns")

    def test_resolve_string(self):
        assert epoch.resolve("UNIX") == epoch.UNIX

    def test_resolve_datetime64(self):
        dt = np.datetime64("2025-01-01T00:00:00", "ns")
        assert epoch.resolve(dt) == dt

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown epoch"):
            epoch.resolve("NONEXISTENT")


# ── Epoch registry edge cases ──────────────────────────────────────


class TestEpochRegistryEdgeCases:
    def test_get_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="unknown epoch"):
            epoch.get("NONEXISTENT")

    def test_overwrite_builtin_raises(self):
        with pytest.raises(ValueError, match="cannot overwrite built-in"):
            epoch.register("UNIX", np.datetime64("2000-01-01"))

    def test_overwrite_custom_allowed(self):
        epoch.register("TEMP_EPOCH", np.datetime64("2025-01-01"))
        epoch.register("TEMP_EPOCH", np.datetime64("2026-01-01"))
        assert epoch.get("TEMP_EPOCH") == np.datetime64("2026-01-01T00:00:00", "ns")

    def test_builtins_unchanged_after_failed_overwrite(self):
        original = epoch.get("GPS")
        with pytest.raises(ValueError):
            epoch.register("GPS", np.datetime64("2000-01-01"))
        assert epoch.get("GPS") == original


# ── Resolution validation ────────────────────────────────────────────


class TestResolutionValidation:
    def test_unix_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution must be one of"):
            Clock.unix(resolution="minutes")

    def test_met_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution must be one of"):
            Clock.met(epoch=epoch.UNIX, resolution="bogus")

    def test_unix_valid_resolutions(self):
        for res in ("s", "ms", "us", "ns"):
            c = Clock.unix(resolution=res)
            assert len(c.fields) == 1
