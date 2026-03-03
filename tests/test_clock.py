from __future__ import annotations

import struct

import numpy as np
import pytest

from escapement import Clock, ClockField, epoch

# ── Construction & validation ──────────────────────────────────────────


class TestClockFieldValidation:
    def test_happy_path(self):
        f = ClockField(1, 1, 4)
        assert f.ticks == 1
        assert f.seconds == 1
        assert f.width == 4

    def test_ticks_lt_1(self):
        with pytest.raises(ValueError, match="ticks"):
            ClockField(0, 1, 4)

    def test_seconds_lt_1(self):
        with pytest.raises(ValueError, match="seconds"):
            ClockField(1, 0, 4)

    def test_width_lt_1(self):
        with pytest.raises(ValueError, match="width"):
            ClockField(1, 1, 0)

    def test_width_gt_8(self):
        with pytest.raises(ValueError, match="width"):
            ClockField(1, 1, 9)


class TestClockValidation:
    def test_empty_fields(self):
        with pytest.raises(ValueError, match="fields"):
            Clock(epoch=np.datetime64("2025-01-01"), fields=())

    def test_epoch_coerced_to_ns(self):
        c = Clock(
            epoch=np.datetime64("2025-01-01"),
            fields=(ClockField(1, 1, 4),),
        )
        assert c.epoch == np.datetime64("2025-01-01T00:00:00.000000000", "ns")

    @pytest.mark.parametrize(
        "res,expected",
        [
            ("D", "2025-06-15T00:00:00"),
            ("h", "2025-06-15T08:00:00"),
            ("m", "2025-06-15T08:30:00"),
            ("s", "2025-06-15T08:30:00"),
            ("ms", "2025-06-15T08:30:00"),
            ("us", "2025-06-15T08:30:00"),
            ("ns", "2025-06-15T08:30:00"),
        ],
    )
    def test_epoch_coerced_from_explicit_resolution(self, res, expected):
        dt = np.datetime64("2025-06-15T08:30:00", res)
        c = Clock(epoch=dt, fields=(ClockField(1, 1, 4),))
        assert c.epoch.dtype == np.dtype("datetime64[ns]")
        assert c.epoch == np.datetime64(expected, "ns")

    def test_epoch_roundtrip_with_coarse_resolution(self):
        """Epoch given as day-resolution still encodes/decodes correctly."""
        dt = np.datetime64("2025-01-01", "D")
        c = Clock(epoch=dt, fields=(ClockField(1, 1, 4),))
        t = c.epoch + np.timedelta64(42, "s")
        assert c.decode(c.encode(t)) == t


# ── Properties ─────────────────────────────────────────────────────────


class TestProperties:
    def test_total_bytes(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        assert c.total_bytes == 6

    def test_total_bits(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        assert c.total_bits == 48


# ── Round-trip encode/decode ───────────────────────────────────────────


class TestUnix:
    def test_unix_seconds(self):
        c = Clock.unix()
        t = np.datetime64("2025-01-01T00:00:00", "ns")
        encoded = c.encode(t)
        unix_ts = int((t - c.epoch) / np.timedelta64(1, "s"))
        assert encoded.tobytes() == struct.pack(">I", unix_ts)
        assert c.decode(encoded) == t

    def test_unix_ms(self):
        c = Clock.unix(resolution="ms", num_bytes=8)
        t = np.datetime64("2025-06-15T12:30:00.123", "ns")
        encoded = c.encode(t)
        decoded = c.decode(encoded)
        assert decoded == t


class TestMET:
    def test_met_100s(self):
        epoch_val = np.datetime64("2025-01-01T00:00:00", "ns")
        c = Clock.met(epoch=epoch_val)
        t = epoch_val + np.timedelta64(100, "s")
        encoded = c.encode(t)
        decoded = c.decode(encoded)
        assert decoded == t
        assert encoded.tobytes() == struct.pack(">I", 100)


class TestCUC:
    def test_cuc_4_0_one_second(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=0)
        t = c.epoch + np.timedelta64(1, "s")
        assert c.encode(t).tobytes() == b"\x00\x00\x00\x01"

    def test_cuc_4_2_half_second(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        t = c.epoch + np.timedelta64(1_500_000_000, "ns")  # 1.5 seconds
        encoded = c.encode(t)
        coarse = int.from_bytes(bytes(encoded[:4]), "big")
        fine = int.from_bytes(bytes(encoded[4:6]), "big")
        assert coarse == 1
        assert fine == 0x8000
        assert c.decode(encoded) == t

    def test_cuc_4_3_sub_microsecond(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=3)
        # 1 second + 1 microsecond = 1_000_001_000 ns
        t = c.epoch + np.timedelta64(1_000_001_000, "ns")
        encoded = c.encode(t)
        decoded = c.decode(encoded)
        # With 3 fine bytes (2^24 = 16777216 ticks/s), resolution is ~59.6 ns.
        # Round-trip should be within 1 fine tick.
        diff_ns = abs(int((decoded - t) / np.timedelta64(1, "ns")))
        assert diff_ns < 60  # less than one fine tick


class TestCDS:
    def test_cds_round_trip(self):
        c = Clock.cds()
        t = c.epoch + np.timedelta64(3, "D") + np.timedelta64(12345, "ms")
        encoded = c.encode(t)
        decoded = c.decode(encoded)
        # CDS has ms resolution, so truncate input to ms
        t_ms = c.epoch + np.timedelta64(3 * 86400 * 1000 + 12345, "ms")
        assert decoded == t_ms

    def test_cds_sub_ms(self):
        c = Clock.cds(sub_ms=True)
        # 2 days + 5000 ms + 500 µs
        t = (
            c.epoch
            + np.timedelta64(2, "D")
            + np.timedelta64(5000, "ms")
            + np.timedelta64(500, "us")
        )
        encoded = c.encode(t)
        decoded = c.decode(encoded)
        assert decoded == t


class TestGPS:
    def test_gps_round_trip(self):
        c = Clock.gps()
        # 2 weeks + 100000 seconds
        t = c.epoch + np.timedelta64(2 * 604800, "s") + np.timedelta64(100000, "s")
        encoded = c.encode(t)
        decoded = c.decode(encoded)
        assert decoded == t


class TestDHMS:
    def test_dhms_round_trip(self):
        c = Clock(
            epoch=np.datetime64("2025-01-01"),
            fields=(
                ClockField(1, 86400, 2),  # days
                ClockField(1, 3600, 1),  # hours
                ClockField(1, 60, 1),  # minutes
                ClockField(1, 1, 1),  # seconds
            ),
        )
        t = (
            c.epoch
            + np.timedelta64(3, "D")
            + np.timedelta64(14, "h")
            + np.timedelta64(30, "m")
            + np.timedelta64(45, "s")
        )
        encoded = c.encode(t)
        # Parse fields
        days = int.from_bytes(bytes(encoded[0:2]), "big")
        hours = int(encoded[2])
        minutes = int(encoded[3])
        seconds = int(encoded[4])
        assert (days, hours, minutes, seconds) == (3, 14, 30, 45)
        assert c.decode(encoded) == t


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_time_at_epoch(self):
        c = Clock.unix()
        encoded = c.encode(c.epoch)
        assert encoded.tobytes() == b"\x00\x00\x00\x00"
        assert c.decode(encoded) == c.epoch

    def test_time_before_epoch(self):
        c = Clock.unix()
        t = c.epoch - np.timedelta64(1, "s")
        with pytest.raises(ValueError, match="before epoch"):
            c.encode(t)

    def test_overflow(self):
        c = Clock(
            epoch=np.datetime64("2025-01-01"),
            fields=(ClockField(1, 1, 1),),  # 1 byte, max 255
        )
        t = c.epoch + np.timedelta64(256, "s")
        with pytest.raises(OverflowError):
            c.encode(t)

    def test_decode_all_zero(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        assert c.decode(b"\x00" * 6) == c.epoch

    def test_decode_max_bytes(self):
        c = Clock.unix(num_bytes=4)
        data = b"\xff\xff\xff\xff"
        result = c.decode(data)
        assert result > c.epoch


# ── CUC P-field ────────────────────────────────────────────────────────


class TestCUCPfield:
    def test_default(self):
        # coarse=4, fine=0, agency_epoch=False
        # 0b0_001_11_00 = 0x1C
        pf = Clock.cuc_pfield(coarse_bytes=4, fine_bytes=0)
        assert pf == bytes([0b0_001_11_00])

    def test_4_2(self):
        # coarse=4, fine=2
        # 0b0_001_11_10 = 0x1E
        pf = Clock.cuc_pfield(coarse_bytes=4, fine_bytes=2)
        assert pf == bytes([0b0_001_11_10])

    def test_agency_epoch(self):
        # coarse=4, fine=0, agency_epoch=True
        # 0b0_011_11_00 = 0x3C
        pf = Clock.cuc_pfield(coarse_bytes=4, fine_bytes=0, agency_epoch=True)
        assert pf == bytes([0b0_011_11_00])


# ── CUC P-field validation ──────────────────────────────────────────


class TestCUCPfieldValidation:
    def test_coarse_bytes_zero(self):
        with pytest.raises(ValueError, match="coarse_bytes"):
            Clock.cuc_pfield(coarse_bytes=0)

    def test_coarse_bytes_five(self):
        with pytest.raises(ValueError, match="coarse_bytes"):
            Clock.cuc_pfield(coarse_bytes=5)

    def test_fine_bytes_negative(self):
        with pytest.raises(ValueError, match="fine_bytes"):
            Clock.cuc_pfield(fine_bytes=-1)

    def test_fine_bytes_four(self):
        with pytest.raises(ValueError, match="fine_bytes"):
            Clock.cuc_pfield(fine_bytes=4)

    def test_all_valid_combos(self):
        """Every valid (coarse, fine) pair produces a 1-byte result."""
        for coarse in range(1, 5):
            for fine in range(0, 4):
                pf = Clock.cuc_pfield(coarse_bytes=coarse, fine_bytes=fine)
                assert len(pf) == 1


# ── Factory construction ──────────────────────────────────────────────


class TestFactories:
    def test_unix_factory(self):
        c = Clock.unix()
        assert c.epoch == np.datetime64("1970-01-01T00:00:00", "ns")
        assert len(c.fields) == 1
        assert c.fields[0].ticks == 1
        assert c.fields[0].seconds == 1
        assert c.fields[0].width == 4

    def test_met_factory(self):
        epoch_val = np.datetime64("2025-06-01")
        c = Clock.met(epoch=epoch_val, resolution="ms", num_bytes=8)
        assert c.epoch == np.datetime64("2025-06-01T00:00:00", "ns")
        assert c.fields[0].ticks == 1000
        assert c.fields[0].width == 8

    def test_cuc_factory(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        assert c.epoch == np.datetime64("1958-01-01T00:00:00", "ns")
        assert len(c.fields) == 2
        assert c.fields[0] == ClockField(1, 1, 4)
        assert c.fields[1] == ClockField(65536, 1, 2)

    def test_cds_factory(self):
        c = Clock.cds()
        assert c.epoch == np.datetime64("1958-01-01T00:00:00", "ns")
        assert len(c.fields) == 2
        assert c.fields[0] == ClockField(1, 86400, 2)
        assert c.fields[1] == ClockField(1000, 1, 4)

    def test_cds_sub_ms_factory(self):
        c = Clock.cds(sub_ms=True)
        assert len(c.fields) == 3
        assert c.fields[2] == ClockField(1_000_000, 1, 2)

    def test_gps_factory(self):
        c = Clock.gps()
        assert c.epoch == np.datetime64("1980-01-06T00:00:00", "ns")
        assert len(c.fields) == 2
        assert c.fields[0] == ClockField(1, 604800, 2)
        assert c.fields[1] == ClockField(1, 1, 4)


# ── Factory non-default parameters ──────────────────────────────────


class TestFactoryNonDefaults:
    def test_gps_4byte_weeks(self):
        c = Clock.gps(week_bytes=4)
        assert c.fields[0].width == 4
        assert c.fields[0].seconds == 604800
        t = c.epoch + np.timedelta64(10 * 604800, "s") + np.timedelta64(12345, "s")
        assert c.decode(c.encode(t)) == t

    def test_gps_custom_seconds_bytes(self):
        c = Clock.gps(num_bytes=8)
        assert c.fields[1].width == 8
        t = c.epoch + np.timedelta64(604800 + 100000, "s")
        assert c.decode(c.encode(t)) == t

    def test_cds_4byte_days(self):
        c = Clock.cds(day_bytes=4)
        assert c.fields[0].width == 4
        # 100000 days — far beyond 2-byte range
        t = c.epoch + np.timedelta64(100000, "D") + np.timedelta64(5000, "ms")
        assert c.decode(c.encode(t)) == t

    def test_unix_ns_resolution(self):
        c = Clock.unix(resolution="ns", num_bytes=8)
        t = np.datetime64("2025-01-01T00:00:00.123456789", "ns")
        assert c.decode(c.encode(t)) == t

    def test_unix_us_resolution(self):
        c = Clock.unix(resolution="us", num_bytes=8)
        t = np.datetime64("2025-01-01T00:00:00.123456", "ns")
        assert c.decode(c.encode(t)) == t


# ── Vectorized ────────────────────────────────────────────────────────


class TestVectorized:
    def test_encode_scalar_shape(self):
        c = Clock.unix()
        t = np.datetime64("2025-01-01T00:00:00", "ns")
        result = c.encode(t)
        assert result.shape == (c.total_bytes,)
        assert result.dtype == np.uint8

    def test_encode_array_shape(self):
        c = Clock.unix()
        times = np.array(
            [
                "2025-01-01T00:00:00",
                "2025-01-02T00:00:00",
                "2025-01-03T00:00:00",
                "2025-01-04T00:00:00",
                "2025-01-05T00:00:00",
            ],
            dtype="datetime64[ns]",
        )
        result = c.encode(times)
        assert result.shape == (5, c.total_bytes)
        assert result.dtype == np.uint8

    def test_encode_array_matches_scalar(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        times = np.array([c.epoch + np.timedelta64(i * 1_500_000_000, "ns") for i in range(10)])
        batch = c.encode(times)
        for i, t in enumerate(times):
            np.testing.assert_array_equal(batch[i], c.encode(t))

    def test_decode_bytes_returns_scalar(self):
        c = Clock.unix()
        result = c.decode(b"\x00\x00\x00\x01")
        assert np.ndim(result) == 0

    def test_decode_1d_returns_scalar(self):
        c = Clock.unix()
        arr = np.array([0, 0, 0, 1], dtype=np.uint8)
        result = c.decode(arr)
        assert np.ndim(result) == 0

    def test_decode_2d_returns_array(self):
        c = Clock.unix()
        arr = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 2],
            ],
            dtype=np.uint8,
        )
        result = c.decode(arr)
        assert result.shape == (2,)

    def test_roundtrip_vectorized_unix(self):
        c = Clock.unix()
        base = np.datetime64("2025-01-01T00:00:00", "ns")
        times = np.array([base + np.timedelta64(i, "s") for i in range(100)])
        encoded = c.encode(times)
        decoded = c.decode(encoded)
        np.testing.assert_array_equal(decoded, times)

    def test_roundtrip_vectorized_cuc(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        times = np.array([c.epoch + np.timedelta64(i * 500_000_000, "ns") for i in range(20)])
        encoded = c.encode(times)
        decoded = c.decode(encoded)
        np.testing.assert_array_equal(decoded, times)

    def test_encode_array_before_epoch(self):
        c = Clock.unix()
        times = np.array(
            [
                c.epoch + np.timedelta64(1, "s"),
                c.epoch - np.timedelta64(1, "s"),
            ]
        )
        with pytest.raises(ValueError, match="before epoch"):
            c.encode(times)

    def test_encode_array_overflow(self):
        c = Clock(
            epoch=np.datetime64("2025-01-01"),
            fields=(ClockField(1, 1, 1),),
        )
        times = np.array(
            [
                c.epoch + np.timedelta64(100, "s"),
                c.epoch + np.timedelta64(256, "s"),
            ]
        )
        with pytest.raises(OverflowError):
            c.encode(times)

    def test_gcd_reduced_fractions(self):
        # seconds: ticks=1, seconds=1
        # gcd(1, 1e9) = 1 → _tick_num=1, _tick_den=1e9
        f = ClockField(1, 1, 4)
        assert f._tick_num == 1
        assert f._tick_den == 1_000_000_000

        # CUC fine 2 bytes: ticks=65536, seconds=1
        # gcd(65536, 1e9) = gcd(2^16, 2^9*5^9) = 512
        # _tick_num = 65536/512 = 128, _tick_den = 1e9/512 = 1953125
        f2 = ClockField(65536, 1, 2)
        assert f2._tick_num == 128
        assert f2._tick_den == 1953125


# ── Decode length validation ────────────────────────────────────────


class TestDecodeLengthValidation:
    def test_bytes_too_short(self):
        c = Clock.unix()  # expects 4 bytes
        with pytest.raises(ValueError, match="expected 4 bytes"):
            c.decode(b"\x00\x00")

    def test_bytes_too_long(self):
        c = Clock.unix()
        with pytest.raises(ValueError, match="expected 4 bytes"):
            c.decode(b"\x00\x00\x00\x00\x00")

    def test_array_too_short(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)  # expects 6 bytes
        arr = np.zeros((3, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected 6 bytes"):
            c.decode(arr)

    def test_array_too_long(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        arr = np.zeros((3, 8), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected 6 bytes"):
            c.decode(arr)

    def test_1d_array_wrong_length(self):
        c = Clock.unix()
        arr = np.zeros(5, dtype=np.uint8)
        with pytest.raises(ValueError, match="expected 4 bytes"):
            c.decode(arr)

    def test_decode_bytearray(self):
        c = Clock.unix()
        data = bytearray(b"\x00\x00\x00\x01")
        result = c.decode(data)
        assert result == c.epoch + np.timedelta64(1, "s")

    def test_decode_empty_bytes(self):
        c = Clock.unix()
        with pytest.raises(ValueError, match="expected 4 bytes"):
            c.decode(b"")


# ── Field capacity boundaries ───────────────────────────────────────


class TestFieldBoundaries:
    def test_encode_at_1byte_max(self):
        c = Clock(epoch=np.datetime64("2025-01-01"), fields=(ClockField(1, 1, 1),))
        t = c.epoch + np.timedelta64(255, "s")
        encoded = c.encode(t)
        assert encoded.tobytes() == b"\xff"
        assert c.decode(encoded) == t

    def test_encode_at_1byte_overflow(self):
        c = Clock(epoch=np.datetime64("2025-01-01"), fields=(ClockField(1, 1, 1),))
        t = c.epoch + np.timedelta64(256, "s")
        with pytest.raises(OverflowError):
            c.encode(t)

    def test_encode_at_2byte_max(self):
        c = Clock(epoch=np.datetime64("2025-01-01"), fields=(ClockField(1, 1, 2),))
        t = c.epoch + np.timedelta64(65535, "s")
        encoded = c.encode(t)
        assert encoded.tobytes() == b"\xff\xff"
        assert c.decode(encoded) == t

    def test_encode_at_2byte_overflow(self):
        c = Clock(epoch=np.datetime64("2025-01-01"), fields=(ClockField(1, 1, 2),))
        t = c.epoch + np.timedelta64(65536, "s")
        with pytest.raises(OverflowError):
            c.encode(t)

    def test_decode_max_cuc_fine(self):
        """Decoding 0xFFFF fine ticks should not error."""
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        data = b"\x00\x00\x00\x00\xff\xff"
        result = c.decode(data)
        assert result > c.epoch


# ── Single-element array input ──────────────────────────────────────


class TestSingleElementArray:
    def test_encode_single_element_array(self):
        c = Clock.unix()
        times = np.array([np.datetime64("2025-01-01T00:00:00", "ns")])
        result = c.encode(times)
        assert result.shape == (1, c.total_bytes)
        assert result.dtype == np.uint8

    def test_decode_single_row_2d(self):
        c = Clock.unix()
        arr = np.array([[0, 0, 0, 1]], dtype=np.uint8)
        result = c.decode(arr)
        assert result.shape == (1,)

    def test_single_element_roundtrip(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        times = np.array([c.epoch + np.timedelta64(42_000_000_000, "ns")])
        decoded = c.decode(c.encode(times))
        np.testing.assert_array_equal(decoded, times)


# ── resolution_ns property ──────────────────────────────────────────


class TestResolutionNs:
    def test_unix_seconds(self):
        c = Clock.unix()
        assert c.resolution_ns == 1_000_000_000

    def test_unix_ms(self):
        c = Clock.unix(resolution="ms", num_bytes=8)
        assert c.resolution_ns == 1_000_000

    def test_unix_us(self):
        c = Clock.unix(resolution="us", num_bytes=8)
        assert c.resolution_ns == 1_000

    def test_unix_ns(self):
        c = Clock.unix(resolution="ns", num_bytes=8)
        assert c.resolution_ns == 1

    def test_cuc_4_2(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        # 65536 ticks/s → 1e9/65536 ≈ 15258.7 → floor = 15258 ns
        assert c.resolution_ns == 15258

    def test_cuc_4_3(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=3)
        # 16777216 ticks/s → 1e9/16777216 ≈ 59.6 → floor = 59 ns
        assert c.resolution_ns == 59

    def test_gps(self):
        c = Clock.gps()
        # Finest field is 1 tick/s
        assert c.resolution_ns == 1_000_000_000

    def test_cds_sub_ms(self):
        c = Clock.cds(sub_ms=True)
        # Finest field is 1e6 ticks/s → 1000 ns
        assert c.resolution_ns == 1_000


# ── Clock repr ───────────────────────────────────────────────────────


class TestClockRepr:
    def test_repr_midnight_epoch(self):
        c = Clock.unix()
        r = repr(c)
        assert "Clock(" in r
        assert "1970-01-01" in r
        assert "ClockField(1, 1, 4)" in r

    def test_repr_non_midnight_epoch(self):
        c = Clock(
            epoch=np.datetime64("2000-01-01T12:00:00"),
            fields=(ClockField(1, 1, 4),),
        )
        r = repr(c)
        assert "12:00:00" in r

    def test_repr_multi_field(self):
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        r = repr(c)
        assert "ClockField(1, 1, 4)" in r
        assert "ClockField(65536, 1, 2)" in r


# ── String epoch in constructors ────────────────────────────────────


class TestStringEpoch:
    def test_clock_constructor_string(self):
        c = Clock(epoch="UNIX", fields=(ClockField(1, 1, 4),))
        assert c.epoch == epoch.UNIX

    def test_cuc_factory_string(self):
        c = Clock.cuc(epoch="CCSDS")
        assert c.epoch == epoch.CCSDS

    def test_cds_factory_string(self):
        c = Clock.cds(epoch="CCSDS")
        assert c.epoch == epoch.CCSDS

    def test_met_factory_string(self):
        epoch.register("MY_SAT", np.datetime64("2025-06-01"))
        c = Clock.met(epoch="MY_SAT", resolution="ms", num_bytes=8)
        assert c.epoch == np.datetime64("2025-06-01T00:00:00", "ns")

    def test_roundtrip_with_string_epoch(self):
        c = Clock.cuc(epoch="CCSDS", coarse_bytes=4, fine_bytes=2)
        t = c.epoch + np.timedelta64(1_500_000_000, "ns")
        assert c.decode(c.encode(t)) == t


# ── Decode-then-encode round-trip ───────────────────────────────────


class TestDecodeEncodeRoundtrip:
    def test_decode_encode_identity(self):
        """Decoding then re-encoding should return the same bytes."""
        c = Clock.cuc(coarse_bytes=4, fine_bytes=2)
        data = b"\x00\x00\x00\x2a\x80\x00"  # 42 seconds + 0.5s
        assert c.encode(c.decode(data)).tobytes() == data

    def test_decode_encode_all_zeros(self):
        c = Clock.unix()
        data = b"\x00\x00\x00\x00"
        assert c.encode(c.decode(data)).tobytes() == data

    def test_decode_encode_vectorized(self):
        c = Clock.gps()
        raw = np.array(
            [
                [0, 1, 0, 0, 0, 100],
                [0, 2, 0, 0, 1, 0],
            ],
            dtype=np.uint8,
        )
        re_encoded = c.encode(c.decode(raw))
        np.testing.assert_array_equal(re_encoded, raw)
