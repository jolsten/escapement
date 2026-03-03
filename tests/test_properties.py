from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from escapement import Clock

# ── Hypothesis strategies ─────────────────────────────────────────────

# Max offset ~194 days in ns — safe for all factory configs
_MAX_OFFSET_NS = 2**24 * 10**9


@st.composite
def factory_clocks(draw):
    kind = draw(
        st.sampled_from(
            [
                "unix_s",
                "unix_ms",
                "cuc_4_0",
                "cuc_4_2",
                "cuc_4_3",
                "cds",
                "cds_sub_ms",
                "gps",
            ]
        )
    )
    if kind == "unix_s":
        return Clock.unix()
    elif kind == "unix_ms":
        return Clock.unix(resolution="ms", num_bytes=8)
    elif kind == "cuc_4_0":
        return Clock.cuc(coarse_bytes=4)
    elif kind == "cuc_4_2":
        return Clock.cuc(coarse_bytes=4, fine_bytes=2)
    elif kind == "cuc_4_3":
        return Clock.cuc(coarse_bytes=4, fine_bytes=3)
    elif kind == "cds":
        return Clock.cds()
    elif kind == "cds_sub_ms":
        return Clock.cds(sub_ms=True)
    else:
        return Clock.gps()


@st.composite
def clock_and_time(draw):
    clock = draw(factory_clocks())
    offset_ns = draw(st.integers(min_value=0, max_value=_MAX_OFFSET_NS))
    time = clock.epoch + np.timedelta64(offset_ns, "ns")
    return clock, time


@st.composite
def clock_and_times(draw):
    clock = draw(factory_clocks())
    n = draw(st.integers(min_value=2, max_value=50))
    offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=_MAX_OFFSET_NS),
            min_size=n,
            max_size=n,
        )
    )
    times = np.array([clock.epoch + np.timedelta64(o, "ns") for o in offsets])
    return clock, times


# ── Property-based tests ──────────────────────────────────────────────


class TestHypothesis:
    @given(data=clock_and_time())
    def test_roundtrip_truncation_bound(self, data):
        """decode(encode(t)) <= t, with error <= ceil(one finest tick)."""
        clock, time = data
        decoded = clock.decode(clock.encode(time))
        diff_ns = int((time - decoded) / np.timedelta64(1, "ns"))
        assert diff_ns >= 0, "decoded must not exceed original"
        finest = clock.fields[-1]
        max_err = (finest._tick_den + finest._tick_num - 1) // finest._tick_num
        assert diff_ns <= max_err

    @given(data=clock_and_time())
    def test_encode_shape_and_dtype(self, data):
        """Scalar encode always returns (total_bytes,) uint8."""
        clock, time = data
        encoded = clock.encode(time)
        assert encoded.shape == (clock.total_bytes,)
        assert encoded.dtype == np.uint8

    @given(clock=factory_clocks(), data=st.data())
    def test_decode_total(self, clock, data):
        """Decode never raises for any valid-length byte pattern."""
        raw = data.draw(st.binary(min_size=clock.total_bytes, max_size=clock.total_bytes))
        clock.decode(raw)  # should not raise

    @given(data=clock_and_times())
    def test_vectorized_encode_matches_scalar(self, data):
        """Batch encode produces identical rows to scalar encode."""
        clock, times = data
        batch = clock.encode(times)
        for i, t in enumerate(times):
            np.testing.assert_array_equal(batch[i], clock.encode(t))

    @given(data=clock_and_times())
    def test_vectorized_decode_matches_scalar(self, data):
        """Batch decode produces identical elements to scalar decode."""
        clock, times = data
        encoded = clock.encode(times)
        batch_decoded = clock.decode(encoded)
        for i in range(len(times)):
            assert batch_decoded[i] == clock.decode(encoded[i])

    @given(clock=factory_clocks())
    def test_epoch_encodes_to_zeros(self, clock):
        """Encoding the epoch always produces all-zero bytes."""
        assert np.all(clock.encode(clock.epoch) == 0)

    @given(data=st.data())
    def test_encode_monotonic(self, data):
        """Later times never encode to lexicographically earlier bytes."""
        clock = data.draw(factory_clocks())
        a = data.draw(st.integers(min_value=0, max_value=_MAX_OFFSET_NS))
        b = data.draw(st.integers(min_value=0, max_value=_MAX_OFFSET_NS))
        ta = clock.epoch + np.timedelta64(a, "ns")
        tb = clock.epoch + np.timedelta64(b, "ns")
        ea = clock.encode(ta).tobytes()
        eb = clock.encode(tb).tobytes()
        if a <= b:
            assert ea <= eb
        else:
            assert ea >= eb
