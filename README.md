# escapement

Composable time code encoder/decoder for Python.

`escapement` converts between `numpy.datetime64` timestamps and the binary
time code formats used in software or protocols. Define a clock as an epoch 
plus an ordered tuple of fields, then encode and decode — scalars or arrays.

```python
from escapement import Clock

cuc = Clock.cuc(coarse_bytes=4, fine_bytes=2)
encoded = cuc.encode(np.datetime64("2025-06-15T12:00:00.5"))
decoded = cuc.decode(encoded)  # round-trips within one finest tick
```

## Why "escapement"?

In a mechanical clock the *escapement* is the mechanism that converts
continuous energy into discrete, evenly spaced ticks. It is the part that
makes a clock *tick* — literally parcelling smooth motion into counted
intervals.

That is exactly what this library does: it takes a continuous time value and
divides it into a cascade of discrete, counted fields — coarse ticks flowing
into finer ticks, each field consuming a portion of the remainder before
passing it on. The analogy runs deeper than the name suggests: a real
escapement couples a fast oscillator (the balance wheel) to a slow gear
train, just as a CUC time code couples a fine sub-second counter to a coarse
whole-second counter.

## Standard-specific factories

| Factory        | Standard              | Fields                         |
|----------------|-----------------------|--------------------------------|
| `Clock.unix()` | Unix / POSIX          | seconds (or ms/us/ns)          |
| `Clock.met()`  | Mission Elapsed Time  | seconds (or ms/us/ns)          |
| `Clock.cuc()`  | CCSDS Unsegmented     | coarse seconds + fine fraction |
| `Clock.cds()`  | CCSDS Day Segmented   | days + ms of day (+ sub-ms)    |
| `Clock.gps()`  | GPS                   | week number + seconds of week  |

Custom segmented clocks (D/H/M/S, or any mixed-radix layout) are built
directly from `Clock` and `ClockField` — see
[Custom clocks](#custom-clocks) below.

## Features

- Pure codec: `datetime64 <-> uint8 array <-> datetime64`
- Vectorized: scalar and array inputs, zero-copy where possible
- Rational tick arithmetic (GCD-reduced) — no floating-point error
- Built-in epoch registry (Unix, J2000, GPS, GLONASS, Galileo, BeiDou, CCSDS)
- CUC P-field generation per CCSDS 301.0-B-4

## Installation

```
pip install escapement
```

Requires Python 3.9+ and NumPy 1.20+.

## Quick start

```python
import numpy as np
from escapement import Clock, ClockField, epoch

# Unix timestamp, millisecond resolution
unix_ms = Clock.unix(resolution="ms", num_bytes=8)
encoded = unix_ms.encode(np.datetime64("2025-01-01T00:00:00.123"))
decoded = unix_ms.decode(encoded)

# CUC 4.2 (CCSDS epoch, 4 coarse + 2 fine bytes)
cuc = Clock.cuc(coarse_bytes=4, fine_bytes=2)
print(cuc.resolution_ns)  # 15258 ns per finest tick

# GPS week + seconds
gps = Clock.gps()
t = gps.epoch + np.timedelta64(2 * 604800 + 100000, "s")
assert gps.decode(gps.encode(t)) == t

# Batch encode 1000 timestamps
times = np.array([cuc.epoch + np.timedelta64(i, "s") for i in range(1000)])
encoded = cuc.encode(times)   # shape (1000, 6)
decoded = cuc.decode(encoded)  # shape (1000,)

# Register a mission-specific epoch
epoch.register("LAUNCH", np.datetime64("2027-03-15T09:30:00"))
met = Clock.met(epoch="LAUNCH", resolution="ms", num_bytes=8)
```

## Custom clocks

Any time code layout can be expressed by composing `ClockField` tuples.
Each `ClockField(ticks, seconds, width)` reads as *"ticks per seconds
seconds, stored in width bytes"*. Fields are ordered coarsest to finest —
during encoding, each field consumes what it can from the remainder and
passes the rest down, exactly like the gear train in a mechanical clock.

```python
from escapement import Clock, ClockField

# Days / hours / minutes / seconds — 5 bytes total
dhms = Clock(
    epoch="UNIX",
    fields=(
        ClockField(1, 86400, 2),  # days    (1 tick per 86_400 s, 2 bytes)
        ClockField(1, 3600,  1),  # hours   (1 tick per  3_600 s,  1 byte)
        ClockField(1,   60,  1),  # minutes (1 tick per     60 s,  1 byte)
        ClockField(1,    1,  1),  # seconds (1 tick per      1 s,  1 byte)
    ),
)

t = dhms.epoch + np.timedelta64(3, "D") + np.timedelta64(14, "h") \
                + np.timedelta64(30, "m") + np.timedelta64(45, "s")
raw = dhms.encode(t)  # 5 bytes: [0, 3, 14, 30, 45]
assert dhms.decode(raw) == t
```

## Leap seconds and timescales

`escapement` is a pure codec — it computes the arithmetic delta between an
epoch and a timestamp, then divides that delta into fields. It has no
notion of UTC, TAI, or leap seconds.

This is by design. Time code standards like CUC, CDS, and GPS define byte
layouts, not timescales. A CUC packet from one spacecraft might count TAI
seconds; from another it might count UTC seconds. The byte-level encoding
is identical in both cases — only the interpretation differs. Mixing
timescale logic into the codec would force one interpretation on everyone.

In practice this means:

- `numpy.datetime64` is leap-second-unaware (it counts SI seconds on a
  proleptic Gregorian calendar), so it behaves like a continuous timescale.
- GPS time is also continuous (no leap seconds since its 1980 epoch), so
  `Clock.gps()` round-trips correctly without any correction.
- If your input timestamps are UTC and you need TAI or GPS time, apply the
  leap second offset *before* encoding (or *after* decoding).

The authoritative source for leap seconds is **IERS Bulletin C**, published
every six months at
[https://www.iers.org/IERS/EN/Publications/Bulletins/Bulletins.html](https://www.iers.org/IERS/EN/Publications/Bulletins/Bulletins.html).

The current TAI-UTC offset and the full history of leap second insertions
are also available in the `leap-seconds.list` file maintained by IETF/NIST
at [https://www.ietf.org/timezones/data/leap-seconds.list](https://www.ietf.org/timezones/data/leap-seconds.list).

## License

`escapement` is licensed under the MIT License - see the LICENSE file for details
