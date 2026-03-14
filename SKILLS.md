# escapement — AI skill file

Use this file to teach LLM coding assistants how to work with `escapement`.

## Installation

```
pip install escapement
```

Requires Python 3.9+ and NumPy 1.20+.

## Import map

```python
from escapement import Clock, ClockField, epoch
```

These are the only public exports.

## ClockField

Defines one segment of a time code. Frozen dataclass.

```python
ClockField(ticks: int, seconds: int, width: int)
```

- `ticks` / `seconds` = rational tick rate ("ticks per seconds seconds")
- `width` = byte count for this field (1-8)

| Meaning | ticks | seconds | Example width |
|---------|-------|---------|---------------|
| seconds | 1 | 1 | 4 |
| milliseconds | 1000 | 1 | 4 |
| microseconds | 1000000 | 1 | 2 |
| nanoseconds | 1000000000 | 1 | 8 |
| days | 1 | 86400 | 2 |
| hours | 1 | 3600 | 1 |
| minutes | 1 | 60 | 1 |
| GPS weeks | 1 | 604800 | 2 |
| CUC fine (2 bytes) | 65536 | 1 | 2 |
| CUC fine (3 bytes) | 16777216 | 1 | 3 |

**Validation:**
- `ticks < 1` → `ValueError`
- `seconds < 1` → `ValueError`
- `width` not in [1, 8] → `ValueError`

## Clock

Composable time code: epoch + ordered fields (coarsest → finest). Frozen dataclass.

```python
Clock(epoch: str | np.datetime64, fields: tuple[ClockField, ...])
```

- `epoch`: a registered epoch name (e.g. `"UNIX"`) or a `datetime64` value. Always stored as `datetime64[ns]`.
- `fields`: non-empty tuple of `ClockField`, ordered coarsest to finest.

**Validation:**
- Empty fields → `ValueError`
- Unknown epoch string → `ValueError` listing registered epochs

**Properties:**

| Property | Type | Meaning |
|----------|------|---------|
| `total_bytes` | int | sum of all field widths |
| `total_bits` | int | total_bytes * 8 |
| `resolution_ns` | int | nanoseconds per finest-field tick: `(finest.seconds * 10**9) // finest.ticks` |

## Encoding

```python
encoded = clock.encode(time)
```

| Input | Output shape | Output dtype |
|-------|-------------|--------------|
| scalar `datetime64` | `(total_bytes,)` | `uint8` |
| array of N `datetime64` | `(N, total_bytes)` | `uint8` |

Fields are written **big-endian**. Each field consumes what it can from the time delta and passes the remainder to the next field.

**Errors:**
- Time before epoch → `ValueError("time before epoch")`
- Field value exceeds byte width → `OverflowError("field value exceeds {width}-byte max {max_val}")`

Both errors apply to entire arrays — if any element fails, the whole call raises.

## Decoding

```python
decoded = clock.decode(data)
```

| Input | Output |
|-------|--------|
| `bytes` or `bytearray` | scalar `datetime64` |
| 1-D `ndarray[uint8]` (length = total_bytes) | scalar `datetime64` |
| 2-D `ndarray[uint8]` shape (N, total_bytes) | 1-D array of N `datetime64` |

**Errors:**
- Wrong byte count → `ValueError("expected {total_bytes} bytes per row, got {actual}")`

## Round-trip behavior

- Encoding **truncates** (floor), never rounds
- `decode(encode(t))` is always `<= t`
- Error is strictly less than one finest-field tick
- `decode(encode(t)) == t` when `t` is exactly representable at the clock's resolution
- Encoding is **monotonic**: `t1 <= t2` implies `encode(t1) <= encode(t2)` lexicographically
- Reverse round-trip: `encode(decode(raw)).tobytes() == raw` for any valid byte sequence

## Factory methods

All parameters are keyword-only.

### Clock.unix()

```python
Clock.unix(*, resolution: str = "s", num_bytes: int = 4) -> Clock
```

- Epoch: `UNIX` (1970-01-01)
- Single field. `resolution` is one of `"s"`, `"ms"`, `"us"`, `"ns"`.
- Invalid resolution → `ValueError`

### Clock.met()

```python
Clock.met(*, epoch: str | np.datetime64, resolution: str = "s", num_bytes: int = 4) -> Clock
```

- Epoch: user-supplied (**required**, no default)
- Single field, same resolution options as `unix()`

### Clock.cuc()

```python
Clock.cuc(*, coarse_bytes: int = 4, fine_bytes: int = 0, epoch: str | np.datetime64 | None = None) -> Clock
```

- Epoch: `CCSDS` (1958-01-01) when `epoch=None`
- Coarse field: `ClockField(1, 1, coarse_bytes)` — 1 tick/s
- Fine field (only if `fine_bytes > 0`): `ClockField(2^(8*fine_bytes), 1, fine_bytes)`
  - fine_bytes=1 → 256 ticks/s
  - fine_bytes=2 → 65536 ticks/s (~15.3 us resolution)
  - fine_bytes=3 → 16777216 ticks/s (~59.6 ns resolution)

### Clock.cds()

```python
Clock.cds(*, epoch: str | np.datetime64 | None = None, day_bytes: int = 2, sub_ms: bool = False) -> Clock
```

- Epoch: `CCSDS` (1958-01-01) when `epoch=None`
- Fields:
  1. `ClockField(1, 86400, day_bytes)` — days
  2. `ClockField(1000, 1, 4)` — milliseconds of day
  3. Only if `sub_ms=True`: `ClockField(1_000_000, 1, 2)` — sub-millisecond remainder (0-999 us, via field cascade)

### Clock.gps()

```python
Clock.gps(*, week_bytes: int = 2, num_bytes: int = 4) -> Clock
```

- Epoch: `GPS` (1980-01-06)
- Fields:
  1. `ClockField(1, 604800, week_bytes)` — week number
  2. `ClockField(1, 1, num_bytes)` — seconds of week

## CUC P-field

```python
Clock.cuc_pfield(*, coarse_bytes: int = 4, fine_bytes: int = 0, agency_epoch: bool = False) -> bytes
```

Returns a 1-byte P-field per CCSDS 301.0-B-4.

- `coarse_bytes`: 1-4 (else `ValueError`)
- `fine_bytes`: 0-3 (else `ValueError`)
- `agency_epoch=True` sets bit 5

**Bit layout** (MSB first): `0 EEE CC FF`
- Bit 7: extension flag (always 0)
- Bits 6-4: time code ID — `0b001` (CCSDS epoch) or `0b011` (agency epoch)
- Bits 3-2: `coarse_bytes - 1`
- Bits 1-0: `fine_bytes`

Formula: `pfield = (0b001 << 4) | (agency << 5) | ((coarse_bytes - 1) << 2) | fine_bytes`

Examples:
- `cuc_pfield(coarse_bytes=4, fine_bytes=0)` → `0b0_001_11_00` = `0x1C`
- `cuc_pfield(coarse_bytes=4, fine_bytes=2)` → `0b0_001_11_10` = `0x1E`
- `cuc_pfield(coarse_bytes=4, fine_bytes=0, agency_epoch=True)` → `0b0_011_11_00` = `0x3C`

## Epoch module

### Built-in epochs

| Name | Value |
|------|-------|
| `epoch.CCSDS` | 1958-01-01T00:00:00 |
| `epoch.UNIX` | 1970-01-01T00:00:00 |
| `epoch.GPS` | 1980-01-06T00:00:00 |
| `epoch.GLONASS` | 1996-01-01T00:00:00 |
| `epoch.GALILEO` | 1999-08-22T00:00:00 |
| `epoch.J2000` | 2000-01-01T12:00:00 |
| `epoch.BEIDOU` | 2006-01-01T00:00:00 |

All stored as `datetime64[ns]`. All are registered for string-based lookup (e.g. `Clock.cuc(epoch="GPS")`).

### epoch.register()

```python
epoch.register(name: str, value: np.datetime64) -> None
```

Register a custom epoch. Can overwrite other custom epochs, but **cannot** overwrite built-ins (`ValueError`).

### epoch.get()

```python
epoch.get(name: str) -> np.datetime64
```

Look up by name. Unknown name → `ValueError`.

### epoch.resolve()

```python
epoch.resolve(epoch: str | np.datetime64) -> np.datetime64
```

String → registry lookup. `datetime64` → coerce to ns. Used internally by `Clock.__init__`.

## Custom clock example

```python
import numpy as np
from escapement import Clock, ClockField

# Days / Hours / Minutes / Seconds — 5 bytes total
dhms = Clock(
    epoch="UNIX",
    fields=(
        ClockField(1, 86400, 2),  # days
        ClockField(1, 3600,  1),  # hours
        ClockField(1,   60,  1),  # minutes
        ClockField(1,    1,  1),  # seconds
    ),
)

t = dhms.epoch + np.timedelta64(3, "D") + np.timedelta64(14, "h") \
                + np.timedelta64(30, "m") + np.timedelta64(45, "s")
raw = dhms.encode(t)           # shape (5,), uint8: [0, 3, 14, 30, 45]
assert dhms.decode(raw) == t   # exact round-trip
```

## repr format

`Clock` has a custom `__repr__` using positional-style field display:

```
Clock(epoch='1970-01-01', fields=(ClockField(1, 1, 4),))
Clock(epoch='2000-01-01T12:00:00', fields=(ClockField(1, 1, 4),))
```

- Epoch shown as date-only (`YYYY-MM-DD`) when it falls exactly at midnight, otherwise with full time
- `ClockField` uses positional format: `ClockField(ticks, seconds, width)` (not keyword args)

## Common pitfalls

- Encoding times before the epoch raises `ValueError`, not a silent negative encoding
- Overflow is per-field: a 1-byte field overflows at 256, a 2-byte at 65536
- `decode()` with wrong byte count raises immediately — check `clock.total_bytes`
- CDS has ms resolution by default; sub-tick nanoseconds are truncated, not rounded
- `epoch.register()` cannot overwrite built-in names like `"UNIX"` or `"GPS"`
- All datetime64 inputs are coerced to nanosecond precision internally
