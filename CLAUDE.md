# Plan: Add `Clock` — composable N-segment time code (package: escapement)

## Context

A single `Clock` class with N segments handles every common time code pattern — linear counters (Unix, MET), fixed-point codes (CUC), and mixed-radix segmented codes (CDS, D/H/M/S). Each clock is an **epoch** plus an **ordered tuple of fields**, where each field has a tick rate and a byte width. Pure codec: `datetime64 → uint8 array → datetime64`, vectorized over arrays of times.

## Design

### `ClockField` (frozen dataclass)

```python
@dataclass(frozen=True)
class ClockField:
    """One segment of a time code.

    The tick rate is expressed as a rational number: ticks / seconds
    ticks per second.  Read as "N ticks per M seconds":

        seconds:  ticks=1,     seconds=1       → 1 tick/s
        ms:       ticks=1000,  seconds=1       → 1000 ticks/s
        days:     ticks=1,     seconds=86400   → 1 tick/86400s
        hours:    ticks=1,     seconds=3600    → 1 tick/3600s
        CUC fine: ticks=65536, seconds=1       → 65536 ticks/s
    """
    ticks: int      # rate numerator
    seconds: int    # rate denominator
    width: int      # output width in bytes
```

### `Clock` (frozen dataclass)

```python
@dataclass(frozen=True)
class Clock:
    """Composable time code: epoch + ordered fields (coarsest → finest).

    encode: datetime64 | ndarray → ndarray[uint8]
        Scalar input  → shape (total_bytes,)
        Array input   → shape (N, total_bytes)

    decode: bytes | ndarray[uint8] → datetime64 | ndarray
        bytes / 1-D array → scalar datetime64
        2-D array         → 1-D array of datetime64
    """
    epoch: np.datetime64
    fields: tuple[ClockField, ...]
```

**CUC P-field:**
- `cuc_pfield(coarse_bytes=4, fine_bytes=2)` matches CCSDS 301.0-B-4 spec
- `agency_epoch=True` flips epoch bits

**Factories:**
- Each factory returns correct epoch, field count, tick rate values, and byte widths

## Execution order

1. Create escapement package structure (`pyproject.toml`, `CLAUDE.md`, `src/escapement/`)
2. Implement `ClockField` and `Clock` in `src/escapement/clock.py`
3. Write tests in `tests/test_clock.py`
4. Run tests: `uv run python -m pytest --tb=short -q`
5. All tests green
