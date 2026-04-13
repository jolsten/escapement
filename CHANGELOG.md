# Changelog

## v0.2.0

### Added

- `Clock.decode_ticks(*positional, **named)`: decode pre-extracted per-field
  tick counts directly to `datetime64[ns]`, without going through the byte
  layout. Accepts Python ints, numpy integer scalars, or 1-D integer arrays;
  array inputs broadcast to a common shape. Validates arity, integer dtype,
  non-negative values, and per-field `ticks * rate` int64 overflow. Does not
  validate against `ClockField.width` (width is a byte-layout concern).
- `ClockField.name` (optional): when every field in a `Clock` has a unique
  name, `decode_ticks` accepts kwargs (e.g. `decode_ticks(day=1, second=2)`).
  Purely ergonomic; does not affect `encode`/`decode`.
- `ClockField.ns_per_tick`: exact integer nanoseconds per tick. Raises
  `ValueError` for sub-ns rates rather than silently rounding.

### Changed

- `Clock.decode` refactored to share its tick-accumulation path with
  `decode_ticks` via new internal helpers `_extract_ticks_from_bytes` and
  `_accumulate_ns`. No behavior change.
