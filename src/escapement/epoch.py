from __future__ import annotations

import numpy as np

_registry: dict[str, np.datetime64] = {}
_builtins: set[str] = set()


def register(name: str, value: np.datetime64, *, _builtin: bool = False) -> None:
    """Register a named epoch for string-based lookup.

    Raises ValueError if *name* would overwrite a built-in epoch.
    """
    if name in _builtins and not _builtin:
        raise ValueError(f"cannot overwrite built-in epoch {name!r}")
    _registry[name] = np.datetime64(value, "ns")
    if _builtin:
        _builtins.add(name)


def get(name: str) -> np.datetime64:
    """Look up a registered epoch by name."""
    try:
        return _registry[name]
    except KeyError:
        raise ValueError(
            f"unknown epoch {name!r}, registered epochs: {sorted(_registry)}"
        ) from None


def resolve(epoch: str | np.datetime64) -> np.datetime64:
    """Resolve an epoch: pass through datetime64, look up strings in the registry."""
    if isinstance(epoch, str):
        try:
            return _registry[epoch]
        except KeyError:
            raise ValueError(
                f"unknown epoch {epoch!r}, registered epochs: {sorted(_registry)}"
            ) from None
    return np.datetime64(epoch, "ns")


# ── Built-in epochs ──────────────────────────────────────────────────

CCSDS = np.datetime64("1958-01-01T00:00:00", "ns")
UNIX = np.datetime64("1970-01-01T00:00:00", "ns")
GPS = np.datetime64("1980-01-06T00:00:00", "ns")
GLONASS = np.datetime64("1996-01-01T00:00:00", "ns")
GALILEO = np.datetime64("1999-08-22T00:00:00", "ns")
J2000 = np.datetime64("2000-01-01T12:00:00", "ns")
BEIDOU = np.datetime64("2006-01-01T00:00:00", "ns")

# Auto-register built-ins
for _name in ("CCSDS", "UNIX", "GPS", "GLONASS", "GALILEO", "J2000", "BEIDOU"):
    register(_name, globals()[_name], _builtin=True)
del _name
