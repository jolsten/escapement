from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from escapement import epoch as _epoch

_NS = 1_000_000_000

_RESOLUTION_TPS: dict[str, tuple[int, int]] = {
    "s": (1, 1),
    "ms": (1_000, 1),
    "us": (1_000_000, 1),
    "ns": (1_000_000_000, 1),
}


def _resolve_tps(resolution: str) -> tuple[int, int]:
    try:
        return _RESOLUTION_TPS[resolution]
    except KeyError:
        raise ValueError(
            f"resolution must be one of {sorted(_RESOLUTION_TPS)}, got {resolution!r}"
        ) from None


@dataclass(frozen=True)
class ClockField:
    """One segment of a time code.

    The tick rate is expressed as a rational number: ticks / seconds
    ticks per second.  Read as "N ticks per M seconds".
    """

    ticks: int
    seconds: int
    width: int
    _tick_num: int = field(init=False, repr=False, compare=False)
    _tick_den: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.ticks < 1:
            raise ValueError("ticks must be >= 1")
        if self.seconds < 1:
            raise ValueError("seconds must be >= 1")
        if not 1 <= self.width <= 8:
            raise ValueError("width must be between 1 and 8")
        g = math.gcd(self.ticks, _NS * self.seconds)
        object.__setattr__(self, "_tick_num", self.ticks // g)
        object.__setattr__(self, "_tick_den", (_NS * self.seconds) // g)


@dataclass(frozen=True, init=False)
class Clock:
    """Composable time code: epoch + ordered fields (coarsest -> finest).

    encode: datetime -> bytes (concatenated big-endian fields)
    decode: bytes -> datetime
    """

    epoch: np.datetime64
    fields: tuple[ClockField, ...]

    def __init__(self, epoch: str | np.datetime64, fields: tuple[ClockField, ...]) -> None:
        if len(fields) == 0:
            raise ValueError("fields must not be empty")
        object.__setattr__(self, "epoch", _epoch.resolve(epoch))
        object.__setattr__(self, "fields", fields)

    def __repr__(self) -> str:
        # Check midnight via integer comparison rather than string matching.
        midnight = np.datetime64(self.epoch, "D") == self.epoch
        epoch_str = str(np.datetime64(self.epoch, "D")) if midnight else str(self.epoch)
        fields_str = ", ".join(
            f"ClockField({f.ticks}, {f.seconds}, {f.width})" for f in self.fields
        )
        return f"Clock(epoch='{epoch_str}', fields=({fields_str},))"

    @property
    def total_bytes(self) -> int:
        return sum(f.width for f in self.fields)

    @property
    def total_bits(self) -> int:
        return self.total_bytes * 8

    @property
    def resolution_ns(self) -> int:
        """Duration of one finest-field tick, in nanoseconds."""
        finest = self.fields[-1]
        return finest._tick_den // finest._tick_num

    def encode(self, time: np.datetime64 | np.ndarray) -> np.ndarray:
        scalar = np.ndim(time) == 0
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        n = times.shape[0]
        delta_ns = (times - self.epoch).astype("timedelta64[ns]").astype(np.int64)
        if np.any(delta_ns < 0):
            raise ValueError("time before epoch")
        result = np.zeros((n, self.total_bytes), dtype=np.uint8)
        remainder_ns = delta_ns.copy()
        col = 0
        for f in self.fields:
            ticks = remainder_ns * f._tick_num // f._tick_den
            consumed_ns = ticks * f._tick_den // f._tick_num
            remainder_ns -= consumed_ns
            max_val = (1 << (f.width * 8)) - 1
            if np.any(ticks > max_val):
                raise OverflowError(
                    f"field value exceeds {f.width}-byte max {max_val}"
                )
            for i in range(f.width):
                shift = (f.width - 1 - i) * 8
                result[:, col + i] = (ticks >> shift) & 0xFF
            col += f.width
        if scalar:
            return result[0]
        return result

    def decode(self, data: bytes | np.ndarray) -> np.datetime64 | np.ndarray:
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(1, -1)
            scalar = True
        else:
            arr = np.asarray(data, dtype=np.uint8)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
                scalar = True
            else:
                scalar = False
        if arr.shape[1] != self.total_bytes:
            raise ValueError(
                f"expected {self.total_bytes} bytes per row, got {arr.shape[1]}"
            )
        n = arr.shape[0]
        total_ns = np.zeros(n, dtype=np.int64)
        col = 0
        for f in self.fields:
            ticks = np.zeros(n, dtype=np.int64)
            for i in range(f.width):
                shift = (f.width - 1 - i) * 8
                ticks |= arr[:, col + i].astype(np.int64) << shift
            total_ns += ticks * f._tick_den // f._tick_num
            col += f.width
        result = self.epoch + total_ns.astype("timedelta64[ns]")
        if scalar:
            return result[0]
        return result

    # -- Factory class methods --

    @classmethod
    def unix(cls, *, resolution: str = "s", num_bytes: int = 4) -> Clock:
        t, s = _resolve_tps(resolution)
        return cls(
            epoch=_epoch.UNIX,
            fields=(ClockField(t, s, num_bytes),),
        )

    @classmethod
    def met(
        cls,
        *,
        epoch: str | np.datetime64,
        resolution: str = "s",
        num_bytes: int = 4,
    ) -> Clock:
        t, s = _resolve_tps(resolution)
        return cls(
            epoch=epoch,
            fields=(ClockField(t, s, num_bytes),),
        )

    @classmethod
    def cuc(
        cls,
        *,
        coarse_bytes: int = 4,
        fine_bytes: int = 0,
        epoch: str | np.datetime64 | None = None,
    ) -> Clock:
        if epoch is None:
            epoch = _epoch.CCSDS
        fields: list[ClockField] = [ClockField(1, 1, coarse_bytes)]
        if fine_bytes > 0:
            fields.append(ClockField(1 << (8 * fine_bytes), 1, fine_bytes))
        return cls(epoch=epoch, fields=tuple(fields))

    @classmethod
    def cds(
        cls,
        *,
        epoch: str | np.datetime64 | None = None,
        day_bytes: int = 2,
        sub_ms: bool = False,
    ) -> Clock:
        if epoch is None:
            epoch = _epoch.CCSDS
        fields: list[ClockField] = [
            ClockField(1, 86400, day_bytes),  # days
            ClockField(1000, 1, 4),  # ms of day
        ]
        if sub_ms:
            fields.append(ClockField(1_000_000, 1, 2))  # microseconds
        return cls(epoch=epoch, fields=tuple(fields))

    @classmethod
    def gps(cls, *, week_bytes: int = 2, num_bytes: int = 4) -> Clock:
        return cls(
            epoch=_epoch.GPS,
            fields=(
                ClockField(1, 604800, week_bytes),  # week number
                ClockField(1, 1, num_bytes),  # seconds of week
            ),
        )

    @staticmethod
    def cuc_pfield(
        *,
        coarse_bytes: int = 4,
        fine_bytes: int = 0,
        agency_epoch: bool = False,
    ) -> bytes:
        """Return 1-byte CUC P-field per CCSDS 301.0-B-4."""
        if not 1 <= coarse_bytes <= 4:
            raise ValueError("coarse_bytes must be between 1 and 4")
        if not 0 <= fine_bytes <= 3:
            raise ValueError("fine_bytes must be between 0 and 3")
        # Bit 7: P-field extension (0 = no extension)
        # Bit 6-4: time code ID = 0b001 for CUC level 1
        # Bit 3-2: coarse octets - 1 (2 bits)
        # Bit 1-0: fine octets (2 bits)
        pfield = 0b0_001_00_00
        if agency_epoch:
            pfield |= 0b0_010_00_00  # set epoch bit (bit 5)
        pfield |= ((coarse_bytes - 1) & 0x03) << 2
        pfield |= fine_bytes & 0x03
        return bytes([pfield])
