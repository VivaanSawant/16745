"""
integration_target_score_aim_SPEC.py
====================================
Resolve **explicit ring + sector** (or named presets) to board-plane aim ``(Δy, Δz)`` in metres
for Monte Carlo / targeting scripts. Scoring matches **B3** via ``score_from_deltas_SPEC``.
"""

from __future__ import annotations

import re
from typing import Literal

from dartrobot.constants import (
    R_DOUBLE_INNER_MM,
    R_DOUBLE_OUTER_MM,
    R_INNER_BULL_MM,
    R_OUTER_BULL_MM,
    SEGMENT_ORDER_CLOCKWISE_FROM_TOP,
)
from dartrobot.flight.scoring import score_from_deltas_SPEC
from dartrobot.integration.target_board_regions import (
    board_deltas_m_at_radius_SPEC,
    inner_single_center_radius_m_SPEC,
    treble_ring_center_radius_m_SPEC,
)

RingName_SPEC = Literal["treble", "double", "single", "bull_inner", "bull_outer"]


def double_ring_center_radius_m_SPEC() -> float:
    """Mid-radius of the **double** scoring band."""
    return 0.5 * (float(R_DOUBLE_INNER_MM) + float(R_DOUBLE_OUTER_MM)) / 1000.0


def board_deltas_for_ring_sector_SPEC(ring: RingName_SPEC, sector: int | None) -> tuple[float, float]:
    """
    Canonical aim point ``(Δy_m, Δz_m)`` at the angular center of ``sector`` on the given ring.

    ``sector`` is ignored for ``bull_*`` (must still be passed as ``None`` or any int).
    """
    if ring in ("bull_inner", "bull_outer"):
        if ring == "bull_inner":
            return 0.0, 0.0
        r_m = 0.5 * (float(R_INNER_BULL_MM) + float(R_OUTER_BULL_MM)) / 1000.0
        return 0.0, r_m
    if sector is None:
        raise ValueError(f"ring '{ring}' requires sector in 1..20")
    s = int(sector)
    if s not in SEGMENT_ORDER_CLOCKWISE_FROM_TOP:
        raise ValueError(f"invalid sector {s}; must be one of {SEGMENT_ORDER_CLOCKWISE_FROM_TOP}")
    if ring == "treble":
        r = treble_ring_center_radius_m_SPEC()
    elif ring == "double":
        r = double_ring_center_radius_m_SPEC()
    elif ring == "single":
        r = inner_single_center_radius_m_SPEC()
    else:
        raise ValueError(f"unknown ring {ring!r}")
    return board_deltas_m_at_radius_SPEC(s, r)


def score_at_aim_SPEC(dy_m: float, dz_m: float) -> int:
    """Dartboard score at landing deltas (B3)."""
    return int(score_from_deltas_SPEC(float(dy_m), float(dz_m)))


def assert_target_score_matches_aim_SPEC(target_score: int, dy_m: float, dz_m: float) -> None:
    """Raise ``ValueError`` if ``score_from_deltas`` at the aim point is not ``target_score``."""
    got = score_at_aim_SPEC(dy_m, dz_m)
    if int(got) != int(target_score):
        raise ValueError(
            f"Aim point (Δy,Δz)=({dy_m:.6f},{dz_m:.6f}) m scores {got}, not target {target_score}. "
            "Check --ring / --sector / --preset."
        )


# Named presets only; treble/double use **t** / **d** + sector (see ``resolve_preset_SPEC``).
_PRESET_TO_RING_SECTOR: dict[str, tuple[RingName_SPEC, int | None]] = {
    "S20": ("single", 20),
    "BULL": ("bull_inner", None),
    "DBULL": ("bull_inner", None),
    "SBULL": ("bull_outer", None),
    "BULL_OUTER": ("bull_outer", None),
}

# Case-insensitive treble/double: ``t10`` = treble 10 (score 30), ``d15`` = double 15 (score 30).
_TD_PRESET_RE = re.compile(r"^([td])(\d{1,2})$", re.IGNORECASE)


def resolve_preset_SPEC(preset: str) -> tuple[RingName_SPEC, int | None]:
    """
    Map shorthand to ``(ring, sector)``.

    - **Treble:** ``t1``…``t20`` or ``T1``…``T20`` (letter = treble).
    - **Double:** ``d1``…``d20`` or ``D1``…``D20`` (letter = double). Same dart *score* as a
      different treble (e.g. ``t10`` vs ``d15`` both 30) but **different ring geometry**.
    - **Other:** ``S20``, ``BULL``, ``DBULL``, ``SBULL``, ``BULL_OUTER`` (case-insensitive).
    """
    raw = preset.strip()
    key_u = raw.upper()
    if key_u in _PRESET_TO_RING_SECTOR:
        return _PRESET_TO_RING_SECTOR[key_u]
    m = _TD_PRESET_RE.match(raw)
    if m:
        letter = m.group(1).lower()
        sec = int(m.group(2))
        if sec < 1 or sec > 20:
            raise ValueError(f"preset sector must be 1..20, got {sec!r} in {preset!r}")
        if letter == "t":
            return ("treble", sec)
        return ("double", sec)
    raise ValueError(
        f"unknown preset {preset!r}. Use t1..t20 (treble), d1..d20 (double), "
        f"or one of: {', '.join(sorted(_PRESET_TO_RING_SECTOR.keys()))}"
    )


def parse_ring_SPEC(s: str) -> RingName_SPEC:
    s = s.strip().lower()
    allowed: tuple[RingName_SPEC, ...] = ("treble", "double", "single", "bull_inner", "bull_outer")
    if s not in allowed:
        raise ValueError(f"ring must be one of {allowed}, got {s!r}")
    return s  # type: ignore[return-value]
