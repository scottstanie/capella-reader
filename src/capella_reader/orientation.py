"""Helpers for deriving and rendering scene orientation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from capella_reader.collect import Collect
from capella_reader.enums import LookSide
from capella_reader.geometry import ECEFPosition
from capella_reader.orbit import StateVector

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - ((WGS84_B * WGS84_B) / (WGS84_A * WGS84_A))
WGS84_EP2 = ((WGS84_A * WGS84_A) - (WGS84_B * WGS84_B)) / (WGS84_B * WGS84_B)

_COMPASS_16 = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

_OCTANT_STEPS = (
    ((0, -1), "^"),
    ((1, -1), "/"),
    ((1, 0), ">"),
    ((1, 1), "\\"),
    ((0, 1), "v"),
    ((-1, 1), "/"),
    ((-1, 0), "<"),
    ((-1, -1), "\\"),
)


@dataclass(frozen=True, slots=True)
class SceneOrientation:
    """Derived north-up scene orientation."""

    heading_deg: float
    look_deg: float
    heading_cardinal: str
    look_cardinal: str
    orbit_direction: str
    look_side: LookSide


def derive_scene_orientation(collect: Collect) -> SceneOrientation:
    """Derive heading and look bearings at the scene center."""
    state_vectors = collect.state.state_vectors
    if not state_vectors:
        msg = "No state vectors available to derive scene orientation"
        raise ValueError(msg)

    center_pixel = collect.image.center_pixel
    lat_rad, lon_rad = _ecef_to_geodetic(center_pixel.target_position)
    east, north = _east_north_basis(lat_rad, lon_rad)

    state_vector = _nearest_state_vector(state_vectors, center_pixel.center_time)
    velocity = state_vector.velocity.as_array()
    east_component = float(np.dot(velocity, east))
    north_component = float(np.dot(velocity, north))

    if math.isclose(east_component, 0.0, abs_tol=1e-12) and math.isclose(
        north_component, 0.0, abs_tol=1e-12
    ):
        msg = "Could not derive heading from the scene-center velocity projection"
        raise ValueError(msg)

    heading_deg = math.degrees(math.atan2(east_component, north_component)) % 360.0
    look_offset = 90.0 if collect.radar.pointing is LookSide.RIGHT else -90.0
    look_deg = (heading_deg + look_offset) % 360.0

    return SceneOrientation(
        heading_deg=heading_deg,
        look_deg=look_deg,
        heading_cardinal=_bearing_to_cardinal(heading_deg),
        look_cardinal=_bearing_to_cardinal(look_deg),
        orbit_direction=collect.state.direction,
        look_side=collect.radar.pointing,
    )


def render_orientation_ascii(orientation: SceneOrientation) -> str:
    """Render a small north-up ASCII compass with flight/look arrows."""
    canvas = [list(" " * 9) for _ in range(9)]
    center = 4

    canvas[0][center] = "N"
    canvas[8][center] = "S"
    canvas[4][0] = "W"
    canvas[4][8] = "E"

    for y in range(1, 8):
        canvas[y][center] = "|"
    for x in range(1, 8):
        canvas[center][x] = "-"
    canvas[center][center] = "+"

    _draw_arrow(canvas, orientation.heading_deg, "A")
    _draw_arrow(canvas, orientation.look_deg, "L")

    art = "\n".join("".join(row).rstrip() for row in canvas)
    return (
        "A=flight  L=look\n\n"
        f"{art}\n\n"
        f"heading: {orientation.heading_deg:6.1f} deg "
        f"({orientation.heading_cardinal}, {orientation.orbit_direction})\n"
        f"look:    {orientation.look_deg:6.1f} deg "
        f"({orientation.look_cardinal}, {orientation.look_side.value})"
    )


def _nearest_state_vector(state_vectors: list[StateVector], center_time) -> StateVector:
    return min(
        state_vectors, key=lambda sv: abs((sv.time - center_time).total_seconds())
    )


def _ecef_to_geodetic(position: ECEFPosition) -> tuple[float, float]:
    x, y, z = position.as_array()
    lon_rad = math.atan2(y, x)
    xy_norm = math.hypot(x, y)
    theta = math.atan2(z * WGS84_A, xy_norm * WGS84_B)

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    lat_rad = math.atan2(
        z + (WGS84_EP2 * WGS84_B * sin_theta**3),
        xy_norm - (WGS84_E2 * WGS84_A * cos_theta**3),
    )
    return lat_rad, lon_rad


def _east_north_basis(lat_rad: float, lon_rad: float) -> tuple[np.ndarray, np.ndarray]:
    east = np.array([-math.sin(lon_rad), math.cos(lon_rad), 0.0], dtype=float)
    north = np.array(
        [
            -math.sin(lat_rad) * math.cos(lon_rad),
            -math.sin(lat_rad) * math.sin(lon_rad),
            math.cos(lat_rad),
        ],
        dtype=float,
    )
    return east, north


def _bearing_to_cardinal(bearing_deg: float) -> str:
    index = int(((bearing_deg + 11.25) % 360.0) // 22.5)
    return _COMPASS_16[index]


def _bearing_to_octant(bearing_deg: float) -> int:
    return int(((bearing_deg + 22.5) % 360.0) // 45.0)


def _draw_arrow(canvas: list[list[str]], bearing_deg: float, shaft_char: str) -> None:
    center = 4
    octant = _bearing_to_octant(bearing_deg)
    (dx, dy), head_char = _OCTANT_STEPS[octant]

    for step in (1, 2):
        _place(canvas, center + (dx * step), center + (dy * step), shaft_char)
    _place(canvas, center + (dx * 3), center + (dy * 3), head_char)


def _place(canvas: list[list[str]], x: int, y: int, char: str) -> None:
    current = canvas[y][x]
    if current in {" ", "|", "-"}:
        canvas[y][x] = char
    elif current != char:
        canvas[y][x] = "*"


def render_geometry_wide(orientation: SceneOrientation, collect: Collect) -> str:
    """Render a wide, labeled ASCII diagram of flight and look directions.

    Arrow endpoints are computed from the actual bearing using trigonometry so
    that the diagram faithfully represents continuous angles (not just 8 or 16
    quantised directions). A 2:1 column-to-row ratio approximates equal visual
    scaling in a monospace font.
    """
    diagram = _build_wide_diagram(orientation.heading_deg, orientation.look_deg)

    center_pixel = collect.image.center_pixel
    lat_rad, lon_rad = _ecef_to_geodetic(center_pixel.target_position)
    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)

    pol = f"{collect.radar.transmit_polarization}{collect.radar.receive_polarization}"
    return (
        f"{diagram}\n\n"
        f"  {collect.platform} | {collect.mode} | {pol}\n"
        f"  {orientation.orbit_direction} | {orientation.look_side.value}-looking\n"
        f"  heading: {orientation.heading_deg:.1f} deg"
        f" ({orientation.heading_cardinal})"
        f"  |  look: {orientation.look_deg:.1f} deg"
        f" ({orientation.look_cardinal})\n"
        f"  incidence: {center_pixel.incidence_angle:.1f} deg"
        f"  |  center: ({lat_deg:.2f}, {lon_deg:.2f})"
    )


_ARROW_RADIUS = 3.0
_ASPECT_RATIO = 2.0


def _build_wide_diagram(heading_deg: float, look_deg: float) -> str:
    """Build a wide ASCII diagram using continuous-angle arrows."""
    rows, cols = 9, 35
    cr, cc = 4, 17

    canvas = [[" "] * cols for _ in range(rows)]

    canvas[0][cc] = "N"
    canvas[rows - 1][cc] = "S"
    canvas[cr][cc] = "+"

    _draw_continuous_arrow(canvas, cr, cc, heading_deg, "flight")
    _draw_continuous_arrow(canvas, cr, cc, look_deg, "look")

    # Fill vertical axis reference where not occupied by arrows
    for r in range(1, rows - 1):
        if r != cr and canvas[r][cc] == " ":
            canvas[r][cc] = "|"

    return "\n".join("".join(row).rstrip() for row in canvas)


def _draw_continuous_arrow(
    canvas: list[list[str]],
    cr: int,
    cc: int,
    bearing_deg: float,
    label: str,
) -> None:
    """Draw an arrow at an arbitrary bearing from the canvas center."""
    rows = len(canvas)
    cols = len(canvas[0])
    theta = math.radians(bearing_deg)

    end_r = round(cr - math.cos(theta) * _ARROW_RADIUS)
    end_c = round(cc + math.sin(theta) * _ARROW_RADIUS * _ASPECT_RATIO)
    end_r = max(1, min(rows - 2, end_r))
    end_c = max(1, min(cols - 2, end_c))

    dr = end_r - cr
    dc = end_c - cc
    steps = max(abs(dr), abs(dc), 1)

    prev_r, prev_c = cr, cc
    for i in range(1, steps + 1):
        r = round(cr + dr * i / steps)
        c = round(cc + dc * i / steps)
        step_dr = r - prev_r
        step_dc = c - prev_c
        ch = _step_char(step_dr, step_dc, is_head=(i == steps))
        if 0 <= r < rows and 0 <= c < cols and canvas[r][c] == " ":
            canvas[r][c] = ch
        prev_r, prev_c = r, c

    # Place label near the arrow tip
    if end_c >= cc:
        start_c = end_c + 2
    else:
        start_c = end_c - len(label) - 1
    for i, ch in enumerate(label):
        c = start_c + i
        if 0 <= end_r < rows and 0 <= c < cols and canvas[end_r][c] == " ":
            canvas[end_r][c] = ch


def _step_char(dr: int, dc: int, *, is_head: bool) -> str:
    if dc == 0 and dr == 0:
        return "+"
    if dc == 0:
        return ("^" if dr < 0 else "v") if is_head else "|"
    if dr == 0:
        return (">" if dc > 0 else "<") if is_head else "-"
    if (dr > 0) == (dc > 0):
        return "\\"
    return "/"
