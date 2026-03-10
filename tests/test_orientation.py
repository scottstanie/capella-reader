"""Tests for scene orientation helpers."""

from __future__ import annotations

from copy import deepcopy

import pytest

from capella_reader.collect import Collect
from capella_reader.enums import LookSide
from capella_reader.orientation import (
    derive_scene_orientation,
    render_orientation_ascii,
)


def _make_equatorial_collect(sample_metadata_dict, *, look_side: str) -> Collect:
    data = deepcopy(sample_metadata_dict)
    data["collect"]["image"]["center_pixel"]["target_position"] = [6378137.0, 0.0, 0.0]
    data["collect"]["radar"]["pointing"] = look_side
    data["collect"]["state"]["state_vectors"] = [
        {
            "time": "2024-07-09T04:03:29Z",
            "position": [6878137.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 7500.0],
        }
    ]
    return Collect.model_validate(data["collect"])


def test_derive_scene_orientation_north_left(sample_metadata_dict):
    """Heading north on the equator should look west for left-looking data."""
    collect = _make_equatorial_collect(sample_metadata_dict, look_side="left")

    orientation = derive_scene_orientation(collect)

    assert orientation.heading_deg == pytest.approx(0.0)
    assert orientation.look_deg == pytest.approx(270.0)
    assert orientation.heading_cardinal == "N"
    assert orientation.look_cardinal == "W"
    assert orientation.look_side is LookSide.LEFT
    assert orientation.orbit_direction == "ascending"


def test_render_orientation_ascii_north_right(sample_metadata_dict):
    """ASCII compass should show flight north and look to the right/east."""
    collect = _make_equatorial_collect(sample_metadata_dict, look_side="right")

    rendered = render_orientation_ascii(derive_scene_orientation(collect))

    assert (
        rendered
        == "A=flight  L=look\n\n"
        "    N\n"
        "    ^\n"
        "    A\n"
        "    A\n"
        "W---+LL>E\n"
        "    |\n"
        "    |\n"
        "    |\n"
        "    S\n\n"
        "heading:    0.0 deg (N, ascending)\n"
        "look:      90.0 deg (E, right)"
    )
