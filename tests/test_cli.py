"""Tests for CLI helpers."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import tifffile

from capella_reader import cli


def _write_test_tiff(path, metadata_dict):
    test_data = np.random.randn(10, 10).astype(np.complex64)
    gcp_values = np.array(
        [
            0.0,
            0.0,
            0.0,
            10.0,
            20.0,
            30.0,
            5.0,
            10.0,
            0.0,
            40.0,
            50.0,
            60.0,
        ],
        dtype=np.float64,
    )
    extratags = [(33922, "d", len(gcp_values), gcp_values, False)]

    with tifffile.TiffWriter(path) as tif:
        tif.write(
            test_data,
            description=json.dumps(metadata_dict),
            extratags=extratags,
        )


def test_cli_dump_suppresses_long_entries(tmp_path, sample_metadata_dict, capsys):
    """Test dump suppresses long entries by default."""
    metadata_path = tmp_path / "test.json"
    metadata_path.write_text(json.dumps(sample_metadata_dict))

    exit_code = cli.main(["dump", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "attitude samples; use --full" in captured.out
    assert "state vectors; use --full" in captured.out
    assert "transmit_antenna" in captured.out
    assert "receive_antenna" in captured.out
    assert "suppressed; use --full" in captured.out


def test_cli_dump_full_includes_entries(tmp_path, sample_metadata_dict, capsys):
    """Test dump includes long entries with --full."""
    metadata_path = tmp_path / "test.json"
    metadata_path.write_text(json.dumps(sample_metadata_dict))

    exit_code = cli.main(["dump", str(metadata_path), "--full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "attitude samples; use --full" not in captured.out
    assert "state vectors; use --full" not in captured.out
    assert "suppressed; use --full" not in captured.out


def test_cli_bounds_text_output(tmp_path, sample_metadata_dict, capsys):
    """Test bounds emits plain text output."""
    tiff_path = tmp_path / "test_gcps.tif"
    _write_test_tiff(tiff_path, sample_metadata_dict)

    exit_code = cli.main(["bounds", str(tiff_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "10.0 20.0 40.0 50.0"


def test_cli_bounds_json_output(tmp_path, sample_metadata_dict, capsys):
    """Test bounds emits JSON output."""
    tiff_path = tmp_path / "test_gcps.tif"
    _write_test_tiff(tiff_path, sample_metadata_dict)

    exit_code = cli.main(["bounds", str(tiff_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "[\n  10.0,\n  20.0,\n  40.0,\n  50.0\n]"


def test_cli_direction_output(tmp_path, sample_metadata_dict, capsys):
    """Test direction emits the ASCII compass for a TIFF input."""
    metadata_dict = deepcopy(sample_metadata_dict)
    metadata_dict["collect"]["image"]["center_pixel"]["target_position"] = [
        6378137.0,
        0.0,
        0.0,
    ]
    metadata_dict["collect"]["radar"]["pointing"] = "right"
    metadata_dict["collect"]["state"]["state_vectors"] = [
        {
            "time": "2024-07-09T04:03:29Z",
            "position": [6878137.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 7500.0],
        }
    ]

    tiff_path = tmp_path / "test_direction.tif"
    _write_test_tiff(tiff_path, metadata_dict)

    exit_code = cli.main(["direction", str(tiff_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        captured.out
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
        "look:      90.0 deg (E, right)\n"
    )


def test_cli_geometry_output(tmp_path, sample_metadata_dict, capsys):
    """Test geometry emits the wide labeled diagram."""
    metadata_dict = deepcopy(sample_metadata_dict)
    metadata_dict["collect"]["image"]["center_pixel"]["target_position"] = [
        6378137.0,
        0.0,
        0.0,
    ]
    metadata_dict["collect"]["radar"]["pointing"] = "right"
    metadata_dict["collect"]["state"]["state_vectors"] = [
        {
            "time": "2024-07-09T04:03:29Z",
            "position": [6878137.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 7500.0],
        }
    ]

    json_path = tmp_path / "test_geometry.json"
    json_path.write_text(json.dumps(metadata_dict))

    exit_code = cli.main(["geometry", str(json_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "flight" in captured.out
    assert "look" in captured.out
    # N heading: flight arrow has ^ arrowhead
    assert "^ flight" in captured.out
    # E look: horizontal arrow with > arrowhead
    assert ">" in captured.out
    assert "look" in captured.out
    # Summary metadata
    assert "capella-14 | spotlight | HH" in captured.out
    assert "ascending | right-looking" in captured.out
    assert "heading: 0.0 deg (N)" in captured.out
    assert "look: 90.0 deg (E)" in captured.out
    assert "incidence: 30.0 deg" in captured.out


def test_cli_geometry_descending_left(tmp_path, sample_metadata_dict, capsys):
    """Test geometry for descending orbit with left-looking radar."""
    metadata_dict = deepcopy(sample_metadata_dict)
    metadata_dict["collect"]["image"]["center_pixel"]["target_position"] = [
        6378137.0,
        0.0,
        0.0,
    ]
    metadata_dict["collect"]["radar"]["pointing"] = "left"
    metadata_dict["collect"]["state"]["direction"] = "descending"
    metadata_dict["collect"]["state"]["state_vectors"] = [
        {
            "time": "2024-07-09T04:03:29Z",
            "position": [6878137.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, -7500.0],
        }
    ]

    json_path = tmp_path / "test_geometry_desc.json"
    json_path.write_text(json.dumps(metadata_dict))

    exit_code = cli.main(["geometry", str(json_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    # Descending: flight arrow points down with v arrowhead
    assert "v flight" in captured.out
    # Left-looking from southward heading -> look is east with > arrowhead
    assert ">" in captured.out
    assert "descending | left-looking" in captured.out
