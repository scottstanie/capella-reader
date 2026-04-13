#!/usr/bin/env python
"""Restore the Doppler deramping phase in a Capella spotlight SLC.

Capella spotlight SLC products are deramped and basebanded by the on-ground
processor: a geometry-dependent phase is removed so that the Doppler spectrum
sits at baseband. This is efficient for storage but breaks interferometric
techniques like phase linking because the phase of a pixel no longer
corresponds to the two-way slant range between the antenna and the target.

The removed phase, for a target P, is

    phi_P = -4 * pi / lambda * ( |ARP - P| - |ARP - P0| )

where ARP is the annotated Antenna Reference Position, P0 is the Scene
Reference Point (both in ECEF, both from the SLC metadata), and P is the
ECEF position of the ground target underneath the pixel. Multiplying the
deramped SLC by ``exp(-1j * phi_P)`` returns it to zero-Doppler geometry.

See ``spotlight_phase_restoration.md`` for a longer explanation.

Dependencies: isce3, capella-reader, numpy, gdal
Optional: sardem (auto-downloads a Copernicus DEM if --dem-file is omitted)

Usage
-----
python coregister_spotlight.py SLC.tif [--dem-file DEM.tif] [--output-dir ./restore]

"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from os import fsdecode
from pathlib import Path

import isce3
import numpy as np
from osgeo import gdal

import capella_reader.adapters.isce3
from capella_reader import CapellaSLC

gdal.UseExceptions()

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

DEFAULT_LINES_PER_BLOCK = 1024


# ---------------------------------------------------------------------------
# 1. DEM (same pattern as coregister_isce3.py)
# ---------------------------------------------------------------------------


def create_dem(slc_file: Path, output_dir: Path, dem_file: Path | None) -> Path:
    """Return a DEM covering the SLC extent; auto-download Copernicus if None."""
    if dem_file is not None:
        return dem_file

    import sardem.dem

    out = output_dir / "dem.tif"
    if out.exists():
        print(f"  DEM already exists: {out}")
        return out

    slc = CapellaSLC.from_file(slc_file)
    w, s, e, n = slc.bounds
    pad = 0.3  # degrees
    bbox = (w - pad, s - pad, e + pad, n + pad)
    print(f"  Downloading Copernicus DEM for bbox {bbox} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    sardem.dem.main(
        output_name=str(out),
        bbox=bbox,
        data_source="COP",
        output_type="float32",
        output_format="GTiff",
    )
    return out


# ---------------------------------------------------------------------------
# 2. Per-pixel lon / lat / height via rdr2geo
# ---------------------------------------------------------------------------

# Note: rdr2geo.topo() in isce3 expects all layer rasters positionally. Only
# the first three (x = lon, y = lat, z = height) are used by the phase
# restoration below; the rest are scratch files.
GEOMETRY_LAYERS = {
    "x": gdal.GDT_Float64,
    "y": gdal.GDT_Float64,
    "z": gdal.GDT_Float64,
    "incidence_angle": gdal.GDT_Float32,
    "heading_angle": gdal.GDT_Float32,
    "local_incidence_angle": gdal.GDT_Float32,
    "local_psi": gdal.GDT_Float32,
    "simulated_amplitude": gdal.GDT_Float32,
    "layover_shadow": gdal.GDT_Byte,
    "los_east": gdal.GDT_Float32,
    "los_north": gdal.GDT_Float32,
}


def run_geometry(slc_file: Path, dem_file: Path, output_dir: Path) -> Path:
    """Compute lon / lat / height per pixel and return the 3-band geometry VRT."""
    geom_dir = output_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    out_vrt = geom_dir / "geometry.vrt"

    slc = CapellaSLC.from_file(slc_file)
    radar_grid = capella_reader.adapters.isce3.get_radar_grid(slc)
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        orbit = capella_reader.adapters.isce3.get_orbit(slc)
    ellipsoid = isce3.core.make_projection(4326).ellipsoid

    rdr2geo = isce3.geometry.Rdr2Geo(
        radar_grid,
        orbit,
        ellipsoid,
        isce3.core.LUT2d(),
        threshold=1e-8,
        numiter=20,
        extraiter=10,
        lines_per_block=1024,
    )

    rasters = [
        isce3.io.Raster(
            fsdecode(geom_dir / f"{name}.tif"),
            radar_grid.width,
            radar_grid.length,
            1,
            dtype,
            "GTiff",
        )
        for name, dtype in GEOMETRY_LAYERS.items()
    ]

    t0 = time.time()
    rdr2geo.topo(isce3.io.Raster(fsdecode(dem_file)), *rasters)
    print(f"  rdr2geo took {time.time() - t0:.1f} s")

    # Stack x / y / z into a 3-band VRT (band 1 = lon, 2 = lat, 3 = height).
    stack = isce3.io.Raster(fsdecode(out_vrt), rasters[:3])
    stack.set_epsg(rdr2geo.epsg_out)
    del stack, rasters
    return out_vrt


# ---------------------------------------------------------------------------
# 3. Phase restoration (the actual point of this example)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpotlightGeometry:
    """Reference geometry for spotlight phase restoration.

    Holds the Antenna Reference Position (ARP), the Scene Reference Point (P0,
    in the Capella spec "reference_target_position"), and the radar wavelength.
    Everything is in ECEF / meters.
    """

    reference_antenna_position: np.ndarray  # shape (3,)
    scene_reference_point: np.ndarray  # shape (3,)
    wavelength: float

    @classmethod
    def from_capella_slc(cls, slc: CapellaSLC) -> SpotlightGeometry:
        image = slc.meta.collect.image
        if image.reference_antenna_position is None:
            msg = "SLC metadata is missing reference_antenna_position"
            raise ValueError(msg)
        if image.reference_target_position is None:
            msg = "SLC metadata is missing reference_target_position"
            raise ValueError(msg)
        return cls(
            reference_antenna_position=image.reference_antenna_position.as_array(),
            scene_reference_point=image.reference_target_position.as_array(),
            wavelength=slc.wavelength,
        )

    @property
    def reference_range(self) -> float:
        """Slant range from the antenna to the scene reference point (meters)."""
        return float(
            np.linalg.norm(
                self.reference_antenna_position - self.scene_reference_point
            )
        )


def llh_to_ecef_wgs84(
    lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray
) -> np.ndarray:
    """Convert geodetic lon/lat/height to ECEF. Returns shape (..., 3)."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + height_m) * cos_lat * cos_lon
    y = (N + height_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + height_m) * sin_lat
    return np.stack([x, y, z], axis=-1)


def compute_restoration_phase(
    geometry: SpotlightGeometry, target_ecef: np.ndarray
) -> np.ndarray:
    """phi_P = -4 * pi / lambda * ( |ARP - P| - |ARP - P0| )."""
    r = np.linalg.norm(target_ecef - geometry.reference_antenna_position, axis=-1)
    return (-4.0 * np.pi / geometry.wavelength) * (r - geometry.reference_range)


def apply_spotlight_phase_restoration(
    slc_file: Path,
    geometry_vrt: Path,
    output_file: Path,
    *,
    lines_per_block: int = DEFAULT_LINES_PER_BLOCK,
) -> Path:
    """Restore the deramping phase, block by block, to a corrected GeoTIFF.

    Parameters
    ----------
    slc_file
        Capella spotlight SLC (GeoTIFF) to correct.
    geometry_vrt
        3-band VRT with band 1 = lon, 2 = lat, 3 = height (output of
        :func:`run_geometry`). Must have the same shape as ``slc_file``.
    output_file
        Destination GeoTIFF. Written as complex64 with the same shape as
        the input.
    lines_per_block
        Row block height for processing.
    """
    slc = CapellaSLC.from_file(slc_file)
    geometry = SpotlightGeometry.from_capella_slc(slc)

    geo_ds = gdal.Open(fsdecode(geometry_vrt))
    slc_ds = gdal.Open(fsdecode(slc_file))
    rows, cols = slc_ds.RasterYSize, slc_ds.RasterXSize
    if (geo_ds.RasterYSize, geo_ds.RasterXSize) != (rows, cols):
        msg = (
            f"Geometry shape ({geo_ds.RasterYSize}, {geo_ds.RasterXSize}) "
            f"does not match SLC shape ({rows}, {cols})"
        )
        raise ValueError(msg)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        fsdecode(output_file),
        cols,
        rows,
        1,
        gdal.GDT_CFloat32,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"],
    )

    lon_band = geo_ds.GetRasterBand(1)
    lat_band = geo_ds.GetRasterBand(2)
    hgt_band = geo_ds.GetRasterBand(3)
    slc_band = slc_ds.GetRasterBand(1)
    out_band = out_ds.GetRasterBand(1)

    t0 = time.time()
    for r0 in range(0, rows, lines_per_block):
        nrow = min(lines_per_block, rows - r0)
        lon = lon_band.ReadAsArray(0, r0, cols, nrow)
        lat = lat_band.ReadAsArray(0, r0, cols, nrow)
        hgt = hgt_band.ReadAsArray(0, r0, cols, nrow)
        slc_data = slc_band.ReadAsArray(0, r0, cols, nrow)

        target_ecef = llh_to_ecef_wgs84(lon, lat, hgt)
        phi = compute_restoration_phase(geometry, target_ecef)
        corrected = (slc_data * np.exp(-1j * phi)).astype(np.complex64, copy=False)

        out_band.WriteArray(corrected, 0, r0)
        print(f"  block {r0 + nrow}/{rows}", end="\r")
    print()
    print(f"  phase restoration took {time.time() - t0:.1f} s")

    # Preserve the Capella TIFF metadata tag so downstream readers still work.
    out_ds.SetMetadataItem(
        "TIFFTAG_IMAGEDESCRIPTION",
        slc_ds.GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION"),
    )
    geo_ds = slc_ds = out_ds = None
    return output_file


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the Doppler deramping phase in a Capella spotlight SLC.",
    )
    parser.add_argument("slc", type=Path, help="Capella spotlight SLC (GeoTIFF)")
    parser.add_argument(
        "--dem-file",
        type=Path,
        default=None,
        help="DEM in EPSG:4326 (auto-downloaded via sardem if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spotlight_restore_output"),
        help="Directory for intermediate files (DEM, geometry layers)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SLC path (defaults to <output-dir>/<stem>.restored.tif)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or output_dir / f"{args.slc.stem}.restored.tif"
    t_start = time.time()

    print("[1/3] DEM")
    dem_file = create_dem(args.slc, output_dir, args.dem_file)

    print("[2/3] Geometry (rdr2geo)")
    geometry_vrt = run_geometry(args.slc, dem_file, output_dir)

    print("[3/3] Phase restoration")
    apply_spotlight_phase_restoration(args.slc, geometry_vrt, output_file)

    print(f"\nDone in {time.time() - t_start:.1f} s")
    print(f"Restored SLC: {output_file}")


if __name__ == "__main__":
    main()
