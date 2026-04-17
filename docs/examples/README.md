# Capella Reader Examples

This directory contains example scripts demonstrating how to use the capella-reader library to visualize satellite orbits and image footprints.

## Installation

To run these examples, you'll need to install additional dependencies:

```bash
pip install matplotlib cartopy pyproj
```

## Examples

### Coregistration and Interferometry

Two example scripts demonstrate SLC coregistration and interferogram formation
using different processing backends. Both share cross-correlation and polynomial
fitting utilities from `coreg_utils.py`.

#### `coregister_isce3.py` -- ISCE3 backend (stripmap)

Coregisters two Capella stripmap SLCs via DEM-based geometry (rdr2geo/geo2rdr)
plus amplitude cross-correlation refinement. Requires `isce3` and `gdal`.

```bash
python coregister_isce3.py REFERENCE.tif SECONDARY.tif [--dem-file DEM.tif]
```

#### `coregister_sarpy.py` -- sarpy backend (PFA / SICD)

Coregisters two SICD SLCs (e.g. Capella PFA spotlight products) and forms an
interferogram. Uses sarpy for I/O and point-projection geometry, scipy for
resampling. Accepts any format sarpy can read (SICD NITF, Capella GeoTIFF, etc.).

```bash
pip install sarpy scipy matplotlib
python coregister_sarpy.py REFERENCE.tif SECONDARY.tif [--output-dir ./coreg_sarpy]
```

### 1. 2D Ground Track Visualization (`orbit_ground_track.py`)

Visualizes the satellite ground track (orbit projected onto Earth's surface) with the image footprint.

```bash
python examples/orbit_ground_track.py
```

Features:
- Plots satellite position over time as lat/lon coordinates
- Shows image center location
- Displays image bounding box
- Indicates orbit direction (ascending/descending)

### 3. Combined Map Visualization (`orbit_footprint_map.py`)

Creates a comprehensive map showing both the orbit track and image footprint together.

```bash
python examples/orbit_footprint_map.py
```

Features:
- Ground track with time annotations
- Image footprint as a polygon
- Scene center point
- Coastlines and political boundaries

## Test Data

Examples use the test data files:
- [CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346_extended.json](https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.amazonaws.com/stac/capella-open-data-by-datetime/capella-open-data-2024/capella-open-data-2024-11/capella-open-data-2024-11-26/CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346/CAPELLA_C13_SP_SLC_HH_20241126045307_20241126045346.json?.language=en&.asset=asset-metadata)
- [CAPELLA_C11_SM_SLC_VV_20251031191104_20251031191109.json](https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.amazonaws.com/stac/capella-open-data-by-datetime/capella-open-data-2025/capella-open-data-2025-10/capella-open-data-2025-10-31/CAPELLA_C11_SM_SLC_VV_20251031191104_20251031191109/CAPELLA_C11_SM_SLC_VV_20251031191104_20251031191109.json?.language=en&.asset=asset-metadata)

This contains metadata for a Capella-13 Spotlight mode image collected on 2024-11-26.
