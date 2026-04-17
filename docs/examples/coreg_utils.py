"""Shared cross-correlation and polynomial fitting utilities for SAR coregistration.

Used by both the ISCE3 and sarpy coregistration examples. Provides FFT-based
cross-correlation with sub-pixel refinement and robust polynomial fitting
with outlier rejection.

Sub-pixel refinement uses upsampled DFT by matrix multiplication.

References
----------
.. [1] Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008).
   Efficient subpixel image registration algorithms. Optics Letters, 33(2),
   156-158. https://doi.org/10.1364/OL.33.000156
"""

import numpy as np
from numpy.polynomial.polynomial import polyval2d, polyvander2d
from scipy.fft import fftn, ifftn


def _upsampled_dft(
    data: np.ndarray,
    upsampled_region_size: int,
    upsample_factor: int,
    axis_offsets: tuple[float, float],
) -> np.ndarray:
    """Upsampled DFT by matrix multiplication (Guizar et al. 2008).

    Computes the cross-correlation in a small neighborhood around the peak
    at sub-pixel resolution without zero-padding the full array.
    """
    im2pi = 1j * 2 * np.pi
    n_rows, n_cols = data.shape
    row_offset, col_offset = axis_offsets

    row_kernel = (np.arange(upsampled_region_size) - row_offset)[
        :, None
    ] * np.fft.fftfreq(n_rows, upsample_factor)
    row_kernel = np.exp(-im2pi * row_kernel)

    col_kernel = (np.arange(upsampled_region_size) - col_offset)[
        :, None
    ] * np.fft.fftfreq(n_cols, upsample_factor)
    col_kernel = np.exp(-im2pi * col_kernel)

    return row_kernel @ data @ col_kernel.T


def correlate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    expected_shift: tuple[float, float] = (0.0, 0.0),
    search_radius: tuple[int, int] | None = None,
    upsample_factor: int = 1,
) -> tuple[np.ndarray, float, float]:
    """Estimate sub-pixel translation between two images via FFT cross-correlation.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (2D, real or complex).
    moving : np.ndarray
        Moving image (same shape as reference).
    expected_shift : tuple[float, float]
        Expected (row, col) shift for bounded search.
    search_radius : tuple[int, int], optional
        Search window half-size (row, col). If None, search entire image.
    upsample_factor : int
        Sub-pixel refinement factor (1 = pixel-level only).

    Returns
    -------
    shift : np.ndarray
        Estimated (row, col) shift, shape (2,).
    snr : float
        Signal-to-noise ratio of the correlation peak.
    peak_ncc : float
        Normalized cross-correlation at the peak.
    """
    assert reference.shape == moving.shape

    ref = (
        np.abs(reference)
        if np.iscomplexobj(reference)
        else reference.astype(np.float32, copy=False)
    )
    mov = (
        np.abs(moving)
        if np.iscomplexobj(moving)
        else moving.astype(np.float32, copy=False)
    )
    # Demean for robust SNR
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    H, W = ref.shape

    F = fftn(ref)
    G = fftn(mov)
    prod = F * np.conj(G)
    cc = ifftn(prod)
    cc_abs = np.abs(cc)
    src_energy = float(np.sum(ref**2))
    mov_energy = float(np.sum(mov**2))

    # Bounded search around expected shift
    if search_radius is None:
        peak_idx = int(np.argmax(cc_abs))
        peak = (peak_idx // W, peak_idx % W)
        peak_cc_abs = float(cc_abs.ravel()[peak_idx])
    else:
        ry, rx = search_radius
        ry = min(ry, H // 2 - 1)
        rx = min(rx, W // 2 - 1)
        cy = int(np.round(expected_shift[0])) % H
        cx = int(np.round(expected_shift[1])) % W
        rows = (np.arange(-ry, ry + 1) + cy) % H
        cols = (np.arange(-rx, rx + 1) + cx) % W
        window = cc_abs[np.ix_(rows, cols)]
        w_peak_idx = int(np.argmax(window))
        w_shape = (2 * ry + 1, 2 * rx + 1)
        pr = w_peak_idx // w_shape[1]
        pc = w_peak_idx % w_shape[1]
        peak = (int(rows[pr]), int(cols[pc]))
        peak_cc_abs = float(cc_abs[peak])

    # Signed shift
    shift = np.array([float(peak[0]), float(peak[1])], dtype=np.float64)
    midpoint = np.array([np.floor(H / 2), np.floor(W / 2)], dtype=np.float64)
    dims = np.array([H, W], dtype=np.float64)
    shift[shift > midpoint] -= dims[shift > midpoint]

    # Sub-pixel refinement via upsampled DFT
    if upsample_factor > 1:
        shift_rounded = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = int(np.ceil(upsample_factor * 1.5))
        dftshift = np.fix(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift_rounded * upsample_factor

        up_cc = np.conj(
            _upsampled_dft(
                np.conj(prod),
                upsampled_region_size,
                upsample_factor,
                (sample_region_offset[0], sample_region_offset[1]),
            )
        )
        up_cc_abs = np.abs(up_cc)
        up_peak_idx = int(np.argmax(up_cc_abs))
        up_peak = (
            up_peak_idx // upsampled_region_size,
            up_peak_idx % upsampled_region_size,
        )
        shift = (
            shift_rounded
            + (np.array(up_peak, dtype=np.float64) - dftshift) / upsample_factor
        )

    # Metrics
    denom = np.sqrt(src_energy * mov_energy)
    peak_ncc = float(peak_cc_abs / max(denom, 1e-12))

    # SNR: peak / RMS of background (excluding guard region around peak)
    guard = max(3, int(min(cc_abs.shape) / 32))
    pr, pc = peak[0], peak[1]
    mask = np.ones(cc_abs.shape, dtype=bool)
    r0, r1 = max(0, pr - guard), min(H, pr + guard + 1)
    c0, c1 = max(0, pc - guard), min(W, pc + guard + 1)
    mask[r0:r1, c0:c1] = False
    bg = cc_abs[mask].astype(np.float64)
    rms = float(np.sqrt(np.mean(bg**2))) if bg.size >= 16 else 1e-12
    snr = float(peak_cc_abs / max(rms, 1e-12))

    return shift.astype(np.float32), snr, peak_ncc


def estimate_bulk_offset(
    ref_data: np.ndarray,
    sec_data: np.ndarray,
    chip_size: tuple[int, int] = (2048, 2048),
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate global (dy, dx) by median of large-chip correlations.

    Parameters
    ----------
    ref_data : np.ndarray
        Reference image data.
    sec_data : np.ndarray
        Secondary image data (same shape).
    chip_size : tuple[int, int]
        Size of correlation chips.

    Returns
    -------
    median_shift : np.ndarray
        Median (row, col) shift across chips, shape (2,).
    mad : np.ndarray
        Median absolute deviation of shifts, shape (2,).
    """
    nrows, ncols = ref_data.shape
    ch, cw = chip_size
    shifts = []

    for r0 in range(0, nrows - ch + 1, ch):
        for c0 in range(0, ncols - cw + 1, cw):
            ref_chip = ref_data[r0 : r0 + ch, c0 : c0 + cw]
            sec_chip = sec_data[r0 : r0 + ch, c0 : c0 + cw]
            ref_mag = np.abs(ref_chip) if np.iscomplexobj(ref_chip) else ref_chip
            if np.var(ref_mag) < 1e-6:
                continue
            s, _, _ = correlate_translation(ref_chip, sec_chip)
            shifts.append(s.astype(np.float64))

    if len(shifts) == 0:
        print("  WARNING: no valid chips for bulk offset, returning zero")
        return np.zeros(2, dtype=np.float32), np.full(2, 32.0, dtype=np.float32)

    arr = np.vstack(shifts)
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0) * 1.4826
    return med.astype(np.float32), mad.astype(np.float32)


def correlate_grid(
    ref_data: np.ndarray,
    sec_data: np.ndarray,
    *,
    chip_size: tuple[int, int] = (256, 256),
    upsample_factor: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correlate at a grid of points across the image pair.

    Parameters
    ----------
    ref_data : np.ndarray
        Reference image data.
    sec_data : np.ndarray
        Secondary image data (same shape).
    chip_size : tuple[int, int]
        Size of correlation chips.
    upsample_factor : int
        Sub-pixel refinement factor.

    Returns
    -------
    row_centers : np.ndarray
        Row centers of correlation grid.
    col_centers : np.ndarray
        Column centers of correlation grid.
    az_offsets : np.ndarray
        Azimuth (row) offsets at each grid point.
    rg_offsets : np.ndarray
        Range (col) offsets at each grid point.
    snr : np.ndarray
        SNR at each grid point.
    peak_ncc : np.ndarray
        Normalized cross-correlation at each grid point.
    """
    nrows, ncols = ref_data.shape
    ch, cw = chip_size

    # Bulk offset first
    bulk_chip = (min(2048, nrows), min(2048, ncols))
    print("  Estimating bulk offset ...")
    bulk_shift, bulk_mad = estimate_bulk_offset(ref_data, sec_data, chip_size=bulk_chip)
    print(f"  Bulk shift (dy, dx): {bulk_shift}, MAD: {bulk_mad}")

    # Search radius from dispersion
    ry = int(min(max(32, np.ceil(4 * bulk_mad[0])), ch // 3))
    rx = int(min(max(32, np.ceil(4 * bulk_mad[1])), cw // 3))
    search_radius = (ry, rx)
    print(f"  Search radius: {search_radius}")

    idy, idx = int(np.round(bulk_shift[0])), int(np.round(bulk_shift[1]))
    fdy, fdx = float(bulk_shift[0]) - idy, float(bulk_shift[1]) - idx

    # Build grid
    row_centers = np.arange(ch // 2, nrows - ch // 2, ch)
    col_centers = np.arange(cw // 2, ncols - cw // 2, cw)
    rr, cc = np.meshgrid(row_centers, col_centers, indexing="ij")
    rr_flat = rr.ravel()
    cc_flat = cc.ravel()
    n_points = len(rr_flat)

    az_off = np.full(n_points, np.nan)
    rg_off = np.full(n_points, np.nan)
    snr_arr = np.full(n_points, np.nan)
    ncc_arr = np.full(n_points, np.nan)

    for i in range(n_points):
        rc, ccc = int(rr_flat[i]), int(cc_flat[i])
        r0, r1 = rc - ch // 2, rc + ch // 2
        c0, c1 = ccc - cw // 2, ccc + cw // 2
        if r0 < 0 or r1 > nrows or c0 < 0 or c1 > ncols:
            continue
        sr0, sr1 = r0 - idy, r1 - idy
        sc0, sc1 = c0 - idx, c1 - idx
        if sr0 < 0 or sr1 > nrows or sc0 < 0 or sc1 > ncols:
            continue

        chip_ref = ref_data[r0:r1, c0:c1]
        chip_sec = sec_data[sr0:sr1, sc0:sc1]
        s, snr, ncc = correlate_translation(
            chip_ref,
            chip_sec,
            expected_shift=(fdy, fdx),
            search_radius=search_radius,
            upsample_factor=upsample_factor,
        )
        az_off[i] = float(-idy) + float(-s[0])
        rg_off[i] = float(-idx) + float(-s[1])
        snr_arr[i] = snr
        ncc_arr[i] = ncc

        if (i + 1) % 50 == 0 or i == n_points - 1:
            print(f"  Correlated {i + 1}/{n_points} grid points", end="\r")

    print()
    return rr_flat, cc_flat, az_off, rg_off, snr_arr, ncc_arr


def fit_poly2d(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    degree: int = 1,
) -> np.ndarray:
    """Fit a 2D polynomial p(x, y) = sum c[i,j] x^i y^j via least squares.

    Parameters
    ----------
    x : np.ndarray
        X coordinates (1D).
    y : np.ndarray
        Y coordinates (1D).
    data : np.ndarray
        Values to fit (1D, same length as x and y).
    degree : int
        Polynomial degree in each dimension.

    Returns
    -------
    np.ndarray
        Coefficient matrix of shape (degree+1, degree+1).
    """
    deg = (degree, degree)
    vander = polyvander2d(x, y, deg)
    coeffs_flat = np.linalg.lstsq(vander, data, rcond=None)[0]
    return coeffs_flat.reshape(degree + 1, degree + 1)


def fit_polynomials_robust(
    rows: np.ndarray,
    cols: np.ndarray,
    az_off: np.ndarray,
    rg_off: np.ndarray,
    *,
    degree: int = 1,
    n_iterations: int = 5,
    outlier_threshold: float = 3.0,
    min_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit az/rg polynomials with joint MAD-based outlier rejection.

    Parameters
    ----------
    rows : np.ndarray
        Row coordinates of grid points.
    cols : np.ndarray
        Column coordinates of grid points.
    az_off : np.ndarray
        Azimuth offsets at grid points.
    rg_off : np.ndarray
        Range offsets at grid points.
    degree : int
        Polynomial degree.
    n_iterations : int
        Number of outlier rejection iterations.
    outlier_threshold : float
        Outlier threshold in sigma units.
    min_samples : int
        Minimum number of inlier samples to keep.

    Returns
    -------
    az_coeffs : np.ndarray
        Azimuth polynomial coefficients.
    rg_coeffs : np.ndarray
        Range polynomial coefficients.
    inlier_mask : np.ndarray
        Boolean mask of inlier points.
    """
    active = np.ones(len(rows), dtype=bool)

    for iteration in range(n_iterations):
        if np.sum(active) < min_samples:
            break

        r, c = rows[active], cols[active]
        az, rg = az_off[active], rg_off[active]

        az_c = fit_poly2d(r, c, az, degree)
        rg_c = fit_poly2d(r, c, rg, degree)

        az_resid = az - polyval2d(r, c, az_c)
        rg_resid = rg - polyval2d(r, c, rg_c)

        az_scale = 1.4826 * np.median(np.abs(az_resid - np.median(az_resid)))
        rg_scale = 1.4826 * np.median(np.abs(rg_resid - np.median(rg_resid)))
        if az_scale < 1e-10 or rg_scale < 1e-10:
            break

        joint = np.sqrt((az_resid / az_scale) ** 2 + (rg_resid / rg_scale) ** 2)
        outliers = joint > outlier_threshold * np.sqrt(2)
        n_out = int(np.sum(outliers))
        if n_out == 0:
            break

        # Map back to full-size mask
        idx = np.where(active)[0]
        keep = ~outliers
        if np.sum(keep) < min_samples:
            # Remove worst ones down to min
            n_to_remove = np.sum(active) - min_samples
            if n_to_remove <= 0:
                break
            worst = np.argsort(joint)[-n_to_remove:]
            keep = np.ones(len(joint), dtype=bool)
            keep[worst] = False
            n_out = int(n_to_remove)
        active[idx[~keep]] = False
        print(
            f"  Iteration {iteration + 1}: removed {n_out} outliers, {np.sum(active)}"
            " remain"
        )

    # Final fit
    r, c = rows[active], cols[active]
    az_coeffs = fit_poly2d(r, c, az_off[active], degree)
    rg_coeffs = fit_poly2d(r, c, rg_off[active], degree)
    return az_coeffs, rg_coeffs, active
