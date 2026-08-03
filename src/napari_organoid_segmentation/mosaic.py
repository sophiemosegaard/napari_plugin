from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.feature import canny
from skimage.io import imread
from skimage.transform import hough_circle, hough_circle_peaks, resize
from skimage.util import img_as_float32

from .preprocessing import as_grayscale

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"} # Works for all common image formats.
HOUGH_RADIUS_TOLERANCE = 0.08
MOSAIC_MIN_INSIDE_FRACTION = 0.80


# def _read_grayscale(path: Path) -> np.ndarray:
#     """Read one 2D grayscale or RGB image as float32."""
#     return img_as_float32(as_grayscale(imread(path)))

def _read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB float32 with shape (Y, X, 3)."""

    image = np.asarray(imread(path))

    # Convert grayscale images to three identical channels.
    if image.ndim == 2:
        image = np.repeat(
            image[..., np.newaxis],
            3,
            axis=-1,
        )

    # Remove the alpha channel from RGBA images.
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]

    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    # return img_as_float32(image)
    return image.astype(np.float32, copy=False)


def _detection_image(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """Downsample and normalize an image for circle detection."""
    scale = min(1.0, max_size / max(image.shape))
    small = image

    if scale < 1.0:
        shape = tuple(max(1, round(size * scale)) for size in image.shape)
        small = resize(image, shape, anti_aliasing=True, preserve_range=True)

    low, high = np.percentile(small, (1, 99))
    if high > low:
        small = np.clip((small - low) / (high - low), 0, 1)

    return small.astype(np.float32), scale

def _circle_inside_fraction(
    image_shape: tuple[int, ...],
    y: int,
    x: int,
    radius: int,
) -> float:
    """Return the fraction of the circle inside the image."""

    # Pixels of the circle that are inside the source image.
    visible_rr, _ = disk((y, x), radius, shape=image_shape[:2])

    # Pixels of a complete circle with the same radius.
    full_rr, _ = disk((radius, radius), radius, shape=(2 * radius + 1, 2 * radius + 1))

    return len(visible_rr) / len(full_rr)

def _estimate_max_wells(
    image_shape: tuple[int, ...],
    well_diameter_px: int,
) -> int:
    """Estimate a safe maximum number of wells."""

    height, width = image_shape[:2]

    rows = int(np.ceil(height / well_diameter_px)) + 2
    columns = int(np.ceil(width / well_diameter_px)) + 2

    return max(1, rows * columns)

def detect_circles(
    image: np.ndarray,
    well_diameter_px: int,
    detection_max_size: int,
    min_inside_fraction: float = 0.80,
    detect_outside_centers: bool = False,
) -> list[tuple[int, int, int, float]]:
    """Detect wells close to the measured well diameter."""

    if well_diameter_px < 4:
        raise ValueError("well_diameter_px must be at least 4 pixels.")

    if not 0.0 <= min_inside_fraction <= 1.0:
        raise ValueError("min_inside_fraction must be between 0 and 1.")

    gray = as_grayscale(image)
    small, scale = _detection_image(gray, detection_max_size)

    # Expected radius in the possibly downsampled image.
    expected_radius = max(2, round(well_diameter_px * scale / 2))

    # Search a small range around the measured size.
    radius_tolerance = max(1, round(expected_radius * HOUGH_RADIUS_TOLERANCE))
    minimum_radius = max(2, expected_radius - radius_tolerance)
    maximum_radius = (expected_radius + radius_tolerance)
    radii = np.arange(minimum_radius, maximum_radius + 1)

    edges = canny(small, sigma=2.0) # FIXME How the canny edge detection is performed can vary a lot between images. Not quite sure how to deal with this fact or if I just set one value?
    hough = hough_circle(edges, radii, normalize=True, full_output=detect_outside_centers)

    # Calculate a safe detection limit internally.
    max_circles = _estimate_max_wells(image_shape=image.shape, well_diameter_px=well_diameter_px)

    # Request extra candidates because some may later
    # be rejected by the border criterion.
    number_of_peaks = max(10, max_circles * 3)

    scores, xs, ys, found_radii = (
        hough_circle_peaks(
            hough,
            radii,
            min_xdistance=expected_radius,
            min_ydistance=expected_radius,
            total_num_peaks=number_of_peaks,
        )
    )

    # full_output=True adds padding around the accumulator.
    hough_padding = (int(radii.max()) if detect_outside_centers else 0)

    circles = []

    for score, x, y, radius in zip(
        scores,
        xs,
        ys,
        found_radii,
    ):
        # Convert coordinates back from the padded,
        # downsampled detection image.
        small_y = int(y) - hough_padding
        small_x = int(x) - hough_padding

        full_y = round(small_y / scale)
        full_x = round(small_x / scale)
        full_radius = round(radius / scale)

        inside_fraction = _circle_inside_fraction(image.shape, full_y, full_x, full_radius)

        if inside_fraction < min_inside_fraction:
            continue

        circles.append((full_y, full_x, full_radius, float(score)))

        if len(circles) == max_circles:
            break

    return circles

def _crop_centered(image: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    """Crop an RGB patch and pad image borders."""
    half = size // 2
    y0, x0 = y - half, x - half
    y1, x1 = y0 + size, x0 + size

    # Create the output patch using the image median.
    patch = np.empty((size, size, 3), dtype=image.dtype)

    patch[:] = np.median(image, axis=(0, 1))

    source_y0 = max(0, y0)
    source_x0 = max(0, x0)
    source_y1 = min(image.shape[0], y1)
    source_x1 = min(image.shape[1], x1)

    target_y0 = source_y0 - y0
    target_x0 = source_x0 - x0

    target_y1 = target_y0 + source_y1 - source_y0
    target_x1 = target_x0 + source_x1 - source_x0

    patch[
        target_y0:target_y1,
        target_x0:target_x1,
    ] = image[
        source_y0:source_y1,
        source_x0:source_x1,
    ]

    return patch


def create_organoid_mosaic(
    folder: Path,
    patches_per_side: int = 4,
    well_diameter_px: int = 280,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Detect organoids, choose them randomly, and build image/label mosaics."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a valid folder: {folder}")
    
    if patches_per_side < 1:
        raise ValueError("patches_per_side must be at least 1.")
    
    if well_diameter_px < 4:
        raise ValueError("well_diameter_px must be at least 4 pixels.")

    # Well diameter plus 15% surrounding space.
    patch_size = int(np.ceil(1.15 * well_diameter_px))
    
    # Downsampling limit used only to make detection faster.
    detection_max_size = 1000
    
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported image files found in: {folder}, supported suffixes: {IMAGE_SUFFIXES}")
    
    circles_by_image: dict[
        Path,
        list[tuple[int, int, int, float]],
    ] = {}

    skipped: list[str] = []

    for path in files:
        try:
            image = _read_rgb(path)

            circles = detect_circles(
                image=image,
                well_diameter_px=well_diameter_px,
                detection_max_size=detection_max_size,
                min_inside_fraction=MOSAIC_MIN_INSIDE_FRACTION,
                detect_outside_centers=False,
            )

            if circles:
                circles_by_image[path] = circles

        except Exception as error:
            skipped.append(f"{path.name}: {error}")
    
    # number_needed = rows * columns
    number_needed = patches_per_side * patches_per_side

    total_detected = sum(len(circles) for circles in circles_by_image.values())

    if total_detected < number_needed:
        details = (f" First skipped file: {skipped[0]}" if skipped else "")

        raise ValueError(
            f"Found {total_detected} circles, "
            f"but {number_needed} are needed. "
            "Try fewer rows/columns or a wider radius range."
            + details
        )

    rng = np.random.default_rng(36) # Set random seed for reproducability.

    first_choices: list[tuple[Path, int, int, int, float]] = []

    remaining_choices: list[tuple[Path, int, int, int, float]] = []

    for path, circles in circles_by_image.items():
        # The Hough results are normally ordered by score.
        # Sorting makes this intention explicit.
        sorted_circles = sorted(
            circles,
            key=lambda circle: circle[3],
            reverse=True,
        )

        # Strongest detected well from this image.
        first_choices.append((path, *sorted_circles[0]))
        
        # Other wells are fallback choices.
        remaining_choices.extend((path, *circle) for circle in sorted_circles[1:])

    # Randomize the order of the source images.
    rng.shuffle(first_choices)

    # First use at most one well from every image.
    selected_candidates = first_choices[:number_needed]

    # Only reuse images when the mosaic needs more wells.
    if len(selected_candidates) < number_needed:
        rng.shuffle(remaining_choices)

        number_missing = (number_needed - len(selected_candidates))

        selected_candidates.extend(remaining_choices[:number_missing])

    mosaic = np.zeros((patches_per_side * patch_size, patches_per_side * patch_size, 3), np.float32)
    labels = np.zeros(mosaic.shape[:2], np.uint8)
    records: list[dict] = []
    
    for tile_index, candidate in enumerate(selected_candidates):
        path, y, x, radius, score = candidate
        
        patch = _crop_centered(_read_rgb(path), y, x, patch_size)

        row, column = divmod(tile_index, patches_per_side)
        y0, x0 = row * patch_size, column * patch_size
        mosaic[y0 : y0 + patch_size, x0 : x0 + patch_size] = patch

        records.append(
            {
                "tile_id": tile_index + 1,
                "row": row,
                "column": column,
                "source": str(path),
                "center_y_px": y,
                "center_x_px": x,
                "radius_px": radius,
                "hough_score": score,
            }
        )

    return mosaic, labels, records
