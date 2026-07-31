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
MIN_CIRCLE_INSIDE_FRACTION = 0.80


def _read_grayscale(path: Path) -> np.ndarray:
    """Read one 2D grayscale or RGB image as float32."""
    return img_as_float32(as_grayscale(imread(path)))

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

    return img_as_float32(image)


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
    visible_rr, _ = disk(
        (y, x),
        radius,
        shape=image_shape[:2],
    )

    # Pixels of a complete circle with the same radius.
    full_rr, _ = disk(
        (radius, radius),
        radius,
        shape=(2 * radius + 1, 2 * radius + 1),
    )

    return len(visible_rr) / len(full_rr)


def detect_circles(
    image: np.ndarray,
    min_radius_px: int,
    max_radius_px: int,
    radius_step_px: int,
    max_circles: int,
    detection_max_size: int,
) -> list[tuple[int, int, int, float]]:
    """Detect circles that are at least 80% inside the image."""

    # Use grayscale only for Hough detection.
    gray = as_grayscale(image)

    small, scale = _detection_image(gray, detection_max_size)
    min_radius = max(2, round(min_radius_px * scale))
    max_radius = max(min_radius, round(max_radius_px * scale))
    radius_step = max(1, round(radius_step_px * scale))
    radii = np.arange(min_radius, max_radius + 1, radius_step)
    edges = canny(small, sigma=2.0) # FIXME How the canny edge detection is performed can vary a lot between images. Not quite sure how to deal with this fact or if I just set one value?
    hough = hough_circle(edges, radii, normalize=True)

    # Request extra candidates because border circles
    # may be rejected by the 80% condition.
    number_of_peaks = max(10, max_circles * 5)

    scores, xs, ys, found_radii = hough_circle_peaks(
        hough,
        radii,
        min_xdistance=min_radius,
        min_ydistance=min_radius,
        total_num_peaks=number_of_peaks,
    )

    circles = []

    for score, x, y, radius in zip(
        scores,
        xs,
        ys,
        found_radii,
    ):
        full_y = round(y / scale)
        full_x = round(x / scale)
        full_radius = round(radius / scale)

        inside_fraction = _circle_inside_fraction(image.shape, full_y, full_x, full_radius)

        if inside_fraction < MIN_CIRCLE_INSIDE_FRACTION:
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
    rows: int = 16,
    columns: int = 16,
    min_radius_px: int = 140,
    max_radius_px: int = 180,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Detect organoids, choose them randomly, and build image/label mosaics."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a valid folder: {folder}")
    if min_radius_px >= max_radius_px:
        raise ValueError("min_radius_px must be smaller than max_radius_px.")
    
    # Patch diameter plus approximately 15% margin on every side
    patch_size = int(np.ceil(2.3 * max_radius_px))
    
    # Test approximately 20 different radii
    radius_range = max_radius_px - min_radius_px
    radius_step_px = max(1, round(radius_range / 20))
    
    # Downsampling limit used only to make detection faster.
    detection_max_size = 1000
    
    # Use this when every image contains one microwell. 
    max_circles_per_image = 2 # FIXME not sure how I should standardize this, depends a lot on the data that the user has.....
    
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported image files found in: {folder}, supported suffixes: {IMAGE_SUFFIXES}")

    candidates: list[tuple[Path, int, int, int, float]] = []
    skipped: list[str] = []

    for path in files:
        try:
            # image = _read_grayscale(path)
            image = _read_rgb(path)
            circles = detect_circles(
                image,
                min_radius_px,
                max_radius_px,
                radius_step_px,
                max_circles_per_image,
                detection_max_size,
            )
            candidates.extend((path, *circle) for circle in circles)
        except Exception as error:
            skipped.append(f"{path.name}: {error}")

    number_needed = rows * columns
    if len(candidates) < number_needed:
        details = f" First skipped file: {skipped[0]}" if skipped else ""
        raise ValueError(
            f"Found {len(candidates)} circles, but {number_needed} are needed. "
            "Try fewer rows/columns or a wider radius range."
            + details
        )

    rng = np.random.default_rng(0) # I have set a random seed to ensure reproducibility of the mosaic generation.
    selected_indices = rng.choice(len(candidates), number_needed, replace=False)

    mosaic = np.zeros((rows * patch_size, columns * patch_size, 3), np.float32)
    labels = np.zeros(mosaic.shape[:2], np.uint8)
    records: list[dict] = []

    for tile_index, candidate_index in enumerate(selected_indices):
        path, y, x, radius, score = candidates[int(candidate_index)]
        # patch = _crop_centered(_read_grayscale(path), y, x, patch_size)
        patch = _crop_centered(_read_rgb(path), y, x, patch_size)

        row, column = divmod(tile_index, columns)
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
