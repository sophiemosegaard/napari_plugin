from pathlib import Path

import numpy as np
from skimage.draw import polygon2mask


def _read_rois(roi_path: Path):
    """Read one ImageJ ROI file or ROI Manager ZIP."""

    try:
        from roifile import roiread
    except ImportError as error:
        raise RuntimeError(
            "Install roifile with: "
            "python -m pip install roifile"
        ) from error

    return roiread(str(roi_path))


def image_stack_shape(
    image,
) -> tuple[int, tuple[int, int]]:
    """Return number of images and their Y, X shape."""

    shape = tuple(image.shape)

    # One grayscale image: (Y, X)
    if len(shape) == 2:
        return 1, shape

    # One RGB image: (Y, X, 3)
    if len(shape) == 3 and shape[-1] in (3, 4):
        return 1, shape[:2]

    # Grayscale image stack: (N, Y, X)
    if len(shape) == 3:
        return shape[0], shape[1:]

    # RGB image stack: (N, Y, X, 3)
    if len(shape) == 4 and shape[-1] in (3, 4):
        return shape[0], shape[1:3]

    raise ValueError(
        "Expected (Y, X), (Y, X, 3), "
        "(N, Y, X), or (N, Y, X, 3), "
        f"but received {shape}."
    )


def load_imagej_roi_mask(
    roi_path: Path,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert one ImageJ ROI ZIP into one mask."""

    roi_path = Path(roi_path)

    if roi_path.suffix.lower() not in {".roi", ".zip"}:
        raise ValueError(
            "Expected an ImageJ .roi or .zip file."
        )

    rois = _read_rois(roi_path)

    # A single .roi may return one object.
    if hasattr(rois, "coordinates"):
        rois = [rois]

    mask = np.zeros(
        image_shape,
        dtype=bool,
    )

    for roi in rois:
        xy = np.asarray(roi.coordinates())

        # Ignore points and open lines.
        if xy.ndim != 2 or len(xy) < 3:
            continue

        # ImageJ coordinates: (x, y)
        # skimage coordinates: (y, x)
        mask |= polygon2mask(
            image_shape,
            xy[:, [1, 0]],
        )

    if not mask.any():
        raise ValueError(
            f"No filled ROI found in {roi_path.name}."
        )

    return mask.astype(np.uint8)


def load_imagej_roi_folder(
    roi_folder: Path,
    number_images: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, list[Path]]:
    """Load one ROI file for every image."""

    roi_folder = Path(roi_folder)

    if not roi_folder.is_dir():
        raise ValueError(
            f"Not a valid ROI folder: {roi_folder}"
        )

    roi_paths = sorted(
        path
        for path in roi_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".roi", ".zip"}
    )

    if len(roi_paths) != number_images:
        raise ValueError(
            f"The image layer contains {number_images} images, "
            f"but the ROI folder contains "
            f"{len(roi_paths)} ROI files."
        )

    masks = [
        load_imagej_roi_mask(
            roi_path=path,
            image_shape=image_shape,
        )
        for path in roi_paths
    ]

    roi_stack = np.stack(masks)

    # Return an ordinary 2D mask for one image.
    if number_images == 1:
        roi_stack = roi_stack[0]

    return roi_stack, roi_paths