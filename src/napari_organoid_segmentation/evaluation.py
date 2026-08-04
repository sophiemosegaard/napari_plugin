from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread

from .roi_io import load_imagej_roi_mask


SEGMENTATION_SUFFIXES = {".tif", ".tiff", ".png"}
ROI_SUFFIXES = {".roi", ".zip"}


def _file_key(path: Path) -> str:
    """Return the original image name used to match files."""

    name = path.stem.lower()

    # Batch analysis saves images as original_name_labels.tif.
    for ending in (
        "_labels",
        "_label",
        "_mask",
        "_segmentation",
    ):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break

    return name


def _iou(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Calculate binary intersection over union."""

    intersection = np.count_nonzero(
        prediction & reference
    )

    union = np.count_nonzero(
        prediction | reference
    )

    # Both masks are empty.
    if union == 0:
        return 1.0

    return intersection / union


def calculate_segmentation_folder_iou(
    segmentation_folder: Path,
    roi_folder: Path,
    output_csv: Path,
) -> tuple[pd.DataFrame, Path]:
    """Compare saved segmentation masks with ImageJ ROIs."""

    segmentation_folder = Path(segmentation_folder)
    roi_folder = Path(roi_folder)

    segmentation_paths = sorted(
        path
        for path in segmentation_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SEGMENTATION_SUFFIXES
    )

    roi_paths = sorted(
        path
        for path in roi_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in ROI_SUFFIXES
    )

    if not segmentation_paths:
        raise ValueError(
            "No segmentation masks were found."
        )

    if not roi_paths:
        raise ValueError(
            "No ImageJ ROI files were found."
        )

    roi_by_name = {
        _file_key(path): path
        for path in roi_paths
    }

    rows = []

    for segmentation_path in segmentation_paths:
        name = _file_key(segmentation_path)
        roi_path = roi_by_name.get(name)

        if roi_path is None:
            raise ValueError(
                f"No matching ROI found for "
                f"{segmentation_path.name}. "
                f"Expected {name}.zip or {name}.roi."
            )

        segmentation = np.squeeze(
            imread(segmentation_path)
        )

        if segmentation.ndim != 2:
            raise ValueError(
                f"{segmentation_path.name} is not 2D. "
                f"Shape: {segmentation.shape}."
            )

        # All positive object IDs are foreground.
        prediction = segmentation > 0

        reference = load_imagej_roi_mask(
            roi_path=roi_path,
            image_shape=prediction.shape,
        ).astype(bool)

        rows.append(
            {
                "image": name,
                "iou": _iou(
                    prediction,
                    reference,
                ),
                "segmentation_file": str(
                    segmentation_path
                ),
                "roi_file": str(roi_path),
            }
        )

    table = pd.DataFrame(rows)

    output_csv = Path(output_csv).with_suffix(
        ".csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_csv,
        index=False,
    )

    return table, output_csv