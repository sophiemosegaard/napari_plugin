from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.measure import label

from .export import save_measurements
from .measurements import measure_organoids
from .preprocessing import as_grayscale
from .segmentation_convpaint import load_convpaint_model, segment_convpaint


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

def label_organoids(
    mask: np.ndarray,
    min_area_px: int,
) -> np.ndarray:
    """Label all sufficiently large foreground objects."""

    # Give every connected foreground region an ID.
    labels = label(
        np.asarray(mask, dtype=bool),
        connectivity=2,
    )

    if labels.max() == 0:
        return np.zeros(
            mask.shape,
            dtype=np.uint32,
        )

    # Number of pixels belonging to each label.
    areas = np.bincount(labels.ravel())

    # Keep objects larger than the minimum area.
    keep = np.flatnonzero(
        areas >= min_area_px
    )

    # Label 0 is background and must not be retained.
    keep = keep[keep != 0]

    # Relabel retained organoids as 1, 2, 3, ...
    mapping = np.zeros(
        len(areas),
        dtype=np.uint32,
    )

    mapping[keep] = np.arange(
        1,
        len(keep) + 1,
        dtype=np.uint32,
    )

    return mapping[labels]

def analyze_image_folder(
    model_path: Path,
    image_folder: Path,
    output_folder: Path,
    output_filename: str = "organoid_measurements.csv",
    # well_diameter_px: int = 280,
    # include_border_wells: bool = False,
    min_area_px: int = 5000,
) -> tuple[int, int, Path]:
    """Load a ConvPaint model, segment all images, and save one measurement table."""

    load_convpaint_model(model_path)

    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv = output_folder / Path(output_filename).name

    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv.with_suffix(".csv")

    mask_folder = output_csv.parent / f"{output_csv.stem}_segmentations"

    if not image_folder.is_dir():
        raise ValueError(
            f"Not a valid image folder: {image_folder}"
        )

    mask_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in image_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and mask_folder not in path.parents
    )

    if not image_paths:
        raise ValueError(f"No supported images found in: {image_folder}")

    tables = []

    for image_path in image_paths:
        image = imread(image_path)
        
        binary_mask = segment_convpaint(image)

        object_labels = label_organoids(
            mask=binary_mask,
            min_area_px=min_area_px,
        )

        relative_path = image_path.relative_to(image_folder)

        mask_relative_path = relative_path.with_name(f"{relative_path.stem}_labels.tif")

        mask_path = mask_folder / mask_relative_path
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(mask_path, object_labels, check_contrast=False)

        # No measurement rows when no organoid was found.
        if object_labels.max() == 0:
            continue

        table = measure_organoids(
            labels=object_labels,
            intensity_image=as_grayscale(image),
            image_name=str(relative_path),
            segmentation_method="ConvPaint",
        )

        table["segmentation_file"] = str(mask_path)

        tables.append(table)

    if tables:
        combined_table = pd.concat(
            tables,
            ignore_index=True,
        )
    else:
        combined_table = pd.DataFrame(
            columns=[
                "image_name",
                "segmentation_method",
                "object_id",
                "centroid_y_px",
                "centroid_x_px",
                "area_px",
                "perimeter_px",
                "mean_intensity",
                "equivalent_radius_px",
                "circularity",
                "segmentation_file",
            ]
        )

    save_measurements(combined_table, output_csv)

    return (len(image_paths), len(combined_table), mask_folder)