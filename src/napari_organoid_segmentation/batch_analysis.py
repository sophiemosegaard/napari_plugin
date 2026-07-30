from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.measure import label

from .export import save_measurements
from .measurements import measure_organoids
from .preprocessing import as_grayscale
from .segmentation_convpaint import segment_convpaint


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def analyze_image_folder(
    image_folder: Path,
    output_csv: Path,
) -> tuple[int, int, Path]:
    """Segment all images and save one measurement table."""

    image_folder = Path(image_folder)
    output_csv = Path(output_csv).with_suffix(".csv")

    if not image_folder.is_dir():
        raise ValueError(
            f"Not a valid image folder: {image_folder}"
        )

    # Segmentations are saved automatically beside the CSV.
    mask_folder = (output_csv.parent / f"{output_csv.stem}_segmentations")

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

        # Binary ConvPaint prediction:
        # 0 = background, 1 = organoid.
        binary_mask = segment_convpaint(image)

        # Give each connected organoid its own object ID.
        object_labels = label(binary_mask > 0).astype(np.uint32)

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