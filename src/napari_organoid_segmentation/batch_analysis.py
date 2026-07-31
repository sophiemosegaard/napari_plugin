from pathlib import Path

import numpy as np
import pandas as pd
from skimage.draw import disk
from skimage.io import imread, imsave
from skimage.measure import label

from .export import save_measurements
from .measurements import measure_organoids
from .preprocessing import as_grayscale
from .segmentation_convpaint import load_convpaint_model, segment_convpaint
from .mosaic import detect_circles


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
WELL_MARGIN_FRACTION = 0.05

def keep_largest_object_per_circle(
    binary_mask: np.ndarray,
    circles: list[tuple[int, int, int, float]],
    well_diameter_px: int,
) -> np.ndarray:
    """Keep one complete segmentation object per detected well."""

    # Label the complete prediction before applying any circle.
    all_objects = label(binary_mask > 0)

    output_labels = np.zeros(
        binary_mask.shape,
        dtype=np.uint32,
    )

    margin_px = max(
        2,
        round(
            well_diameter_px
            * WELL_MARGIN_FRACTION
        ),
    )

    used_object_ids: set[int] = set()
    next_output_id = 1

    for y, x, radius, _score in circles:
        # This circle is used only to select an object.
        selection_radius = radius + margin_px

        rr, cc = disk(
            (y, x),
            selection_radius,
            shape=binary_mask.shape,
        )

        object_ids = all_objects[rr, cc]
        object_ids = object_ids[
            object_ids > 0
        ]

        if object_ids.size == 0:
            continue

        # Count how strongly every complete object overlaps
        # the enlarged well region.
        overlap_areas = np.bincount(object_ids)

        # Do not assign one object to two different wells.
        for object_id in used_object_ids:
            if object_id < len(overlap_areas):
                overlap_areas[object_id] = 0

        selected_object_id = int(
            np.argmax(overlap_areas)
        )

        if overlap_areas[selected_object_id] == 0:
            continue

        # Copy the COMPLETE connected object.
        # It is not cropped by the Hough circle.
        complete_object = (
            all_objects == selected_object_id
        )

        output_labels[
            complete_object
        ] = next_output_id

        used_object_ids.add(
            selected_object_id
        )

        next_output_id += 1

    return output_labels

def analyze_image_folder(
    model_path: Path,
    image_folder: Path,
    output_folder: Path,
    output_filename: str = "organoid_measurements.csv",
    well_diameter_px: int = 280,
    max_wells_per_image: int = 50,
) -> tuple[int, int, Path]:
    """Load a ConvPaint model, segment all images, and save one measurement table."""
    
    if well_diameter_px < 4:
        raise ValueError("well_diameter_px must be at least 4 pixels.")

    if max_wells_per_image < 1:
        raise ValueError("max_wells_per_image must be at least 1.")

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
        
        circles = detect_circles(
            image=image,
            well_diameter_px=well_diameter_px,
            max_circles=max_wells_per_image,
            detection_max_size=1000,
        )

        if circles:
            binary_mask = segment_convpaint(image)

            object_labels = keep_largest_object_per_circle(
                binary_mask=binary_mask,
                circles=circles,
                well_diameter_px=well_diameter_px,
            )
        else:
            # No microwell detected: save an empty segmentation.
            object_labels = np.zeros(image.shape[:2], dtype=np.uint32)

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