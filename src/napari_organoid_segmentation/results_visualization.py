from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread

from .batch_analysis import IMAGE_SUFFIXES


DEFAULT_MEASUREMENT_FEATURES = (
    "area_px",
    "perimeter_px",
    "mean_intensity",
    "equivalent_radius_px",
    "circularity",
)


def measurement_feature_columns(table: pd.DataFrame) -> tuple[str, ...]:
    """Return numeric columns that make sense as object color features."""

    excluded = {
        "object_id",
        "centroid_y_px",
        "centroid_x_px",
    }

    numeric = tuple(
        column
        for column in table.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(table[column])
    )

    if numeric:
        return numeric

    # Empty CSV files may not preserve numeric dtypes when read by pandas.
    return tuple(
        column
        for column in DEFAULT_MEASUREMENT_FEATURES
        if column in table.columns
    )


def read_measurements_csv(csv_path: Path) -> pd.DataFrame:
    """Read and validate a measurement CSV created by this plugin."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    table = pd.read_csv(csv_path)

    required = {
        "image_name",
        "object_id",
        "segmentation_file",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            "The measurement CSV is missing required columns: "
            + ", ".join(missing)
        )

    return table


def _remove_named_layer(viewer, name: str) -> None:
    """Remove an earlier result layer with the same name."""

    for layer in list(viewer.layers):
        if layer.name == name:
            viewer.layers.remove(layer)


def _image_paths(
    image_folder: Path,
    mask_folder: Path | None = None,
) -> list[Path]:
    image_folder = Path(image_folder)
    if not image_folder.is_dir():
        raise ValueError(f"Not a valid image folder: {image_folder}")

    mask_folder = Path(mask_folder) if mask_folder is not None else None

    paths = sorted(
        path
        for path in image_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and (
            mask_folder is None
            or mask_folder not in path.parents
        )
    )

    if not paths:
        raise ValueError(
            f"No supported images found in: {image_folder}"
        )

    return paths


def _mask_path_for_image(
    image_path: Path,
    image_folder: Path,
    mask_folder: Path,
) -> Path:
    relative_path = image_path.relative_to(image_folder)
    mask_relative_path = relative_path.with_name(
        f"{relative_path.stem}_labels.tif"
    )
    return mask_folder / mask_relative_path


def _all_same_shape(arrays: list[np.ndarray]) -> bool:
    return bool(arrays) and all(
        array.shape == arrays[0].shape for array in arrays
    )


def _is_rgb_image(array: np.ndarray) -> bool:
    return array.ndim == 3 and array.shape[-1] in (3, 4)


def display_segmentation_results(
    viewer,
    image_folder: Path,
    mask_folder: Path,
) -> tuple[list[Path], list[Path]]:
    """Display source images and saved instance masks in napari.

    Equal-sized inputs are displayed as one browsable stack. Inputs with
    different shapes are displayed as one image/mask layer pair per file.
    """

    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)

    if not mask_folder.is_dir():
        raise ValueError(
            f"Not a valid segmentation folder: {mask_folder}"
        )

    image_paths = _image_paths(image_folder, mask_folder)
    mask_paths = [
        _mask_path_for_image(
            image_path=image_path,
            image_folder=image_folder,
            mask_folder=mask_folder,
        )
        for image_path in image_paths
    ]

    missing_masks = [path for path in mask_paths if not path.is_file()]
    if missing_masks:
        raise FileNotFoundError(
            "No saved segmentation was found for: "
            + ", ".join(path.name for path in missing_masks[:5])
        )

    images = [np.asarray(imread(path)) for path in image_paths]
    masks = [np.asarray(imread(path), dtype=np.uint32) for path in mask_paths]

    for image_path, image, mask in zip(
        image_paths,
        images,
        masks,
    ):
        expected_shape = (
            image.shape[:2] if _is_rgb_image(image) else image.shape
        )
        if mask.shape != expected_shape:
            raise ValueError(
                f"Image and segmentation shapes do not match for "
                f"{image_path.name}: image {image.shape}, "
                f"segmentation {mask.shape}."
            )

    can_stack = (
        _all_same_shape(images)
        and _all_same_shape(masks)
        and all(_is_rgb_image(image) for image in images)
        or (
            _all_same_shape(images)
            and _all_same_shape(masks)
            and all(image.ndim == 2 for image in images)
        )
    )

    if can_stack:
        image_data = (
            images[0]
            if len(images) == 1
            else np.stack(images)
        )
        mask_data = (
            masks[0]
            if len(masks) == 1
            else np.stack(masks)
        )
        rgb = _is_rgb_image(images[0])

        _remove_named_layer(viewer, "batch_images")
        _remove_named_layer(viewer, "batch_segmentations")

        image_options = {
            "name": "batch_images",
            "rgb": rgb,
            "metadata": {
                "source_files": [str(path) for path in image_paths],
            },
        }
        label_options = {
            "name": "batch_segmentations",
            "opacity": 0.55,
            "metadata": {
                "segmentation_files": [
                    str(path) for path in mask_paths
                ],
            },
        }

        if len(images) > 1:
            image_options["axis_labels"] = ("image", "y", "x")
            label_options["axis_labels"] = ("image", "y", "x")

        viewer.add_image(image_data, **image_options)
        viewer.add_labels(mask_data, **label_options)

    else:
        for index, (image_path, mask_path, image, mask) in enumerate(
            zip(image_paths, mask_paths, images, masks),
            start=1,
        ):
            base_name = image_path.stem
            image_name = f"{index:03d}_{base_name}"
            mask_name = f"{index:03d}_{base_name}_segmentation"

            _remove_named_layer(viewer, image_name)
            _remove_named_layer(viewer, mask_name)

            viewer.add_image(
                image,
                name=image_name,
                rgb=_is_rgb_image(image),
                metadata={"source_file": str(image_path)},
            )
            viewer.add_labels(
                mask,
                name=mask_name,
                opacity=0.55,
                metadata={"segmentation_file": str(mask_path)},
            )

    return image_paths, mask_paths


def _normalise_relative_path(value: object) -> Path:
    # CSV files written on Windows contain backslashes. Replacing them first
    # also makes the path usable when the project is later opened elsewhere.
    return Path(str(value).replace("\\", "/"))


def _resolve_segmentation_path(
    group: pd.DataFrame,
    segmentation_folder: Path,
) -> Path:
    recorded = Path(str(group["segmentation_file"].iloc[0]))
    if recorded.is_file():
        return recorded

    image_relative_path = _normalise_relative_path(
        group["image_name"].iloc[0]
    )
    mask_relative_path = image_relative_path.with_name(
        f"{image_relative_path.stem}_labels.tif"
    )
    fallback = segmentation_folder / mask_relative_path

    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        "Could not find the segmentation for "
        f"{group['image_name'].iloc[0]}. Expected either "
        f"{recorded} or {fallback}."
    )


def _safe_contrast_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        raise ValueError(
            "The selected measurement contains no finite values."
        )

    minimum = float(finite.min())
    maximum = float(finite.max())

    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 0.5)
        return minimum - padding, maximum + padding

    return minimum, maximum


def display_measurement_map(
    viewer,
    table: pd.DataFrame,
    segmentation_folder: Path,
    feature: str,
    colormap: str = "turbo",
):
    """Color each segmented object by one numeric measurement.

    A floating-point Image layer is used rather than a Labels layer because
    continuous measurements need a continuous colormap and calibrated
    contrast limits. Background pixels are NaN and therefore transparent.
    """

    segmentation_folder = Path(segmentation_folder)
    if not segmentation_folder.is_dir():
        raise ValueError(
            f"Not a valid segmentation folder: {segmentation_folder}"
        )

    if feature not in table.columns:
        raise ValueError(
            f"Measurement column not found: {feature}"
        )

    if not pd.api.types.is_numeric_dtype(table[feature]):
        raise ValueError(
            f"Measurement column is not numeric: {feature}"
        )

    contrast_limits = _safe_contrast_limits(
        table[feature].to_numpy()
    )

    maps: list[np.ndarray] = []
    image_names: list[str] = []
    segmentation_paths: list[Path] = []

    for image_name, group in table.groupby(
        "image_name",
        sort=False,
    ):
        segmentation_path = _resolve_segmentation_path(
            group=group,
            segmentation_folder=segmentation_folder,
        )
        labels = np.asarray(
            imread(segmentation_path),
            dtype=np.int64,
        )

        if labels.ndim != 2:
            raise ValueError(
                f"Measurement maps require 2D masks, but "
                f"{segmentation_path.name} has shape {labels.shape}."
            )

        lookup = np.full(
            int(labels.max()) + 1,
            np.nan,
            dtype=np.float32,
        )

        object_ids = pd.to_numeric(
            group["object_id"],
            errors="coerce",
        ).to_numpy()
        feature_values = pd.to_numeric(
            group[feature],
            errors="coerce",
        ).to_numpy(dtype=np.float32)

        valid = (
            np.isfinite(object_ids)
            & np.isfinite(feature_values)
            & (object_ids >= 1)
            & (object_ids < len(lookup))
        )

        lookup[object_ids[valid].astype(np.int64)] = (
            feature_values[valid]
        )

        maps.append(lookup[labels])
        image_names.append(str(image_name))
        segmentation_paths.append(segmentation_path)

    if not maps:
        raise ValueError(
            "No measurement maps could be created from the CSV."
        )

    layer_name = f"measurement_{feature}"

    if _all_same_shape(maps):
        map_data = maps[0] if len(maps) == 1 else np.stack(maps)
        _remove_named_layer(viewer, layer_name)

        options = {
            "name": layer_name,
            "colormap": colormap,
            "contrast_limits": contrast_limits,
            "opacity": 0.8,
            "blending": "translucent",
            "interpolation2d": "nearest",
            "metadata": {
                "feature": feature,
                "image_names": image_names,
                "segmentation_files": [
                    str(path) for path in segmentation_paths
                ],
                "value_range": contrast_limits,
            },
        }

        if len(maps) > 1:
            options["axis_labels"] = ("image", "y", "x")

        layer = viewer.add_image(map_data, **options)
        layer.colorbar.visible = True
        return layer

    # Different image sizes cannot form one ndarray stack. Add one feature
    # map per source image, but keep one global range for comparable colors.
    first_layer = None
    for index, (feature_map, image_name, segmentation_path) in enumerate(
        zip(maps, image_names, segmentation_paths),
        start=1,
    ):
        individual_name = (
            f"{index:03d}_{Path(image_name).stem}_{feature}"
        )
        _remove_named_layer(viewer, individual_name)
        layer = viewer.add_image(
            feature_map,
            name=individual_name,
            colormap=colormap,
            contrast_limits=contrast_limits,
            opacity=0.8,
            blending="translucent",
            interpolation2d="nearest",
            metadata={
                "feature": feature,
                "image_name": image_name,
                "segmentation_file": str(segmentation_path),
                "value_range": contrast_limits,
            },
        )
        layer.colorbar.visible = True
        if first_layer is None:
            first_layer = layer

    return first_layer