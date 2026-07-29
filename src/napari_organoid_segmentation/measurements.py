import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def measure_organoids(
    labels: np.ndarray,
    intensity_image: np.ndarray,
    image_name: str,
    segmentation_method: str,
) -> pd.DataFrame:
    """Measure every individually labeled organoid.

    Parameters
    ----------
    labels:
        Integer label image. Background must be 0.
        Each organoid must have its own positive integer ID.
    intensity_image:
        Two-dimensional grayscale source image.
    image_name:
        Name of the original napari image layer.
    segmentation_method:
        Name of the segmentation method used.

    Returns
    -------
    pandas.DataFrame
        One row per organoid.
    """

    label_data = np.asarray(labels)
    image_data = np.asarray(intensity_image)

    if label_data.ndim != 2:
        raise ValueError(
            "The segmentation must be a two-dimensional Labels layer. "
            f"Received shape {label_data.shape}."
        )

    if image_data.ndim != 2:
        raise ValueError(
            "The intensity image must be two-dimensional. "
            f"Received shape {image_data.shape}."
        )

    if label_data.shape != image_data.shape:
        raise ValueError(
            "Image and segmentation dimensions do not match. "
            f"Image: {image_data.shape}; labels: {label_data.shape}."
        )

    if label_data.max() == 0:
        raise ValueError(
            "The selected Labels layer does not contain any organoids."
        )

    properties = regionprops_table(
        label_data.astype(np.int32),
        intensity_image=image_data,
        properties=(
            "label",
            "centroid",
            "area",
            "perimeter",
            "intensity_mean",
        ),
    )

    table = pd.DataFrame(properties)

    table = table.rename(
        columns={
            "label": "object_id",
            "centroid-0": "centroid_y_px",
            "centroid-1": "centroid_x_px",
            "area": "area_px",
            "perimeter": "perimeter_px",
            "intensity_mean": "mean_intensity",
        }
    )

    # Radius of a circle having the same area as the segmented object.
    table["equivalent_radius_px"] = np.sqrt(
        table["area_px"] / np.pi
    )

    # Circularity is approximately 1 for a perfect circle.
    valid_perimeter = table["perimeter_px"] > 0

    table["circularity"] = np.nan

    table.loc[valid_perimeter, "circularity"] = (
        4
        * np.pi
        * table.loc[valid_perimeter, "area_px"]
        / table.loc[valid_perimeter, "perimeter_px"] ** 2
    )

    # Add metadata columns at the beginning.
    table.insert(0, "segmentation_method", segmentation_method)
    table.insert(0, "image_name", image_name)

    return table