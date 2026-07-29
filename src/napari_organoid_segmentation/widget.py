from pathlib import Path

import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info

from .preprocessing import as_grayscale
from .segmentation_convpaint import (load_convpaint_model, segment_convpaint, train_and_save_convpaint)
from .export import save_measurements
from .measurements import measure_organoids
from .mosaic import create_organoid_mosaic


@magic_factory(
    call_button="Create Organoid Mosaic",
    folder={"label": "Image folder", "mode": "d"},
    rows={"min": 1},
    columns={"min": 1},
    min_radius_px={"min": 2},
    max_radius_px={"min": 3},  
)
def organoid_mosaic_widget(
    folder: Path,
    rows: int = 16,
    columns: int = 16,
    min_radius_px: int = 140,
    max_radius_px: int = 180,
) -> list[LayerDataTuple]:
    """Detect organoids, choose them randomly, and build image/label mosaics."""
    mosaic_image, mosaic_labels, detected_organoids = create_organoid_mosaic(
        folder=folder,
        rows=rows,
        columns=columns,
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
    )

    show_info(f"Created organoid mosaic {rows} x {columns} with {len(detected_organoids)} detected organoids.")
    metadata = {"tiles": detected_organoids}

    return [
        (mosaic_image, {"name": "organoid_mosaic", "rgb": True, "metadata": metadata}, "image"),
        (mosaic_labels, {"name": "organoid_annotations", "opacity": 0.45, "metadata": metadata}, "labels"),
    ]


@magic_factory(
    call_button="Train and save model",
    model_folder={"label": "Model folder", "mode": "d"},
    )
def train_convpaint_widget(
    image: Image,
    annotation: Labels,
    model_folder: Path = Path("models"),
    model_name: str = "organoid_convpaint",
) -> None:
    model_path = train_and_save_convpaint(
        image=np.asarray(image.data),
        annotations=np.asarray(annotation.data),
        model_folder=model_folder,
        model_name=model_name,
    )

    show_info(
        f"ConvPaint model saved to:\n{model_path}"
    )
    
@magic_factory(
    call_button="Load model",
    model_path={"label": "ConvPaint model", "mode": "r", "filter": "ConvPaint model (*.pkl)"},
    )
def load_convpaint_widget(
    model_path: Path = Path(
        "organoid_convpaint.pkl"
    ),
) -> None:
    loaded_path = load_convpaint_model(
        model_path
    )

    show_info(
        f"Loaded ConvPaint model:\n{loaded_path}"
    )


@magic_factory(call_button="Segment organoids")
def organoid_analysis_widget(
    image: Image,
) -> LayerDataTuple:
    mask = segment_convpaint(
        np.asarray(image.data)
    )

    return (mask, {"name": f"{image.name}_organoids"}, "labels")
    
@magic_factory(
    call_button="Save measurements CSV",
    output_path={
        "label": "CSV output file", 
        "mode": "w", 
        "filter": "CSV files (*.csv)"
    },
    segmentation_method={"label": "Segmentation method"},
)
def export_measurements_widget(
    image: Image,
    labels: Labels,
    output_path: Path = Path("../reports/organoid_measurements.csv"),
    segmentation_method: Literal[
        "Otsu + morphology", "ConvPaint",
    ] = "Otsu + morphology",
) -> None:
    """Measure segmented organoids and save the results as CSV."""

    # Convert RGB images to grayscale for intensity measurements.
    grayscale = as_grayscale(np.asarray(image.data))
    output_path = Path(output_path)
    # Create the reports folder if it does not already exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add .csv if the user did not enter an extension.
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")

    table = measure_organoids(
        labels=np.asarray(labels.data),
        intensity_image=grayscale,
        image_name=image.name,
        segmentation_method=segmentation_method,
    )
    save_measurements(table, output_path)
    show_info(f"Saved measurements for {len(table)} organoids to:\n{output_path}")