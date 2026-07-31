from pathlib import Path

import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info

from .batch_analysis import analyze_image_folder
from .mosaic import create_organoid_mosaic
from .segmentation_convpaint import train_and_save_convpaint


@magic_factory(
    call_button="Create Organoid Mosaic",
    folder={"label": "Image folder", "mode": "d"},
    rows={"min": 1},
    columns={"min": 1},
    well_diameter_px={"label": "Well diameter (px)", "min": 4},  
)
def organoid_mosaic_widget(
    folder: Path,
    rows: int = 16,
    columns: int = 16,
    well_diameter_px: int = 280,
) -> list[LayerDataTuple]:
    """Detect organoids, choose them randomly, and build image/label mosaics."""
    mosaic_image, mosaic_labels, detected_organoids = create_organoid_mosaic(
        folder=folder,
        rows=rows,
        columns=columns,
        well_diameter_px=well_diameter_px,
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
    call_button="Segment folder and save measurements",
    model_path={"label": "ConvPaint model", "mode": "r", "filter": "ConvPaint model (*.pkl)"},
    image_folder={"label": "Image folder", "mode": "d"},
    well_diameter_px={"label": "Well diameter (px)", "min": 4},
    max_wells_per_image={"label": "Max wells per image", "min": 1},
    output_folder={"label": "Measurements folder", "mode": "d"},
    output_filename={"label": "Measurements CSV name"},
)
def batch_analysis_widget(
    model_path: Path = Path("organoid_convpaint.pkl"),
    image_folder: Path = Path("."),
    well_diameter_px: int = 280,
    max_wells_per_image: int = 50,
    output_folder: Path = Path("reports"),
    output_filename: str = "organoid_measurements.csv",
) -> None:
    """Segment a folder and save the measurements and object masks."""

    number_images, number_organoids, mask_folder = analyze_image_folder(
        model_path=model_path,
        image_folder=image_folder,
        well_diameter_px=well_diameter_px,
        max_wells_per_image=max_wells_per_image,
        output_folder=output_folder,
        output_filename=output_filename,
    )

    output_csv = Path(output_folder) / Path(output_filename).name
    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv.with_suffix(".csv")

    show_info(
        f"Processed {number_images} images.\n"
        f"Measured {number_organoids} organoids.\n"
        f"CSV saved to:\n{output_csv}\n"
        f"Segmentations saved to:\n{mask_folder}"
    )

@magic_factory(
    call_button="Save annotation project",
    output_path={
        "label": "Project file",
        "mode": "w",
        "filter": "Annotation project (*.npz)",
    },
)
def save_annotation_project_widget(
    mosaic: Image,
    annotation: Labels,
    output_path: Path = Path("organoid_annotations.npz"),
) -> None:
    """Save the mosaic and edited annotations together."""

    mosaic_data = np.asarray(mosaic.data)
    annotation_data = np.asarray(annotation.data)

    if mosaic_data.shape[:2] != annotation_data.shape:
        raise ValueError(
            "Mosaic and annotation dimensions do not match."
        )

    output_path = Path(output_path).with_suffix(".npz")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        mosaic=mosaic_data,
        annotations=annotation_data.astype(np.uint8),
    )

    show_info(
        f"Annotation project saved to:\n{output_path}"
    )
    
    
@magic_factory(
    call_button="Load annotation project",
    input_path={
        "label": "Project file",
        "mode": "r",
        "filter": "Annotation project (*.npz)",
    },
)
def load_annotation_project_widget(
    input_path: Path = Path("organoid_annotations.npz"),
) -> list[LayerDataTuple]:
    """Load a saved mosaic and annotation layer."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Project not found: {input_path}"
        )

    with np.load(input_path) as project:
        mosaic = project["mosaic"]
        annotations = project["annotations"]

    return [
        (mosaic, {"name": "organoid_mosaic", "rgb": True}, "image"),
        (annotations.astype(np.uint8), {"name": "organoid_annotations", "opacity": 0.45}, "labels"),
    ]
    

@magic_factory(
    call_button="Show credits",
)
def about_plugin_widget() -> None:
    """Show plugin author and optional support information."""

    show_info(
        "Organoid Segmentation\n"
        "Created by Sophie Mosegaard\n\n"
        "This plugin runs on Python; "
        "its student developer runs on coffee. ☕\n\n"
        "Optional coffee support:\n"
        "<YOUR_DONATION_LINK>"
    )