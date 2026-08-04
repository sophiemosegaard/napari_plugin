from pathlib import Path

import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info

from .batch_analysis import analyze_image_folder
from .mosaic import create_organoid_mosaic
from .segmentation_convpaint import train_and_save_convpaint
from .patches import create_organoid_patch_stack
from .roi_io import image_stack_shape, load_imagej_roi_folder
from .evaluation import calculate_segmentation_folder_iou

@magic_factory(
    call_button="Create Organoid Mosaic",
    folder={"label": "Image folder", "mode": "d"},
    patches_per_side={"label": "Patches per side", "min": 1},
    well_diameter_px={"label": "Well diameter (px)", "min": 4},  
)
def organoid_mosaic_widget(
    folder: Path,
    patches_per_side: int = 4,
    well_diameter_px: int = 280,
) -> list[LayerDataTuple]:
    """Detect organoids, choose them randomly, and build image/label mosaics."""
    mosaic_image, mosaic_labels, detected_organoids = create_organoid_mosaic(
        folder=folder,
        patches_per_side=patches_per_side,
        well_diameter_px=well_diameter_px,
    )
    
    total_patches = patches_per_side**2

    show_info(f"Created organoid mosaic {patches_per_side} x {patches_per_side} with {total_patches} detected organoids.")
    metadata = {"tiles": detected_organoids}

    return [
        (mosaic_image, {"name": "organoid_mosaic", "rgb": True, "metadata": metadata}, "image"),
        (mosaic_labels, {"name": "organoid_annotations", "opacity": 0.45, "metadata": metadata}, "labels"),
    ]
    

@magic_factory(
    call_button="Create Individual Well Patches",
    folder={
        "label": "Image folder",
        "mode": "d",
    },
    patches_per_side={
        "label": "Patches per side",
        "min": 1,
    },
    well_diameter_px={
        "label": "Well diameter (px)",
        "min": 4,
    },
)
def organoid_patch_stack_widget(
    folder: Path,
    patches_per_side: int = 4,
    well_diameter_px: int = 280,
) -> list[LayerDataTuple]:
    """Show selected wells as separate napari slices."""

    patches, annotations, records = (
        create_organoid_patch_stack(
            folder=folder,
            patches_per_side=patches_per_side,
            well_diameter_px=well_diameter_px,
        )
    )

    metadata = {
        "patches": records,
    }

    show_info(
        f"Created {len(patches)} individual well patches. "
        "Use the patch slider to annotate them."
    )

    return [
        (patches, {"name": "organoid_patches", "rgb": True, "metadata": metadata, "axis_labels": ("patch", "y", "x")}, "image"),
        (annotations, {"name": "patch_annotations", "opacity": 0.45, "metadata": metadata, "axis_labels": ("patch", "y", "x",)}, "labels"),
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
    output_folder={"label": "Measurements folder", "mode": "d"},
    output_filename={"label": "Measurements CSV name"},
)
def batch_analysis_widget(
    model_path: Path = Path("organoid_convpaint.pkl"),
    image_folder: Path = Path("."),
    output_folder: Path = Path("reports"),
    output_filename: str = "organoid_measurements.csv",
) -> None:
    """Segment a folder and save the measurements and object masks."""

    number_images, number_organoids, mask_folder = analyze_image_folder(
        model_path=model_path,
        image_folder=image_folder,
        min_area_px=5000,
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
    call_button="Load ROI folder",
    roi_folder={
        "label": "ROI folder",
        "mode": "d",
    },
)
def load_imagej_roi_widget(
    image: Image,
    roi_folder: Path,
) -> list[LayerDataTuple]:
    """Load one ImageJ ROI ZIP per image slice."""

    number_images, spatial_shape = (
        image_stack_shape(image.data)
    )

    roi_stack, roi_paths = load_imagej_roi_folder(
        roi_folder=roi_folder,
        number_images=number_images,
        image_shape=spatial_shape,
    )

    show_info(
        f"Loaded {len(roi_paths)} ROI files."
    )

    options = {
        "name": "reference_rois",
        "opacity": 0.45,
        "metadata": {
            "roi_files": [
                str(path)
                for path in roi_paths
            ],
        },
    }

    if roi_stack.ndim == 3:
        options["axis_labels"] = (
            "image",
            "y",
            "x",
        )

    return [
        (
            roi_stack,
            options,
            "labels",
        )
    ] 
    

@magic_factory(
    call_button="Calculate Saved Segmentation IoU",
    segmentation_folder={
        "label": "Segmentation folder",
        "mode": "d",
    },
    roi_folder={
        "label": "ROI folder",
        "mode": "d",
    },
    output_csv={
        "label": "IoU results CSV",
        "mode": "w",
        "filter": "CSV file (*.csv)",
    },
)
def segmentation_iou_widget(
    segmentation_folder: Path,
    roi_folder: Path,
    output_csv: Path = Path(
        "segmentation_iou.csv"
    ),
) -> None:
    """Compare saved segmentations with ROI files."""

    table, saved_path = (
        calculate_segmentation_folder_iou(
            segmentation_folder=(
                segmentation_folder
            ),
            roi_folder=roi_folder,
            output_csv=output_csv,
        )
    )

    # Show individual values in the terminal.
    print(
        table[
            ["image", "iou"]
        ].to_string(index=False)
    )

    show_info(
        f"Mean IoU: "
        f"{table['iou'].mean():.4f}\n"
        f"Evaluated images: {len(table)}\n"
        f"CSV saved to:\n{saved_path}"
    )

@magic_factory(
    call_button="Save annotated mosaic",
    output_path={
        "label": "Project file",
        "mode": "w",
        "filter": "Annotation project (*.npz)",
    },
)
def save_annotation_project_widget(
    image: Image,
    annotation: Labels,
    output_path: Path = Path("organoid_annotations.npz"),
) -> None:
    """Save either a mosaic or a stack of annotated patches."""

    image_data = np.asarray(image.data)
    annotation_data = np.asarray(annotation.data)

    # Mosaic:
    # image       = (Y, X, 3)
    # annotations = (Y, X)
    #
    # Patch stack:
    # image       = (N, Y, X, 3)
    # annotations = (N, Y, X)
    if image_data.shape[:-1] != annotation_data.shape:
        raise ValueError(
            "Image and annotation dimensions do not match. "
            f"Image: {image_data.shape}; "
            f"annotations: {annotation_data.shape}."
        )

    output_path = Path(output_path).with_suffix(".npz")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        image=image_data,
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
    """Load either a mosaic or a stack of annotated patches."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Project not found: {input_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as project:
        # New projects use "image".
        # Old mosaic projects used "mosaic".
        image_key = (
            "image"
            if "image" in project
            else "mosaic"
        )

        image = project[image_key]
        annotations = project["annotations"]

    if image.shape[:-1] != annotations.shape:
        raise ValueError(
            "Saved image and annotation dimensions "
            "do not match."
        )

    # Three-dimensional annotations mean:
    # (patch, y, x)
    if annotations.ndim == 3:
        image_options = {
            "name": "organoid_patches",
            "rgb": True,
            "axis_labels": (
                "patch",
                "y",
                "x",
            ),
        }

        label_options = {
            "name": "patch_annotations",
            "opacity": 0.45,
            "axis_labels": (
                "patch",
                "y",
                "x",
            ),
        }

    # Two-dimensional annotations mean:
    # (y, x), so this is a mosaic.
    else:
        image_options = {
            "name": "organoid_mosaic",
            "rgb": True,
        }

        label_options = {
            "name": "organoid_annotations",
            "opacity": 0.45,
        }

    return [
        (
            image,
            image_options,
            "image",
        ),
        (
            annotations.astype(np.uint8),
            label_options,
            "labels",
        ),
    ]
# def save_annotation_project_widget(
#     mosaic: Image,
#     annotation: Labels,
#     output_path: Path = Path("organoid_annotations.npz"),
# ) -> None:
#     """Save the mosaic and edited annotations together."""

#     mosaic_data = np.asarray(mosaic.data)
#     annotation_data = np.asarray(annotation.data)

#     if mosaic_data.shape[:2] != annotation_data.shape:
#         raise ValueError(
#             "Mosaic and annotation dimensions do not match."
#         )

#     output_path = Path(output_path).with_suffix(".npz")
#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     np.savez_compressed(
#         output_path,
#         mosaic=mosaic_data,
#         annotations=annotation_data.astype(np.uint8),
#     )

#     show_info(
#         f"Annotation project saved to:\n{output_path}"
#     )
    
    
# @magic_factory(
#     call_button="Load annotated mosaic",
#     input_path={
#         "label": "Project file",
#         "mode": "r",
#         "filter": "Annotation project (*.npz)",
#     },
# )
# def load_annotation_project_widget(
#     input_path: Path = Path("organoid_annotations.npz"),
# ) -> list[LayerDataTuple]:
#     """Load a saved mosaic and annotation layer."""

#     input_path = Path(input_path)

#     if not input_path.is_file():
#         raise FileNotFoundError(
#             f"Project not found: {input_path}"
#         )

#     with np.load(input_path) as project:
#         mosaic = project["mosaic"]
#         annotations = project["annotations"]

#     return [
#         (mosaic, {"name": "organoid_mosaic", "rgb": True}, "image"),
#         (annotations.astype(np.uint8), {"name": "organoid_annotations", "opacity": 0.45}, "labels"),
#     ]

    

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
        "IBAN: CH41 0079 0042 9430 0433 1\n"
    )