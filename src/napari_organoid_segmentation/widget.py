from pathlib import Path
from typing import Optional

import numpy as np
import napari
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info

from .batch_analysis import analyze_image_folder
from .evaluation import calculate_segmentation_folder_iou
from .mosaic import create_organoid_mosaic
from .patches import create_organoid_patch_stack
from .roi_io import image_stack_shape, load_imagej_roi_folder
from .segmentation_convpaint import train_and_save_convpaint


@magic_factory(
    call_button="Create Annotation Layer",
    folder={"label": "Image folder", "mode": "d"},
    annotation_mode={"label": "Annotation mode", "choices": ["Mosaic", "Patch stack"]},
    patches_per_side={"label": "Patches per side", "min": 1},
    number_patches={"label": "Number patches", "min": 1},
    well_diameter_px={"label": "Well diameter (px)", "min": 4},
)
def organoid_workflow_widget(
    folder: Path,
    annotation_mode: str = "Mosaic",
    patches_per_side: int = 4,
    number_patches: int = 16,
    well_diameter_px: int = 280,
) -> list[LayerDataTuple]:
    """Create either a mosaic view or a patch-stack view for annotation."""

    if annotation_mode == "Mosaic":
        mosaic_image, mosaic_labels, detected_organoids = create_organoid_mosaic(
            folder=folder,
            patches_per_side=patches_per_side,
            well_diameter_px=well_diameter_px,
        )

        total_patches = patches_per_side**2
        show_info(
            f"Created mosaic view with {total_patches} tiles and {len(detected_organoids)} detected organoids."
        )

        metadata = {"mode": "mosaic", "tiles": detected_organoids}

        return [
            (
                mosaic_image,
                {
                    "name": "organoid_mosaic",
                    "rgb": True,
                    "metadata": metadata,
                },
                "image",
            ),
            (
                mosaic_labels,
                {
                    "name": "organoid_annotations",
                    "opacity": 0.45,
                    "metadata": metadata,
                },
                "labels",
            ),
        ]

    
    patches, annotations, records = create_organoid_patch_stack(
    folder=folder,
    number_patches=number_patches,
    well_diameter_px=well_diameter_px,
    )
    
    show_info(
    f"Created patch-stack view with {len(patches)} selected wells. "
    "Use the patch slider to annotate them."
    )

    metadata = {"mode": "patch_stack", "patches": records}

    return [
        (
            patches,
            {
                "name": "organoid_patches",
                "rgb": True,
                "metadata": metadata,
                "axis_labels": ("patch", "y", "x"),
            },
            "image",
        ),
        (
            annotations,
            {
                "name": "patch_annotations",
                "opacity": 0.45,
                "metadata": metadata,
                "axis_labels": ("patch", "y", "x"),
            },
            "labels",
        ),
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

    show_info(f"ConvPaint model saved to:\n{model_path}")


@magic_factory(
    call_button="Save annotation project",
    image={"label": "Image layer"},
    annotation={"label": "Annotation layer"},
    project_file={
        "label": "Save project as",
        "mode": "w",
        "filter": "Annotation project (*.npz)",
    },
)
def save_annotation_project_widget(
    image: Image,
    annotation: Labels,
    project_file: Path = Path("organoid_annotation_project.npz"),
) -> None:
    """Save the selected image and Labels layers as one .npz project."""

    # Napari layer selectors can be empty when no matching layer exists.
    if image is None or annotation is None:
        show_info(
            "Choose an Image layer and a Labels layer before saving."
        )
        return

    if project_file is None:
        show_info("Choose a file name and location for the project.")
        return

    image_data = np.asarray(image.data)
    annotation_data = np.asarray(annotation.data)

    image_is_grayscale = image_data.shape == annotation_data.shape
    image_is_rgb = (
        image_data.ndim == annotation_data.ndim + 1
        and image_data.shape[-1] in (3, 4)
        and image_data.shape[:-1] == annotation_data.shape
    )

    if not (image_is_grayscale or image_is_rgb):
        raise ValueError(
            "Image and annotation dimensions do not match. "
            f"Image: {image_data.shape}; "
            f"annotations: {annotation_data.shape}."
        )

    project_path = Path(project_file).with_suffix(".npz")
    project_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        project_path,
        image=image_data,
        annotations=annotation_data,
        image_name=np.asarray(image.name),
        annotation_name=np.asarray(annotation.name),
    )

    show_info(f"Annotation project saved to:\n{project_path}")


@magic_factory(
    call_button="Open annotation project",
    project_file={
        "label": "Project file",
        "mode": "r",
        "filter": "Annotation project (*.npz)",
    },
)
def open_annotation_project_widget(
    project_file: Path = Path("organoid_annotation_project.npz"),
) -> None:
    """Open an .npz project without requiring pre-existing layers."""

    if project_file is None:
        show_info("Choose an .npz annotation project to open.")
        return

    project_path = Path(project_file)

    if not project_path.is_file():
        show_info(f"Project not found: {project_path}")
        return

    with np.load(project_path, allow_pickle=False) as project:
        image_key = "image" if "image" in project else "mosaic"

        if image_key not in project or "annotations" not in project:
            raise ValueError(
                "The selected file is not a valid annotation project. "
                "It must contain 'image' (or legacy 'mosaic') and "
                "'annotations'."
            )

        image_data = np.asarray(project[image_key])
        annotations = np.asarray(project["annotations"])

        image_name = (
            str(project["image_name"].item())
            if "image_name" in project
            else None
        )
        annotation_name = (
            str(project["annotation_name"].item())
            if "annotation_name" in project
            else None
        )

    image_is_grayscale = image_data.shape == annotations.shape
    image_is_rgb = (
        image_data.ndim == annotations.ndim + 1
        and image_data.shape[-1] in (3, 4)
        and image_data.shape[:-1] == annotations.shape
    )

    if not (image_is_grayscale or image_is_rgb):
        raise ValueError(
            "Saved image and annotation dimensions do not match. "
            f"Image: {image_data.shape}; "
            f"annotations: {annotations.shape}."
        )

    viewer = napari.current_viewer()
    if viewer is None:
        show_info("No napari viewer is open.")
        return

    is_patch_stack = annotations.ndim == 3

    if image_name is None:
        image_name = (
            "organoid_patches" if is_patch_stack else "organoid_mosaic"
        )
    if annotation_name is None:
        annotation_name = (
            "patch_annotations"
            if is_patch_stack
            else "organoid_annotations"
        )

    image_options = {
        "name": image_name,
        "rgb": image_is_rgb,
    }
    label_options = {
        "name": annotation_name,
        "opacity": 0.45,
    }

    if is_patch_stack:
        image_options["axis_labels"] = ("patch", "y", "x")
        label_options["axis_labels"] = ("patch", "y", "x")

    viewer.add_image(image_data, **image_options)
    viewer.add_labels(annotations, **label_options)

    show_info(f"Annotation project loaded from:\n{project_path}")


def annotation_project_widget() -> Container:
    """One dock widget containing independent Save and Open controls."""

    return Container(
        widgets=[
            save_annotation_project_widget(),
            open_annotation_project_widget(),
        ]
    )


@magic_factory(
    call_button="Segment folder and save measurements",
    model_path={
        "label": "ConvPaint model",
        "mode": "r",
        "filter": "ConvPaint model (*.pkl)",
    },
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
    call_button="Visualize ROI folder",
    image={"label": "Image stack layer"},
    roi_folder={"label": "ROI folder", "mode": "d"},
)
def visualize_roi_widget(
    image: Image,
    roi_folder: Path = Path("."),
) -> list[LayerDataTuple]:
    """Overlay an ImageJ ROI folder on one selected image stack."""

    if image is None:
        show_info(
            "Choose the image stack that corresponds to the ROI files."
        )
        return []

    number_images, spatial_shape = image_stack_shape(image.data)

    try:
        roi_stack, roi_paths = load_imagej_roi_folder(
            roi_folder=roi_folder,
            number_images=number_images,
            image_shape=spatial_shape,
        )
    except (ValueError, FileNotFoundError) as error:
        show_info(str(error))
        return []

    show_info(
        f"Loaded {len(roi_paths)} ROI files for "
        f"the selected layer '{image.name}'."
    )

    options = {
        "name": "reference_rois",
        "opacity": 0.45,
        "metadata": {
            "roi_files": [str(path) for path in roi_paths],
            "source_image_layer": image.name,
        },
    }

    if roi_stack.ndim == 3:
        options["axis_labels"] = ("image", "y", "x")

    return [(roi_stack, options, "labels")]


@magic_factory(
    call_button="Calculate IoU",
    roi_folder={"label": "ROI folder", "mode": "d"},
    segmentation_folder={
        "label": "Segmentation folder",
        "mode": "d",
    },
    output_csv={
        "label": "IoU results CSV",
        "mode": "w",
        "filter": "CSV file (*.csv)",
    },
)
def calculate_iou_widget(
    roi_folder: Path = Path("."),
    segmentation_folder: Path = Path("."),
    output_csv: Path = Path("segmentation_iou.csv"),
) -> None:
    """Calculate IoU from saved files; no napari image layer is needed."""

    try:
        table, saved_path = calculate_segmentation_folder_iou(
            segmentation_folder=segmentation_folder,
            roi_folder=roi_folder,
            output_csv=output_csv,
        )
    except (ValueError, FileNotFoundError) as error:
        show_info(str(error))
        return

    print(table[["image", "iou"]].to_string(index=False))

    show_info(
        f"Mean IoU: {table['iou'].mean():.4f}\n"
        f"Evaluated images: {len(table)}\n"
        f"CSV saved to:\n{saved_path}"
    )


def roi_iou_widget() -> Container:
    """One dock widget with independent ROI visualization and IoU tools."""

    return Container(
        widgets=[
            visualize_roi_widget(),
            calculate_iou_widget(),
        ]
    )


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
        "IBAN: CH41 0079 042 9430 0433 1\n"
    )


# # Backward-compatible aliases for older command names.
# organoid_mosaic_widget = organoid_workflow_widget
# organoid_patch_stack_widget = organoid_workflow_widget
# save_annotation_project_widget = annotation_project_widget
# load_annotation_project_widget = annotation_project_widget
# load_imagej_roi_widget = roi_iou_widget
# segmentation_iou_widget = roi_iou_widget
