from pathlib import Path
from typing import Optional

import numpy as np
import napari
from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, FileEdit, Label, LineEdit, PushButton, SpinBox, Table
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from napari.utils.notifications import show_info

from .batch_analysis import analyze_image_folder
from .evaluation import calculate_segmentation_folder_iou
from .mosaic import create_organoid_mosaic
from .patches import create_organoid_patch_stack
from .roi_io import image_stack_shape, load_imagej_roi_folder
from .segmentation_convpaint import train_and_save_convpaint
from .results_visualization import DEFAULT_MEASUREMENT_FEATURES, display_measurement_map, display_segmentation_results, measurement_feature_columns, read_measurements_csv


def organoid_workflow_widget() -> Container:
    """Create mosaic or patch-stack annotation layers.

    The image folder and well diameter are shared. Each output mode has
    its own dedicated parameter and button.
    """

    # Shared settings.
    folder_widget = FileEdit(
        value=Path("."),
        mode="d",
        label="Image folder",
    )
    well_diameter_widget = SpinBox(
        value=280,
        min=4,
        label="Well diameter (px)",
    )

    # Mosaic-only settings.
    patches_per_side_widget = SpinBox(
        value=4,
        min=1,
        label="Patches per side",
    )
    create_mosaic_button = PushButton(
        text="Create mosaic annotation layer"
    )

    # Patch-stack-only settings.
    number_patches_widget = SpinBox(
        value=16,
        min=1,
        label="Number patches",
    )
    create_patch_stack_button = PushButton(
        text="Create patch-stack annotation layer"
    )

    def _viewer_and_folder():
        viewer = napari.current_viewer()

        if viewer is None:
            show_info("No napari viewer is open.")
            return None, None

        folder = folder_widget.value

        if folder is None or not Path(folder).is_dir():
            show_info("Choose a valid image folder.")
            return None, None

        return viewer, Path(folder)

    @create_mosaic_button.clicked.connect
    def _create_mosaic() -> None:
        viewer, folder = _viewer_and_folder()

        if viewer is None:
            return

        try:
            mosaic_image, mosaic_labels, detected_organoids = (
                create_organoid_mosaic(
                    folder=folder,
                    patches_per_side=patches_per_side_widget.value,
                    well_diameter_px=well_diameter_widget.value,
                )
            )
        except (ValueError, FileNotFoundError) as error:
            show_info(str(error))
            return

        metadata = {
            "mode": "mosaic",
            "tiles": detected_organoids,
        }

        viewer.add_image(
            mosaic_image,
            name="organoid_mosaic",
            rgb=True,
            metadata=metadata,
        )

        viewer.add_labels(
            mosaic_labels,
            name="organoid_annotations",
            opacity=0.45,
            metadata=metadata,
        )

        show_info(
            "Created mosaic view with "
            f"{patches_per_side_widget.value ** 2} tiles."
        )

    @create_patch_stack_button.clicked.connect
    def _create_patch_stack() -> None:
        viewer, folder = _viewer_and_folder()

        if viewer is None:
            return

        try:
            patches, annotations, records = (
                create_organoid_patch_stack(
                    folder=folder,
                    number_patches=number_patches_widget.value,
                    well_diameter_px=well_diameter_widget.value,
                )
            )
        except (ValueError, FileNotFoundError) as error:
            show_info(str(error))
            return

        metadata = {
            "mode": "patch_stack",
            "patches": records,
        }

        viewer.add_image(
            patches,
            name="organoid_patches",
            rgb=True,
            metadata=metadata,
            axis_labels=("patch", "y", "x"),
        )

        viewer.add_labels(
            annotations,
            name="patch_annotations",
            opacity=0.45,
            metadata=metadata,
            axis_labels=("patch", "y", "x"),
        )

        show_info(
            f"Created patch-stack view with {len(patches)} selected wells. "
            "Use the patch slider to annotate them."
        )

    shared_settings = Container(
        widgets=[
            folder_widget,
            well_diameter_widget,
        ],
        labels=True,
    )

    shared_section = Container(
        widgets=[
            Label(value="Shared settings"),
            shared_settings,
        ],
        labels=False,
    )

    mosaic_settings = Container(
        widgets=[patches_per_side_widget],
        labels=True,
    )

    mosaic_section = Container(
        widgets=[
            Label(value="Mosaic"),
            mosaic_settings,
            create_mosaic_button,
        ],
        labels=False,
    )

    patch_stack_settings = Container(
        widgets=[number_patches_widget],
        labels=True,
    )

    patch_stack_section = Container(
        widgets=[
            Label(value="Patch stack"),
            patch_stack_settings,
            create_patch_stack_button,
        ],
        labels=False,
    )

    return Container(
        widgets=[
            shared_section,
            mosaic_section,
            patch_stack_section,
        ],
        labels=False,
        scrollable=True,
    )


@magic_factory(
    call_button="Train and save model",
    model_folder={"label": "Model folder", "mode": "d"},
)
def train_convpaint_widget(
    image: Image,
    annotation: Labels,
    model_folder: Path = Path("."),
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
    call_button="Save annotation",
    image={"label": "Image layer"},
    annotation={"label": "Annotation layer"},
    project_file={
        "label": "Save annotation as",
        "mode": "w",
        "filter": "Annotation (*.npz)",
    },
)
def save_annotation_project_widget(
    image: Image,
    annotation: Labels,
    project_file: Path = Path("annotation.npz"),
) -> None:
    """Save the selected image and Labels layers as one .npz project."""

    # Napari layer selectors can be empty when no matching layer exists.
    if image is None or annotation is None:
        show_info(
            "Choose an Image layer and a Labels layer before saving."
        )
        return

    if project_file is None:
        show_info("Choose a file name and location for the annotation.")
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

    show_info(f"Annotation saved to:\n{project_path}")


@magic_factory(
    call_button="Open annotation",
    project_file={
        "label": "Annotation file",
        "mode": "r",
        "filter": "Annotation (*.npz)",
    },
)
def open_annotation_project_widget(
    project_file: Path = Path("annotation.npz"),
) -> None:
    """Open an .npz annotation without requiring pre-existing layers."""

    if project_file is None:
        show_info("Choose an .npz annotation to open.")
        return

    project_path = Path(project_file)

    if not project_path.is_file():
        show_info(f"Annotation not found: {project_path}")
        return

    with np.load(project_path, allow_pickle=False) as project:
        image_key = "image" if "image" in project else "mosaic"

        if image_key not in project or "annotations" not in project:
            raise ValueError(
                "The selected file is not a valid annotation. "
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

    show_info(f"Annotation loaded from:\n{project_path}")


def annotation_project_widget() -> Container:
    """One dock widget containing independent Save and Open controls."""

    save_widget = save_annotation_project_widget()
    save_widget.label = "Save annotation"

    open_widget = open_annotation_project_widget()
    open_widget.label = "Open annotation"

    return Container(
        widgets=[
            save_widget,
            open_widget,
        ],
        labels=True,
    )


# @magic_factory(
#     call_button="Segment folder and save outputs",
#     model_path={
#         "label": "ConvPaint model",
#         "mode": "r",
#         "filter": "ConvPaint model (*.pkl)",
#     },
#     image_folder={"label": "Image folder", "mode": "d"},
#     output_folder={"label": "Output folder", "mode": "d"},
#     output_filename={"label": "Output CSV name"},
# )
# def batch_analysis_widget(
#     model_path: Path = Path("organoid_convpaint.pkl"),
#     image_folder: Path = Path("."),
#     output_folder: Path = Path("reports"),
#     output_filename: str = "output.csv",
# ) -> None:
#     """Segment a folder and save the measurements and object masks."""

#     number_images, number_organoids, mask_folder = analyze_image_folder(
#         model_path=model_path,
#         image_folder=image_folder,
#         min_area_px=5000,
#         output_folder=output_folder,
#         output_filename=output_filename,
#     )

#     output_csv = Path(output_folder) / Path(output_filename).name
#     if output_csv.suffix.lower() != ".csv":
#         output_csv = output_csv.with_suffix(".csv")

#     show_info(
#         f"Processed {number_images} images.\n"
#         f"Measured {number_organoids} organoids.\n"
#         f"CSV saved to:\n{output_csv}\n"
#         f"Segmentations saved to:\n{mask_folder}"
#     )


def batch_analysis_widget() -> Container:
    """Segment a folder, display the results, and inspect measurements."""

    # Segmentation settings.
    model_path_widget = FileEdit(
        value=Path("organoid_convpaint.pkl"),
        mode="r",
        filter="ConvPaint model (*.pkl)",
        label="ConvPaint model",
    )
    image_folder_widget = FileEdit(
        value=Path("."),
        mode="d",
        label="Image folder",
    )
    output_folder_widget = FileEdit(
        value=Path("reports"),
        mode="d",
        label="Measurements folder",
    )
    output_filename_widget = LineEdit(
        value="organoid_measurements.csv",
        label="Measurements CSV name",
    )
    min_area_widget = SpinBox(
        value=5000,
        min=1,
        max=10_000_000,
        label="Minimum object area (px)",
    )
    segment_button = PushButton(
        text="Segment folder, save, and display"
    )

    # Saved-result visualization settings.
    measurements_csv_widget = FileEdit(
        value=Path("organoid_measurements.csv"),
        mode="r",
        filter="Measurements CSV (*.csv)",
        label="Measurements CSV",
    )
    segmentation_folder_widget = FileEdit(
        value=Path("."),
        mode="d",
        label="Segmentation folder",
    )
    feature_widget = ComboBox(
        choices=DEFAULT_MEASUREMENT_FEATURES,
        value="area_px",
        label="Color objects by",
    )
    colormap_widget = ComboBox(
        choices=("turbo", "viridis", "plasma", "inferno", "magma"),
        value="turbo",
        label="Colormap",
    )
    display_button = PushButton(
        text="Display saved results and measurement map"
    )

    measurement_table = Table(
        value=[],
        label="Measurements",
    )
    measurement_table.read_only = True
    measurement_table.max_height = 300

    def _viewer():
        viewer = napari.current_viewer()
        if viewer is None:
            show_info("No napari viewer is open.")
        return viewer

    def _show_table_and_update_features(table):
        measurement_table.value = table

        features = measurement_feature_columns(table)
        if not features:
            return ()

        previous_feature = feature_widget.value
        feature_widget.choices = features

        if previous_feature in features:
            feature_widget.value = previous_feature
        else:
            feature_widget.value = features[0]

        return features

    @segment_button.clicked.connect
    def _segment_folder() -> None:
        viewer = _viewer()
        if viewer is None:
            return

        image_folder = Path(image_folder_widget.value)
        output_folder = Path(output_folder_widget.value)
        output_filename = output_filename_widget.value.strip()

        if not output_filename:
            show_info("Enter a measurements CSV file name.")
            return

        try:
            number_images, number_organoids, mask_folder = (
                analyze_image_folder(
                    model_path=Path(model_path_widget.value),
                    image_folder=image_folder,
                    min_area_px=min_area_widget.value,
                    output_folder=output_folder,
                    output_filename=output_filename,
                )
            )
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            show_info(str(error))
            return

        output_csv = output_folder / Path(output_filename).name
        if output_csv.suffix.lower() != ".csv":
            output_csv = output_csv.with_suffix(".csv")

        measurements_csv_widget.value = output_csv
        segmentation_folder_widget.value = mask_folder

        display_warning = ""
        try:
            display_segmentation_results(
                viewer=viewer,
                image_folder=image_folder,
                mask_folder=mask_folder,
            )
        except (ValueError, FileNotFoundError) as error:
            display_warning = f"\n\nDisplay warning:\n{error}"

        table = read_measurements_csv(output_csv)
        features = _show_table_and_update_features(table)

        if not table.empty and features:
            try:
                display_measurement_map(
                    viewer=viewer,
                    table=table,
                    segmentation_folder=mask_folder,
                    feature=feature_widget.value,
                    colormap=colormap_widget.value,
                )
            except (ValueError, FileNotFoundError) as error:
                display_warning += (
                    f"\n\nMeasurement-map warning:\n{error}"
                )

        show_info(
            f"Processed {number_images} images.\n"
            f"Measured {number_organoids} organoids.\n"
            f"CSV saved to:\n{output_csv}\n"
            f"Segmentations saved to:\n{mask_folder}"
            f"{display_warning}"
        )

    @display_button.clicked.connect
    def _display_saved_results() -> None:
        viewer = _viewer()
        if viewer is None:
            return

        image_folder = Path(image_folder_widget.value)
        segmentation_folder = Path(
            segmentation_folder_widget.value
        )
        csv_path = Path(measurements_csv_widget.value)

        try:
            table = read_measurements_csv(csv_path)
            features = _show_table_and_update_features(table)

            display_segmentation_results(
                viewer=viewer,
                image_folder=image_folder,
                mask_folder=segmentation_folder,
            )

            if table.empty:
                show_info(
                    "The CSV was loaded, but it contains no measured objects."
                )
                return

            if not features:
                raise ValueError(
                    "The CSV contains no numeric measurement columns."
                )

            layer = display_measurement_map(
                viewer=viewer,
                table=table,
                segmentation_folder=segmentation_folder,
                feature=feature_widget.value,
                colormap=colormap_widget.value,
            )
        except (ValueError, FileNotFoundError) as error:
            show_info(str(error))
            return

        minimum, maximum = layer.contrast_limits
        show_info(
            f"Displayed {feature_widget.value}.\n"
            f"Color range: {minimum:.4g} to {maximum:.4g}."
        )

    segmentation_settings = Container(
        widgets=[
            model_path_widget,
            image_folder_widget,
            output_folder_widget,
            output_filename_widget,
            min_area_widget,
        ],
        labels=True,
    )
    segmentation_section = Container(
        widgets=[
            Label(value="Run segmentation"),
            segmentation_settings,
            segment_button,
        ],
        labels=False,
    )

    result_settings = Container(
        widgets=[
            measurements_csv_widget,
            segmentation_folder_widget,
            feature_widget,
            colormap_widget,
        ],
        labels=True,
    )
    result_section = Container(
        widgets=[
            Label(value="Display measurements"),
            result_settings,
            display_button,
            measurement_table,
        ],
        labels=False,
    )

    return Container(
        widgets=[
            segmentation_section,
            result_section,
        ],
        labels=False,
        scrollable=True,
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
        "label": "Predicted Segmentation folder",
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
    output_csv: Path = Path("predicted_iou.csv"),
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

    roi_widget = visualize_roi_widget()
    roi_widget.label = "Visualize ROI"
    
    iou_widget = calculate_iou_widget()
    iou_widget.label = "Calculate IoU"

    return Container(
        widgets=[
            roi_widget,
            iou_widget,
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
