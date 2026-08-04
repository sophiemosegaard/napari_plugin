# Organoid Segmentation

Napari plugin for building organoid annotation layers, saving projects, training a ConvPaint model, segmenting folders, and evaluating saved segmentations against ImageJ ROIs.

## Install

Install the package in editable mode from this repository:

```bash
python -m pip install -e .
```

Then start napari from this workspace:

```bash
python start_napari.py
```

Open the plugin from `Plugins -> Organoid Segmentation`.

## Main Widgets

### Create Annotation Layer

Use this widget to choose how the annotation data should be presented.

Choose `Mosaic` when you want a single tiled image for fast review. Choose `Patch stack` when you want the selected wells shown as a stack of separate slices.

Useful settings:

* `Image folder`: folder containing the raw images.
* `Annotation mode`: switch between mosaic and patch-stack output.
* `Patches per side`: number of tiles per row and column.
* `Number patches`: total number of patches to use when `Annotation mode` is `Patch stack`.
* `Well diameter (px)`: approximate well diameter in pixels.

### Save or Open Annotation Project

This widget combines saving and loading into one place.

* `Save project` stores the currently active image and annotation layers as an `.npz` file in the folder you choose, using the file name you type.
* `Open project` opens a saved `.npz` project file and restores the corresponding image and annotation layers.
* Opening does not require any image or annotation layers to be loaded already.

When saving, make sure the correct image and annotation layers are visible in napari.


### Train and Save Model

Train a ConvPaint model from the currently displayed image and annotation layers, then save the model to a folder of your choice.

### Segment Folder and Save Measurements

Run batch segmentation on a folder of images, save the measurements table, and write the object masks to a sibling segmentation folder.

### Load ROI Folder or Calculate IoU

This widget overlays ImageJ ROIs on top of the current image stack for visual inspection. It can also calculate IoU against previously saved segmentation masks and write the results to CSV.

* `Visualize ROIs` loads the ROI labels only.
* `Visualize ROIs and calculate IoU` loads the ROI labels and also compares them with saved segmentation masks.

You need the current image stack loaded in napari so the ROI folder can be matched to the correct shape.

## Suggested Workflow

1. Create a mosaic or patch stack from an image folder.
2. Clean up the annotation in napari.
3. Save the project if you want to continue later.
4. Train the ConvPaint model.
5. Segment a folder and save measurements.
6. Optionally load ROI folders and calculate IoU for comparison.

## Author

Created by Sophie Mosegaard.