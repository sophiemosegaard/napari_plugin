# Organoid Segmentation Workflow

Install the plugin from this folder:

```bash
python -m pip install -e . 
```

Start napari and open:

`Plugins -> Organoid Segmentation -> Create Organoid Mosaic`

Suggested first settings:

### Well diameter

Measure the diameter of one representative well in FIJI and enter
the result as `well_diameter_px`.

The plugin automatically searches a small radius range around this
measurement. During segmentation postprocessing, an additional
internal margin is used to associate organoids with their wells.
The complete connected segmentation is retained and is not cropped
at the Hough-circle boundary.

FIXME: I want to make the plugin very easy with as less parameters as possible and it should still work.....
- `rows = 4`, `columns = 4` while testing
- `patch_size` larger than twice the largest organoid radius
- `well_diameter_px`: the well diameter measured in pixels, for example in FIJI
- lower `hough_score` if too few circles are detected
- increase `random_seed` to create a different random mosaic

The workflow is split into three steps:

1. Create a mosaic from an image folder and optionally save the edited annotation as an `*.npz` project.
2. Train a ConvPaint model from the mosaic image and its annotation, then save the model to a folder you choose.
3. Load a saved model and run segmentation plus measurements on an image folder, saving the measurement CSV and the object masks to locations you choose.

The first widget creates:

1. `organoid_mosaic`: the image mosaic
2. `organoid_annotations`: editable labels, with 0 as background and one label for the organoids to segment

Use napari's Labels paint/area tools to correct the borders and to draw specific organoid borders.

Suggested starting values:

- `rows = 4`, `columns = 4` while testing
- lower the circle detection threshold if too few circles are detected
- increase the number of selected tiles if you want a larger mosaic



## Author

Created by **Sophie Mosegaard**.

## Coffee-powered development ☕

This plugin runs on Python; its student developer runs on coffee.

If the plugin saves you some time and you enjoy using it, an optional coffee donation is always appreciated:

<YOUR_DONATION_LINK>