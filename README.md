# STEP 1: random organoid mosaic

Install the plugin from this folder:

```bash
python -m pip install -e . 
```

Start napari and open:

`Plugins -> Organoid Segmentation -> Create Organoid Mosaic`

Suggested first settings:

FIXME: I want to make the plugin very easy with as less parameters as possible and it should still work.....
- `rows = 4`, `columns = 4` while testing
- `patch_size` larger than twice the largest organoid radius
- `min_radius_px` and `max_radius_px` around the expected well radius
- lower `hough_score` if too few circles are detected
- increase `random_seed` to create a different random mosaic

The widget creates:

1. `organoid_mosaic`: the image mosaic
2. `organoid_circle_labels`: editable labels, with 0 as background and one label for the organoids to segment

Use napari's Labels paint/area tools to correct the borders and to draw specific organoid borders.