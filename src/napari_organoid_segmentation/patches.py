from pathlib import Path
import math

import numpy as np

from .mosaic import create_organoid_mosaic


def create_organoid_patch_stack(
    folder: Path,
    number_patches: int = 16,
    well_diameter_px: int = 280,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return exactly ``number_patches`` separate RGB patches."""

    if number_patches < 1:
        raise ValueError("number_patches must be at least 1.")

    patches_per_side = math.ceil(math.sqrt(number_patches))

    mosaic, _, records = create_organoid_mosaic(
        folder=folder,
        patches_per_side=patches_per_side,
        number_patches=number_patches,
        well_diameter_px=well_diameter_px,
    )

    patch_size = mosaic.shape[0] // patches_per_side
    patches = []

    for record in records:
        y0 = record["row"] * patch_size
        x0 = record["column"] * patch_size

        patches.append(
            mosaic[
                y0 : y0 + patch_size,
                x0 : x0 + patch_size,
            ].copy()
        )

    patch_stack = np.stack(patches)

    annotations = np.zeros(
        patch_stack.shape[:3],
        dtype=np.uint8,
    )

    return patch_stack, annotations, records