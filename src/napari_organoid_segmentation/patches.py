from pathlib import Path
import numpy as np

from .mosaic import create_organoid_mosaic


def create_organoid_patch_stack(
    folder: Path,
    patches_per_side: int = 4,
    well_diameter_px: int = 280,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return the mosaic wells as separate RGB patches."""

    mosaic, _, records = create_organoid_mosaic(
        folder=folder,
        patches_per_side=patches_per_side,
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