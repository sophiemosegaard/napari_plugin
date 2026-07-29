# import math
# import numpy as np
# from skimage.measure import label
# from skimage.morphology import closing, disk, opening


# def clean_and_label_mask(
#     mask: np.ndarray,
#     opening_radius: int = 0,
#     closing_radius: int = 0,
#     min_radius_px: float = 0,
# ) -> np.ndarray:
#     """Clean a binary mask, remove small objects, and label organoids.

#     Parameters
#     ----------
#     mask:
#         Binary foreground mask.
#     opening_radius:
#         Radius of the disk used for morphological opening.
#     closing_radius:
#         Radius of the disk used for morphological closing.
#     min_radius_px:
#         Minimum equivalent organoid radius in pixels.

#     Returns
#     -------
#     np.ndarray
#         Integer label image:
#         0 = background
#         1, 2, 3, ... = individual organoids
#     """

#     cleaned_mask = np.asarray(mask, dtype=bool)

#     if opening_radius > 0:
#         cleaned_mask = opening(
#             cleaned_mask,
#             footprint=disk(opening_radius),
#         )

#     if closing_radius > 0:
#         cleaned_mask = closing(
#             cleaned_mask,
#             footprint=disk(closing_radius),
#         )

#     initial_labels = label(cleaned_mask)

#     # No radius filtering requested.
#     if min_radius_px <= 0:
#         return initial_labels.astype(np.int32)

#     minimum_area_px = math.pi * min_radius_px**2

#     # Count the pixels belonging to every label.
#     object_areas = np.bincount(initial_labels.ravel())

#     keep_labels = np.flatnonzero(
#         object_areas >= minimum_area_px
#     )

#     # Label 0 is the background and must never be retained as an object.
#     keep_labels = keep_labels[keep_labels != 0]

#     # Create a lookup table that also relabels objects consecutively.
#     mapping = np.zeros(
#         initial_labels.max() + 1,
#         dtype=np.int32,
#     )

#     mapping[keep_labels] = np.arange(
#         1,
#         len(keep_labels) + 1,
#         dtype=np.int32,
#     )

#     filtered_labels = mapping[initial_labels]

#     return filtered_labels