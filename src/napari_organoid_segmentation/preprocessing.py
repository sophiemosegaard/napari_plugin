import numpy as np
from skimage.color import rgb2gray


def as_grayscale(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)

    if data.ndim == 2:
        return data.astype(np.float32)

    if data.ndim == 3 and data.shape[-1] in (3, 4):
        return rgb2gray(data[..., :3]).astype(np.float32)

    raise ValueError(
        f"Expected a 2D grayscale or RGB image, got {data.shape}."
    )