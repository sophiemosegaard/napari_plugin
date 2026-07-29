from pathlib import Path
from typing import Any

import numpy as np
from .preprocessing import as_grayscale
from skimage.util import img_as_float32


# The model remains available while napari is open.
_CONVPAINT_MODEL: Any | None = None

def _model_class():
    try:
        from napari_convpaint import ConvpaintModel
    except ImportError as error:
        raise RuntimeError(
            "Install ConvPaint with: pip install napari-convpaint"
        ) from error

    return ConvpaintModel

def _prepare_image(image: np.ndarray) -> np.ndarray:
    """Prepare RGB data for ConvPaint as (3, Y, X)."""

    data = np.asarray(image)

    # Convert grayscale to three identical RGB channels.
    if data.ndim == 2:
        data = np.repeat(data[..., np.newaxis], 3, axis=-1)

    # Accept RGB and RGBA images.
    elif data.ndim == 3 and data.shape[-1] in (3, 4):
        data = data[..., :3]

    else:
        raise ValueError(
            "Expected a 2D grayscale or RGB image, "
            f"but received {data.shape}."
        )

    data = img_as_float32(data)

    # ConvPaint requires channel-first RGB data:
    # (Y, X, 3) -> (3, Y, X)
    return np.moveaxis(data, -1, 0)


def train_and_save_convpaint(
    image: np.ndarray,
    annotations: np.ndarray,
    model_folder: Path,
    model_name: str,
) -> Path:
    """Train and save a ConvPaint model."""

    global _CONVPAINT_MODEL

    prepared_image = _prepare_image(image)

    annotation_data = np.asarray(
        annotations,
        dtype=np.uint16,
    )

    if annotation_data.shape != prepared_image.shape[1:]:
        raise ValueError(
            "Image and annotation shapes do not match."
            f"Image: {prepared_image.shape}; "
            f"annotations: {annotation_data.shape}."
        )

    classes = set(np.unique(annotation_data))

    if not {1, 2}.issubset(classes):
        raise ValueError(
            "Annotations need class 1 for background "
            "and class 2 for organoids."
        )

    model = _model_class()("vgg")

    model.set_params(
        channel_mode="rgb",
        normalize = 3,
    )

    model.train(
        prepared_image,
        annotation_data,
    )

    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_name).stem.strip()

    if not model_name:
        raise ValueError("Please enter a model name.")

    model_path = model_folder / model_name

    model.save(str(model_path), create_pkl=True,create_yml=True)

    _CONVPAINT_MODEL = model

    return model_path.with_suffix(".pkl")


def load_convpaint_model(
    model_path: Path,
) -> Path:
    """Load a saved ConvPaint model."""

    global _CONVPAINT_MODEL

    model_path = Path(model_path)

    if model_path.suffix.lower() != ".pkl":
        model_path = model_path.with_suffix(".pkl")

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = _model_class()(
        model_path=str(model_path)
    )
    
    channel_mode = model.get_param("channel_mode")
    
    if channel_mode != "rgb":
        raise ValueError(
            "The selected model was trained as "
            f"{channel_mode!r}, not as RGB. "
            "Please train a new model using an RGB mosaic."
        )

    _CONVPAINT_MODEL = model

    return model_path


def segment_convpaint(
    image: np.ndarray,
) -> np.ndarray:
    """Return 0=background and 1=organoid."""

    if _CONVPAINT_MODEL is None:
        raise RuntimeError(
            "Train or load a ConvPaint model first."
        )

    prepared = _prepare_image(image)

    prediction = np.asarray(
        _CONVPAINT_MODEL.segment(prepared)
    )

    prediction = np.squeeze(prediction)

    expected_shape = prepared.shape[1:]

    if prediction.shape != expected_shape:
        raise RuntimeError(
            "ConvPaint returned an unexpected shape. "
            f"RGB input: {prepared.shape}; "
            f"expected prediction: {expected_shape}; "
            f"received prediction: {prediction.shape}."
        )

    return (prediction == 2).astype(np.uint8)