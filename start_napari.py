"""Start napari with PyTorch loaded before Qt."""

# This must be the first important import.
import torch

print(f"PyTorch loaded successfully: {torch.__version__}")

import napari


def main() -> None:
    viewer = napari.Viewer()
    napari.run()


if __name__ == "__main__":
    main()