from pathlib import Path
import cv2
import numpy as np


# -----------------------------
# Settings
# -----------------------------
INPUT_FOLDER = Path("C:\\Users\\sophi\\semesterproject2\\data\\raw\\Junho\\6d_molde\\4x\\1\\TIFF")
OUTPUT_FOLDER = Path("C:\\Users\\sophi\\napari-organoid-segmentation\\reports\\segmentations\\otsu_results")

OUTPUT_FOLDER.mkdir(exist_ok=True)

MIN_AREA = 5000

# Image extensions to process
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# -----------------------------
# Process all images
# -----------------------------
for image_path in INPUT_FOLDER.iterdir():

    if image_path.suffix.lower() not in image_extensions:
        continue

    # Load image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load: {image_path}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional small blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu thresholding
    otsu_threshold, mask = cv2.threshold(
        gray_blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    # Remove objects smaller than MIN_AREA
    # Create empty filtered mask
    filtered_mask = np.zeros_like(mask)
    
    # Label image for napari
    label_image = np.zeros(mask.shape, dtype=np.uint16)
    
    new_label = 1  # Start labeling from 1

    # Start at 1 because label 0 is the background
    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_AREA:
            filtered_mask[labels == label] = 255
            label_image[labels == label] = new_label
            new_label += 1
    
    
    # Apply mask to original image
    segmented_image = cv2.bitwise_and(
        image,
        image,
        mask=filtered_mask
    )

    # -----------------------------
    # Save results
    # -----------------------------
    name = image_path.stem
    label_output_path = OUTPUT_FOLDER / "6d-1" / f"{name}.png"
    label_output_path.parent.mkdir(exist_ok=True)

    # cv2.imwrite(
    #     str(mask_output_path),
    #     mask
    # )

    cv2.imwrite(
        str(label_output_path),
        label_image
    )

    print(
        f"{image_path.name}: "
        f"Otsu threshold = {otsu_threshold:.2f}"
    )


print("Finished!")