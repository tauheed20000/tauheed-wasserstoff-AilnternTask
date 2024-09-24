import numpy as np
import cv2

def apply_threshold(mask, threshold=0.5):
    """
    Apply a threshold to the mask to create a binary segmentation.

    Args:
    mask (np.ndarray): Input segmentation mask.
    threshold (float): Threshold value (0 to 1).

    Returns:
    np.ndarray: Binary mask.
    """
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

def mask_to_image(mask, original_image):
    """
    Convert a binary mask to an image by applying the mask to the original image.

    Args:
    mask (np.ndarray): Binary mask.
    original_image (np.ndarray): Original image.

    Returns:
    np.ndarray: Image with the mask applied.
    """
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask == 1] = [0, 255, 0]  # Example: Green color for mask
    return cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)

def postprocess_output(segmentation_output, original_image):
    """
    Postprocess the model output.

    Args:
    segmentation_output (np.ndarray): Model's output mask.
    original_image (np.ndarray): Original input image.

    Returns:
    np.ndarray: Image with applied segmentation mask.
    """
    # Ensure that the output mask is resized to match the original image dimensions
    height, width = original_image.shape[:2]
    resized_mask = cv2.resize(segmentation_output, (width, height), interpolation=cv2.INTER_LINEAR)
    
    binary_mask = apply_threshold(resized_mask)
    return mask_to_image(binary_mask, original_image)

# Example usage
if __name__ == "__main__":
    # Assume segmentation_output is the output from the model
    segmentation_output = np.random.rand(640, 640)  # Dummy segmentation output
    original_image = cv2.imread("data directory/input_images/cycle.jpg")

    postprocessed_image = postprocess_output(segmentation_output, original_image)
    cv2.imwrite("data directory/output/postprocessed_image.png", postprocessed_image)
    print("Postprocessing completed.")
