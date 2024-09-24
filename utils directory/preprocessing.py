import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path):
    """
    Load an image from the specified path.

    Args:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: Loaded image in BGR format.
    """
    image = cv2.imread(image_path)
    return image

def resize_image(image, target_size=(640, 640)):
    """
    Resize the image to the target size.

    Args:
    image (np.ndarray): Input image.
    target_size (tuple): Desired size (width, height).

    Returns:
    np.ndarray: Resized image.
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Normalize the image for model input.

    Args:
    image (np.ndarray): Input image.

    Returns:
    tensor: Normalized image tensor.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    return transform(image)

def preprocess_image(image_path):
    """
    Preprocess the image by loading, resizing, and normalizing it.

    Args:
    image_path (str): Path to the image file.

    Returns:
    tensor: Preprocessed image tensor.
    """
    image = load_image(image_path)
    image = resize_image(image)
    return normalize_image(image)

# Example usage
if __name__ == "__main__":
    preprocessed_image = preprocess_image("data directory/input_images/cycle.jpg")
    print("Preprocessing completed.")
