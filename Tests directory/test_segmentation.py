import unittest
import numpy as np
from PIL import Image
from image_processing import segment_image  # Replace with your actual module name
import torch

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        # Set up a sample image for testing
        self.sample_image = np.ones((200, 300, 3), dtype=np.uint8) * 255  # White image
        self.pil_image = Image.fromarray(self.sample_image)

    def test_segmentation_output(self):
        # Test if the segmentation returns a list of segmented objects
        segmented_images = segment_image(self.pil_image)
        
        # Ensure it's a list
        self.assertIsInstance(segmented_images, list)

        # Ensure the list is not empty if objects are found
        self.assertGreater(len(segmented_images), 0)

    def test_segmentation_tensor_shape(self):
        # Test that the segmentation process handles the image dimensions correctly
        image_tensor = torch.tensor(self.sample_image).permute(2, 0, 1)  # Convert to tensor (C, H, W)
        self.assertEqual(image_tensor.shape, (3, 200, 300))  # Check if the shape is correct

if __name__ == '__main__':
    unittest.main()
