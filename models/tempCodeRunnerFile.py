import cv2
import torch
from torchvision import transforms
from torchvision.models import detection
import os

# Step 1: Define directories
input_images = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\input_images\\'
segmented_objects = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\segmented_objects\\'
output = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\output\\'

# Create output directories if they don't exist
os.makedirs(segmented_objects, exist_ok=True)
os.makedirs(output, exist_ok=True)

# Load a pre-trained Mask R-CNN model
model = detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image
image_path = os.path.join(input_images, 'cycle.jpg')
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert image to tensor and add batch dimension
image_tensor = transforms.ToTensor()(image).unsqueeze(0)  

# Perform object detection
with torch.no_grad():
    predictions = model(image_tensor)

# Step 2: Process and save segmented objects
segmented_images = []  # List to store segmented images
for i, (mask, score) in enumerate(zip(predictions[0]['masks'], predictions[0]['scores'])):
    if score > 0.5:  # Filter out low-confidence predictions
        # Get the mask and convert it to a binary image
        mask = mask[0].mul(255).byte().cpu().numpy()  # Scale to 0-255
        segmented_image = image.copy()  # Make a copy of the original image
        segmented_image[mask == 0] = 0  # Set background pixels to black

        # Save segmented object
        segmented_object_path = os.path.join(segmented_objects, f'segmented_object_{i + 1}.png')
        cv2.imwrite(segmented_object_path, segmented_image)
        print(f'Saved segmented object {i + 1} at {segmented_object_path}')

        # Append the segmented image to the list for further use
        segmented_images.append(segmented_image)

