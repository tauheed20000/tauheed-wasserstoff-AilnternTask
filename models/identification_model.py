import cv2
import torch
from torchvision import models, transforms
import os
import pandas as pd
import requests  # Add this import

# Step 1: Define directories
segmented_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\segmented_objects\\'
output_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\output\\'

# Load the pre-trained classification model
classification_model = models.resnet50(pretrained=True)
classification_model.eval()  # Set the model to evaluation mode

# Define the transformation for the images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet labels for the classification
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()  # Fetch labels

# Step 2: Process segmented objects
results = []

# Iterate through each segmented object image
for segmented_image_file in os.listdir(segmented_dir):
    segmented_image_path = os.path.join(segmented_dir, segmented_image_file)
    
    # Load and preprocess the segmented image
    segmented_image = cv2.imread(segmented_image_path)
    segmented_image_tensor = preprocess(segmented_image).unsqueeze(0)  # Add batch dimension

    # Identify the object
    with torch.no_grad():
        output = classification_model(segmented_image_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = labels[predicted_idx.item()]  # Get the predicted label

    # Save the result
    results.append({
        'Segmented Image': segmented_image_file,
        'Identified Class': predicted_label
    })

# Step 3: Save results to a summary table
results_df = pd.DataFrame(results)
summary_table_path = os.path.join(output_dir, 'identification_summary.csv')
results_df.to_csv(summary_table_path, index=False)
print(f'Summary table saved at {summary_table_path}')
