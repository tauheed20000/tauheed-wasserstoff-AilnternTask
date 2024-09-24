import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import detection
import torch

# Step 1: Define directories
input_images_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\input_images\\'
segmented_objects_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\segmented_objects\\'
output_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\output\\'

# Load pre-trained Mask R-CNN model
model = detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()  # Set the model to evaluation mode

# Function to perform segmentation
def segment_image(image):
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    
    segmented_images = []
    for i, (mask, score) in enumerate(zip(predictions[0]['masks'], predictions[0]['scores'])):
        if score > 0.5:  # Filter out low-confidence predictions
            mask = mask[0].mul(255).byte().cpu().numpy()
            
            # Resize the mask to match the input image size
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segmented_image[mask_resized == 0] = 0  # Apply the resized mask
            segmented_images.append(segmented_image)

            # Save segmented object
            segmented_object_path = os.path.join(segmented_objects_dir, f'segmented_object_{i + 1}.png')
            cv2.imwrite(segmented_object_path, segmented_image)
            print(f'Saved segmented object {i + 1} at {segmented_object_path}')
    
    return segmented_images

# Streamlit UI
st.title("Image Segmentation App")
st.write("Upload an image to segment objects.")

# Step 2: File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Step 3: Perform Segmentation
    if st.button("Segment Image"):
        segmented_images = segment_image(image)
        st.session_state.segmented_images = segmented_images  # Store in session state
        
        # Step 4: Display Segmented Objects
        st.write("Segmented Objects:")
        for idx, seg_img in enumerate(segmented_images):
            st.image(seg_img, caption=f'Segmented Object {idx + 1}', use_column_width=True)

    # Step 5: Final Output Display
    if st.button("Show Final Output"):
        if 'segmented_images' in st.session_state:  # Check if segmented_images exists
            st.write("Final Output with Annotations")
            segmented_images = st.session_state.segmented_images  # Retrieve from session state
            output_image = image.copy()
            for idx, seg_img in enumerate(segmented_images):
                # Ensure mask is the same size as the image
                output_image[seg_img != 0] = seg_img[seg_img != 0]  # Overlay the segmented images
            st.image(output_image, caption='Final Output Image', use_column_width=True)

            # Displaying table of mapped data
            st.write("Mapped Data Table:")
            mapped_data = [{"Object ID": f"Object {i + 1}", "Description": f"Segmented object {i + 1}"} for i in range(len(segmented_images))]
            st.table(mapped_data)
        else:
            st.write("Please segment the image first before showing the final output.")
