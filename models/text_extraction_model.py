import cv2
import pytesseract
import os
import pandas as pd

# Define the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Hp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Adjust path if necessary

# Step 1: Define directories
segmented_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\segmented_objects\\'
output_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\output\\'

# Step 2: Process segmented objects for text extraction
results = []

# Iterate through each segmented object image
for segmented_image_file in os.listdir(segmented_dir):
    segmented_image_path = os.path.join(segmented_dir, segmented_image_file)

    # Read the image
    image = cv2.imread(segmented_image_path)

    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(image)

    # Save the result
    results.append({
        'Segmented Image': segmented_image_file,
        'Extracted Text': extracted_text.strip()  # Strip whitespace from the extracted text
    })

# Step 3: Save results to a summary table
results_df = pd.DataFrame(results)
text_summary_table_path = os.path.join(output_dir, 'extraction_summary.csv')
results_df.to_csv(text_summary_table_path, index=False)
print(f'Text extraction summary table saved at {text_summary_table_path}')
