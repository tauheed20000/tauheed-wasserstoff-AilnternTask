import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def annotate_image(original_image, masks, object_data):
    """
    Annotate the original image with segmentation masks and object IDs.

    Args:
    original_image (np.ndarray): The original input image.
    masks (list): List of binary masks for each object.
    object_data (list of dicts): List of dictionaries containing object information.

    Returns:
    np.ndarray: Annotated image.
    """
    annotated_image = original_image.copy()
    
    for i, mask in enumerate(masks):
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask == 1] = [0, 255, 0]  # Green color for mask
        
        # Add the mask to the annotated image
        annotated_image = cv2.addWeighted(annotated_image, 0.7, colored_mask, 0.3, 0)
        
        # Annotate with object ID
        object_id = object_data[i]['id']
        cv2.putText(annotated_image, f'ID: {object_id}', 
                    (10, 30 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
    
    return annotated_image

def generate_summary_table(object_data):
    """
    Generate a summary table of mapped data.

    Args:
    object_data (list of dicts): List of dictionaries containing object information.

    Returns:
    pd.DataFrame: Summary table as a DataFrame.
    """
    df = pd.DataFrame(object_data)
    return df

def save_final_output(original_image, annotated_image, summary_table, output_path):
    """
    Save the final output with the annotated image and summary table.

    Args:
    original_image (np.ndarray): The original input image.
    annotated_image (np.ndarray): The annotated image.
    summary_table (pd.DataFrame): Summary table.
    output_path (str): Path to save the final output.
    """
    # Save annotated image
    annotated_image_path = output_path + "/annotated_image.png"
    cv2.imwrite(annotated_image_path, annotated_image)

    # Save summary table
    summary_table_path = output_path + "/summary_table.csv"
    summary_table.to_csv(summary_table_path, index=False)

    # Display final output
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Annotated Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.table(cellText=summary_table.values, colLabels=summary_table.columns, 
              cellLoc = 'center', loc='center')
    plt.axis('off')
    plt.title('Summary Table')

    plt.savefig(output_path + "/final_output.png", bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    original_image_path = "data directory/input_images/cycle.jpg"
    original_image = cv2.imread(original_image_path)

    # Dummy segmentation masks and object data for demonstration
    masks = [np.random.randint(0, 2, original_image.shape[:2]).astype(np.uint8) for _ in range(3)]
    object_data = [
        {'id': 1, 'description': 'Object 1', 'summary': 'Summary of object 1'},
        {'id': 2, 'description': 'Object 2', 'summary': 'Summary of object 2'},
        {'id': 3, 'description': 'Object 3', 'summary': 'Summary of object 3'},
    ]

    annotated_image = annotate_image(original_image, masks, object_data)
    summary_table = generate_summary_table(object_data)
    save_final_output(original_image, annotated_image, summary_table, "data directory/output")
