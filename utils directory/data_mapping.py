import json
import os

def map_data(segmented_objects, extracted_texts, master_image_id):
    """
    Maps extracted data to segmented objects with unique IDs and descriptions.

    Args:
    segmented_objects (list): List of segmented object filenames.
    extracted_texts (dict): Dictionary mapping filenames to extracted texts.
    master_image_id (str): Unique ID for the master input image.

    Returns:
    dict: A mapping of unique IDs, descriptions, and extracted texts for each object.
    """
    mapped_data = []

    for i, obj in enumerate(segmented_objects):
        unique_id = f"obj_{i+1}"  # Unique ID for each segmented object
        description = f"Segmented Object {i+1}"  # Description for the object
        extracted_text = extracted_texts.get(obj, "No text extracted")  # Get extracted text

        # Create a mapping entry for the object
        entry = {
            "unique_id": unique_id,
            "description": description,
            "extracted_text": extracted_text,
            "master_image_id": master_image_id
        }
        mapped_data.append(entry)

    return mapped_data

def save_mapping_to_json(mapped_data, output_path):
    """
    Saves the mapped data to a JSON file.

    Args:
    mapped_data (list): List of mapped data entries.
    output_path (str): Path to save the JSON file.
    """
    # Create output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w') as json_file:
        json.dump(mapped_data, json_file, indent=4)

# Example usage
if __name__ == "__main__":
    segmented_objects = ["segmented_object_1.png", "segmented_object_2.png"]  # Example filenames
    extracted_texts = {
        "segmented_object_1.png": "Text from object 1",
        "segmented_object_2.png": "Text from object 2"
    }
    master_image_id = "input_image_1"  # Example master image ID

    mapped_data = map_data(segmented_objects, extracted_texts, master_image_id)
    save_mapping_to_json(mapped_data, "output/mapped_data.json")
    print("Mapping saved to output/mapped_data.json")
