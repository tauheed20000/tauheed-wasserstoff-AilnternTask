import pandas as pd
import os

# Define the path to the extraction summary table
output_dir = 'C:\\Users\\Hp\\OneDrive\\Desktop\\streamlit_dashboards\\AI_model Projects\\data directory\\output\\'
summary_table_path = os.path.join(output_dir, 'extraction_summary.csv')

# Step 1: Load the extraction summary
results_df = pd.read_csv(summary_table_path)

# Step 2: Generate a summary of extracted text
# Convert all values to strings, handling NaNs
results_df['Extracted Text'] = results_df['Extracted Text'].astype(str)

# Generate summary by joining all extracted texts
summary = " ".join(results_df['Extracted Text'].tolist())

# Step 3: Save the summary to a text file
summary_file_path = os.path.join(output_dir, 'extraction_summary.txt')
with open(summary_file_path, 'w') as summary_file:
    summary_file.write(summary)

print(f'Summarized text saved at {summary_file_path}')
