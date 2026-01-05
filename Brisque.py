import cv2
import csv

# Function to calculate BRISQUE score
def calculate_brisque_score(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at path: {image_path}")
        return None
    brisque_model = cv2.quality.QualityBRISQUE_create("brisque_model_live.yml", "brisque_range_live.yml")
    brisque_score = brisque_model.compute(image)[0]
    return brisque_score

# Read paths and labels from CSV, calculate BRISQUE scores, and save to a new CSV
input_csv = "data_to_train_NN.csv"  # Your input CSV file
output_csv = "artifact_scores_brisque.csv"  # Output CSV file

with open(input_csv, mode='r') as file, open(output_csv, mode='w', newline='') as outfile:
    reader = csv.reader(file)
    writer = csv.writer(outfile)
    writer.writerow(['path', 'label', 'artifact_score'])  # Write header row
    next(reader)  # Skip header in input CSV if present

    for row in reader:
        path, label = row[0], int(row[1])
        artifact_score = calculate_brisque_score(path)
        if artifact_score is not None:
            writer.writerow([path, label, artifact_score])

print(f"Artifact scores saved to {output_csv}")
