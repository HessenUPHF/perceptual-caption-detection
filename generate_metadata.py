import csv
import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import models, transforms

# Load pretrained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode

# Freeze the model's parameters to prevent training
for param in model.parameters():
    param.requires_grad = False

# Adjusted projection layer for 1D input from ResNet
projection_layer = nn.Sequential(
    nn.Linear(model.fc.out_features, 128),  # Directly reduce dimensions from the ResNet output
    nn.ReLU()
)
projection_layer.eval()

# Define the image transformation for ResNet input size
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet input size
    transforms.ToTensor(),
])

# Define quality metric calculations
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std() / 255.0  # Normalize contrast to a range of 0 to 1
    return contrast

def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2]) / 255.0  # Normalize brightness to a range of 0 to 1
    return brightness

# Function to extract features, project, and calculate quality metrics
def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension for model

    # Extract features with ResNet and pass through the projection layer
    with torch.no_grad():
        features = model(image_tensor)
        projected_features = projection_layer(features)

    # Calculate quality metrics using OpenCV functions
    image_cv2 = cv2.imread(image_path)
    sharpness = calculate_sharpness(image_cv2)
    contrast = calculate_contrast(image_cv2)
    brightness = calculate_brightness(image_cv2)

    return sharpness, contrast, brightness

# Read from existing CSV and add quality metrics to create a new metadata file
input_csv = "artifact_scores_brisque.csv"  # CSV file you uploaded with path, label, and artifact_score
output_csv = "metadata.csv"  # New output CSV file with added quality metrics

with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    writer.writerow(['path', 'label', 'artifact_score', 'sharpness', 'contrast', 'brightness'])  # Header row
    
    next(reader)  # Skip header row in input CSV
    for row in reader:
        path, label, artifact_score = row[0], int(row[1]), float(row[2])
        sharpness, contrast, brightness = process_image(path)
        
        # Write all metadata to the new output CSV
        writer.writerow([path, label, artifact_score, sharpness, contrast, brightness])

print(f"Metadata saved to {output_csv}")
