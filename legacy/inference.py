import os
import shutil
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('.\\best.pt')  # Replace with your pre-trained model path

# Specify input and output folders
input_folder = 'M:\\datasets\\training\\regularization\\1_woman'  # Replace with your input image folder path
output_folder = '.\\liked_images'  # Replace with your output folder path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through images in folder
for img_name in os.listdir(input_folder):
    if img_name.endswith(('.jpg', '.png')):
        img_path = os.path.join(input_folder, img_name)

        # Run inference
        results = model.predict(img_path, conf=0.97, show=False)

        # If objects are detected, copy the file to the output folder
        if len(results[0].boxes) > 0:
            shutil.copy2(img_path, os.path.join(output_folder, img_name))
            print(f"Copied {img_name} to {output_folder}")

print("Inference and copying completed.")
