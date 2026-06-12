import os
from PIL import Image

# Specify the folder containing 'liked' images
liked_folder = os.path.abspath('./dislike')  # Replace with your actual 'liked' folder path

# Create YOLO bounding box text files for each 'liked' image
for img_name in os.listdir(liked_folder):
    if img_name.lower().endswith(('.png', '.jpg')):
        img_path = os.path.join(liked_folder, img_name)
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Calculate bounding box attributes
        class_idx = 0
        x_center = 0.5  # Covering the whole image, so the center is at 0.5
        y_center = 0.5  # Covering the whole image, so the center is at 0.5
        width = 1.0  # Covering the whole image, so width is 1.0
        height = 1.0  # Covering the whole image, so height is 1.0

        # Generate YOLO bounding box text
        yolo_bbox = f"{class_idx} {x_center} {y_center} {width} {height}"

        # Save this data into a text file
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(liked_folder, txt_name)
        with open(txt_path, 'w') as txt_file:
            txt_file.write(yolo_bbox)

print("YOLO bounding box text files have been generated for all 'liked' images.")
