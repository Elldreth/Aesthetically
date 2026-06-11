import os
import shutil
from PIL import Image, PngImagePlugin


def find_and_copy_large_images(root_folder, target_folder, width=1024, height=1024):
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                source_file_path = os.path.join(foldername, filename)

                try:
                    with Image.open(source_file_path) as img:
                        img_width, img_height = img.size
                        if img_width >= width and img_height >= height:

                            # Extract metadata
                            metadata = img.info.get("parameters", "")
                            if metadata:
                                start_idx = 0
                                end_idx = metadata.find("Negative prompt:")
                                model_hash_idx = metadata.find("Model hash:")
                                model_hash_value = metadata[model_hash_idx:].split(",")[0].split(":")[1].strip()

                                extracted_metadata = metadata[start_idx:end_idx].strip()

                                # Create a folder for each unique Model hash
                                model_folder = os.path.join(target_folder, model_hash_value)
                                if not os.path.exists(model_folder):
                                    os.makedirs(model_folder)

                                target_file_path = os.path.join(model_folder, filename)

                                # Handle duplicate filenames
                                counter = 1
                                while os.path.exists(target_file_path):
                                    name, ext = os.path.splitext(filename)
                                    target_file_path = os.path.join(model_folder, f"{name}_{counter}{ext}")
                                    counter += 1

                                shutil.copy2(source_file_path, target_file_path)
                                print(f"Copied {source_file_path} to {target_file_path}")

                                # Create a .txt file with the same name as the image
                                txt_name, _ = os.path.splitext(os.path.basename(target_file_path))
                                txt_file_path = os.path.join(model_folder, f"{txt_name}.txt")
                                with open(txt_file_path, 'w') as f:
                                    f.write(extracted_metadata)

                except Exception as e:
                    print(f"Could not open {source_file_path}: {e}")


if __name__ == "__main__":
    target_folder = 'M:\\repos\\imghotornot\\found\\anime'
    root_folder = 'M:\\datasets\\anime_stuff\\holding\\2023-10-02'

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    find_and_copy_large_images(root_folder, target_folder)
