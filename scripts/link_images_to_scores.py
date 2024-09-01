import os
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.preprocess import preprocess_image
import os

def link_images_to_scores(image_dir, csv_dict):
    X, y = [], []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)
                # print(image_path)

                # Extracting useful information from the file name
                info = file.split('_')[-1]  # Extract useful info (e.g., H07fld04)
                plate_num = info.split('fld')[0]  # Extract plate number (e.g., H07)
                fld_num_str = info.split('fld')[-1]  # Extract field number with extension (e.g., 04.tif)
                fld_num = int(fld_num_str.split('.')[0]) - 1  # Remove extension, convert to zero-indexed integer

                rep_folder = os.path.basename(root)  # Extract replicate number (e.g., rep1)
                plate_info = f"{rep_folder}_{plate_num}"  # Combine to form plate_info (e.g., rep1_H07)

                if plate_info in csv_dict:
                    scores = csv_dict[plate_info][fld_num]

                    # Preprocess the image
                    processed_image, _, _, _= preprocess_image(image_path)

                    

                    X.append(processed_image)
                    y.append(scores)
                else:
                    print(f"Plate info {plate_info} not found in CSV dictionary.")
    print(X)
    # print(y)
    return X, y