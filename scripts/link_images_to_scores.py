import os
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.preprocess import preprocess_image

def link_images_to_scores(image_dir, csv_dict):
    X, y = [], []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)
                plate_num = file.split('_')[-2]  # Extract plate number (e.g., A01)
                fld_num = int(file.split('fld')[-1]) - 1  # Extract field number (Fld index starting from 0)

                rep_folder = os.path.basename(root)  # Extract replicate number (e.g., rep1)
                plate_info = f"{rep_folder}_{plate_num}"

                if plate_info in csv_dict:
                    scores = csv_dict[plate_info][fld_num]
                    X.append(preprocess_image(image_path))
                    y.append(scores)

    return X, y

