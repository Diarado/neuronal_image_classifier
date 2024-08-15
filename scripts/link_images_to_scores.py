import os
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.preprocess import preprocess_image

def link_images_to_scores(image_dir, csv_dict):
    """
    This function links images to their corresponding scores based on a CSV file.

    Parameters:
    - image_dir (str): The directory containing subdirectories of images (replicates).
    - csv_dict (dict): A dictionary where keys are plate_info and values are lists of lists containing scores.

    Returns:
    - X (list): A list of preprocessed images. Each entry corresponds to the image data ready for model input.
    - y (list): A list of labels/scores. Each entry corresponds to the scores for an image, containing Peeling, Contaminants, Cell Density, and Empty/Dead.
    """
    
    X, y = [], []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)
                # print(file)
                info = file.split('_')[-1]  # useful info, like H07fld04 
                plate_num = info.split('fld')[0]# Extract plate number (e.g., H07)
                # print(plate_num)
                fld_num_str = info.split('fld')[-1]  # Extract field number with extension (e.g., 01.tif)
                fld_num = int(fld_num_str.split('.')[0]) - 1  # Remove extension and convert to zero-indexed integer

                rep_folder = os.path.basename(root)  # Extract replicate number (e.g., rep1)
                plate_info = f"{rep_folder}_{plate_num}"
                # print(plate_info)

                if plate_info in csv_dict:
                    scores = csv_dict[plate_info][fld_num]
                    X.append(preprocess_image(image_path))
                    y.append(scores)
    
    return X, y
