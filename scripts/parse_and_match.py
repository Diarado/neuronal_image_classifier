import os
import pandas as pd

def load_scores(csv_path):
    df = pd.read_csv(csv_path, skiprows=4, header=[0, 1])
    df.columns = df.columns.map('_'.join).str.strip('_')
    return df

def match_image_to_score(image_name, df):
    plate_num = image_name.split('_')[-2]
    fld_num = image_name.split('fld')[-1]
    
    rep_folder = image_name.split('_')[-3]
    rep_key = f"{rep_folder}_{plate_num}"
    
    scores = df[(df['Unnamed: 1_level_0_'] == rep_key)]
    
    if scores.empty:
        return None  # If no match is found
    
    score_row = scores.iloc[0]  # Get the first match
    peeling_score = score_row[f'Peeling_Fld{fld_num}']
    contaminant_score = score_row[f'Contaminants_Fld{fld_num}']
    density_score = score_row[f'Cell Density_Fld{fld_num}']
    
    return peeling_score, contaminant_score, density_score

def preprocess_images(image_dir, csv_path):
    df = load_scores(csv_path)
    X, y = [], []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)
                scores = match_image_to_score(file, df)
                if scores:
                    X.append(preprocess_image(image_path))  # From preprocess.py
                    y.append(scores)
    
    return X, y
