from skimage import io, color, filters, morphology, measure, transform
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import pandas as pd
import glob

def detect_toxicity(image_path, target_size=(512, 512)):
    image = io.imread(image_path)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image) * 255  # Scale to [0, 255] if needed
    else:
        gray_image = image
    
    # Downsample and resize the image
    downsampled_image = transform.downscale_local_mean(gray_image, (2, 2))
    resized_image = transform.resize(downsampled_image, target_size, anti_aliasing=True)
    
    # Compute weighted average intensity
    weighted_avg_intensity = np.average(resized_image, weights=resized_image)
    
    print("image_path: ", image_path)
    print("Weighted average intensity:", weighted_avg_intensity)
    
    return weighted_avg_intensity

def _hasAxon(region) -> bool:
    # Function implementation remains unchanged
    pass

def _isFilled(region) -> bool:
    # Function implementation remains unchanged
    pass

if __name__ == "__main__":
    rounds = [
        'round02', 'round03', 'round04', 'round05', 'round06',
        'round07', 'round08', 'round09', 'round10', 'round11', 'round12'
    ]  

    for current_round in rounds:
        pre_image_dir = f'test/{current_round}_pre_images' 
        output_csv = f'toxicity_{current_round}.csv'
        res = []
        
        for root, dirs, files in os.walk(pre_image_dir):
            for file in files:
                if file.endswith('.tif'):
                    pre_image_path = os.path.join(root, file)
                    
                    try:
                        info = file.split('_')[-1]  # e.g., H07fld04.tif
                        plate_num = info.split('fld')[0]  # e.g., H07
                        fld_num_str = info.split('fld')[-1]  # e.g., 04.tif
                        fld_num = int(fld_num_str.split('.')[0]) # e.g., 04
                    except (IndexError, ValueError) as e:
                        print(f"Filename parsing error for {file}: {e}")
                        continue  # Skip files that don't match the expected pattern
                    
                    rep_folder = os.path.basename(root)  # e.g., rep1
                    plate_info = f"{rep_folder}_{plate_num}"  # e.g., rep1_H07
                        
                    # Detect toxicity for pre image
                    pre_total_intensity = detect_toxicity(pre_image_path)
                    
                    # Determine post image path
                    if current_round == 'round11':
                        post_image_path = pre_image_path.replace('pre_', '').replace('Pre', 'Post')
                    else:
                        search_dir = f"test/{current_round}_images/rep1/"
                        matching_files = glob.glob(os.path.join(search_dir, f"*{info}*"))
                
                        if matching_files:
                            post_image_path = matching_files[0]
                        else:
                            print(f"No matching post image found for plate_info: {info}")
                            continue
                    
                    if os.path.exists(post_image_path):
                        post_total_intensity = detect_toxicity(post_image_path)
                    else:
                        print(f"Post image not found for {pre_image_path}. Skipping this pair.")
                        continue  # Skip to the next file if post image is missing
                    
                    image_name = f'{plate_info}_fld{fld_num}'
                    row = {'Image': image_name, 'Killed': False}
                    
                    if pre_total_intensity * 0.9 > post_total_intensity:
                        row['Killed'] = True
                        
                    res.append(row)
                    
                    # Optional: Free up memory if processing large datasets
                    gc.collect()
    
        # Create a DataFrame from the results
        df = pd.DataFrame(res, columns=['Image', 'Killed'])
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_csv, index=False)
        
        print(f"Processing complete. Results saved to {output_csv}.")
    
    print('All processing complete.')
