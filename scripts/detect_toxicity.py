# this script label images and only calculate dramatic drop in overall intensity to detect drug toxticity
# so, it's simpler version of preprocess.py

from skimage import io, color, filters, morphology, measure
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import gc
from skimage import io, color, transform
import os
import pandas as pd
import glob

def detect_toxicity(image_path, target_size=(512, 512)):
    image = io.imread(image_path)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    
    # Downsample and resize the image
    downsampled_image = transform.downscale_local_mean(gray_image, (2, 2))
    resized_image = transform.resize(downsampled_image, target_size, anti_aliasing=True)
    
    # Compute different brightness metrics
    # mean_intensity = np.mean(resized_image)
    # median_intensity = np.median(resized_image)
    weighted_avg_intensity = np.average(resized_image, weights=resized_image)
    
    # Compute a robust brightness metric as a combination of the above
    # combined_brightness = (mean_intensity + median_intensity + weighted_avg_intensity) / 3

    print("image_path: ", image_path)
    # print("Mean intensity:", mean_intensity)
    # print("Median intensity:", median_intensity)
    print("Weighted average intensity:", weighted_avg_intensity)

    return weighted_avg_intensity
    # # Plot a histogram of the intensity values
    # hist, bins = np.histogram(resized_image.ravel(), bins=256)
    
    # # Find two peaks in the histogram
    # peak1 = np.argmax(hist[:128])  # contamination has lower intensity
    # peak2 = np.argmax(hist[128:]) + 128  # desired cells have higher intensity
    
    # # Set threshold as the midpoint between the two peaks
    # thresh = (bins[peak1] + bins[peak2]) / 2
    # print("Computed threshold between peaks:", thresh)

    # # Initialize the labeled image with 0 (background)
    # labeled_image = np.zeros_like(resized_image, dtype=np.uint8)
    
    # # Labeling regions based on intensity and area
    # thresh_adj = 500
    # small_area_thresh = 100
    # mid_area_thresh = 400
    # large_area_thresh = 800
    # black_threshold = 2500
    # dim_thresh = thresh-12000
    # if round_num in [6]:
    #     binary_image = resized_image > thresh - 14000 # each round could be different
    #     # -20000 work on dimmer ones
        
    #     if mean_intensity < 4000:
    #         binary_image = resized_image > thresh - 22000
    # elif round_num in [1,2,3,4,7,8,11,12]:
    #     binary_image = resized_image > thresh - 24000 # TODO
    #     thresh_adj = -200 if round_num == 1 else 500
    #     dim_thresh = thresh-22000
    #     if round_num == 8:
    #         black_threshold = 1200 # or even lower
    # else: # round 9 and 10
    #     binary_image = resized_image > thresh - 9000 # TODO
    #     thresh_adj = 500
    #     black_threshold = 900

    # is_dead = mean_intensity < black_threshold
    # if is_dead:
    #     print("Image is mostly black. Marking as dead cell.")
    #     gc.collect()
    #     return 0, 0  # No cell density for dead images

    # labeled_regions = label(binary_image)
    # regions = regionprops(labeled_regions, intensity_image=resized_image)
    
    # cell_density = 0  
    # is_round = True
    # good_cell_cnt = 0
    # good_cell_pixels = 0
    # bad_cell_cnt = 0

    # for region in regions:
        
    #     is_round = region.eccentricity < 0.8
    #     extent = region.extent > 0.5
    #     aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
        

    #     if region.mean_intensity > (thresh - thresh_adj) and region.area > 10000 or (_isFilled(region) and region.area > 10000):
    #         if region.area < 512*512*0.9:
    #             labeled_image[region.coords[:, 0], region.coords[:, 1]] = 3  # Peeling
    #         else:
    #             print("Image is mostly black. Marking as dead cell.")
    #             print("case 2")
    #             gc.collect()
    #             return 0, 0 # No cell density for dead images

    #     elif 1 < region.area:
    #         if region.mean_intensity < dim_thresh and 12 < region.area <= 100 : # too dim, contamination automatically
    #             labeled_image[region.coords[:, 0], region.coords[:, 1]] = 1  # Contamination
    #             bad_cell_cnt += 1
    #         elif (
    #             (is_round and extent and aspect_ratio < 2) or _hasAxon(region) or # if it has axon, then no matter what it's a neuron
    #             (region.area > small_area_thresh and aspect_ratio < 3) or 
    #             region.area > large_area_thresh or
    #             (region.area > mid_area_thresh and region.extent < 0.5)):  # multiple neurons linked together by thin axons
    #             labeled_image[region.coords[:, 0], region.coords[:, 1]] = 2  # Desired neuron cells
    #             good_cell_cnt += 1
    #             good_cell_pixels += region.area
    #         else:
    #             if region.area > 12:
    #                 if aspect_ratio > 6: # likely a long axon
    #                     labeled_image[region.coords[:, 0], region.coords[:, 1]] = 2  # Desired neuron cells
    #                     good_cell_cnt += 1
    #                 else:
    #                     labeled_image[region.coords[:, 0], region.coords[:, 1]] = 1  # Contamination
    #                     bad_cell_cnt += 1
                        
    #             else:
    #                 labeled_image[region.coords[:, 0], region.coords[:, 1]] = 0 # too small, not contamination

    #     else:
    #         labeled_image[region.coords[:, 0], region.coords[:, 1]] = 0  # Background

    # alpha = 0.8  # Weight for the area
    # beta = 100   # Weight for the count
    # cell_density = (alpha * good_cell_pixels + beta * good_cell_cnt)
   
    # print(f"good_cell_cnt: {(good_cell_cnt)}")
    # print(f"density: {(cell_density)}")
    
    # # Save memory     
    # del gray_image, downsampled_image
    # gc.collect()

    # # Add this to visualize the labeled image, very helpful
    # plt.imshow(labeled_image, cmap='nipy_spectral')  # 'nipy_spectral' gives distinct colors to labels
    # plt.title(f"Labeled Image for {os.path.basename(image_path)}")
    # plt.colorbar()  
    # plt.show()
    # # save to test_label_imgs
    # save_folder = "test_label_imgs"
    # save_path = os.path.join(save_folder, f"{os.path.basename(image_path)}_labeled.png")
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.clf()

    # return good_cell_cnt, total_intensity


def _hasAxon(region) -> bool:
    """
    Helper function to determine whether a region has spikes coming out of it (axon).
    This version is adapted to work with a skimage.measure.RegionProperties object.
    
    We examine the region's binary mask, looking for small protrusions (spikes) by:
    1. Extracting the perimeter.
    2. Identifying small, thin structures as spikes.
    """
    # Extract the region's binary mask
    binary_mask = region.image
    
    # a morphological gradient using XOR to emphasize the boundaries (where spikes are expected)
    gradient = np.bitwise_xor(morphology.dilation(binary_mask), morphology.erosion(binary_mask))
    
    # Label the potential spikes in the gradient image
    labeled_spikes, num_spikes = measure.label(gradient, return_num=True, connectivity=1)
    
    # Define a minimum size for what we consider a spike (tiny protrusions)
    min_spike_size = 2 

    # Filter out small regions that are not likely to be spikes
    spike_regions = [r for r in measure.regionprops(labeled_spikes) if r.area >= min_spike_size]
    
    # Count the number of significant spikes (regions that meet the criteria)
    num_significant_spikes = len(spike_regions)
    
    # a threshold for spikes
    spike_threshold = 5  

    # Check if there are many spikes
    if num_significant_spikes > spike_threshold:
        return True
    return False

def _isFilled(region) -> bool:
    """
    Check if the region is filled, meaning it takes up most of its bounding box
    and doesn't have significant holes or gaps.
    
    Parameters:
    - region: skimage.measure._regionprops.RegionProperties
    
    Returns:
    - bool: True if the region is filled, otherwise False.
    """
    # Get the bounding box dimensions
    min_row, min_col, max_row, max_col = region.bbox
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col
    
    # Calculate extent: ratio of the region's area to its bounding box area
    extent = region.area / (bbox_height * bbox_width)
    
    # Calculate solidity: ratio of the region's area to its convex hull area
    solidity = region.solidity
    
    # Define thresholds for extent and solidity to consider the region as filled
    extent_threshold = 0.7 
    solidity_threshold = 0.85  
    
    # Check if both extent and solidity are above their respective thresholds
    is_filled_region = (extent > extent_threshold) and (solidity > solidity_threshold)
    
    return is_filled_region

if __name__ == "__main__":
    import os
    import pandas as pd  # Ensure pandas is imported as pd
    import gc

    rounds = ['round02', 'round03', 'round04', 'round05', 'round06', 'round07', 'round08', 'round09', 'round10', 'round11', 'round12']  
    pre_image_dir_lst = [f'test/{round}_pre_images' for round in rounds]
    post_image_dir_lst = [f'test/{round}_images' for round in rounds] 
    
    for round in rounds:
        # Define the output CSV file
        output_csv = f'toxicity_{round}.csv'
        
        # Initialize a list to store results
        res = []
        
        # Iterate over paired pre and post image directories
        for pre_image_dir, post_image_dir in zip(pre_image_dir_lst, post_image_dir_lst):
            # Walk through the pre_image_dir
            for root, dirs, files in os.walk(pre_image_dir):
                for file in files:
                    if file.endswith('.tif'):
                        pre_image_path = os.path.join(root, file)

                        # Extract useful information from the file name
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
                        
                        # Construct the corresponding post image path
                        post_image_path = ''
                        if round in ['round11']:
                            post_image_path = pre_image_path.replace('pre_', '').replace('Pre', 'Post')
                        else:
                            search_dir = f"test/{round}_pre_images/rep1/"
                            matching_files = glob.glob(os.path.join(search_dir, f"*{info}*"))
              
                            if matching_files:
                                # Assuming you take the first match if there are multiple matches
                                post_image_path = matching_files[0]
                            else:
                                print(f"No matching file found for plate_info: {info}")
                                continue

                        # print(post_image_path)
                        # Check if the post image exists
                        
                        if os.path.exists(post_image_path):
                            post_total_intensity = detect_toxicity(post_image_path)
                        else:
                            print(f"Post image not found for {pre_image_path}. Skipping this pair.")
                            continue  # Skip to the next file if post image is missing
                        
                        # Create a unique image name based on plate information and field number
                        image_name = f'{plate_info}_fld{fld_num}'
                        
                        # Initialize the row with default values
                        row = {'Image': image_name, 'Killed': False}
                        
                        # Determine if the well is killed based on the defined criteria
                        # if ((pre_num_neuron / 2 > post_num_neuron) or (pre_total_intensity / 2 > post_total_intensity) or 
                        #     (pre_num_neuron != 0 and post_num_neuron == 0) or 
                        #     (pre_total_intensity != 0 and post_total_intensity == 0)):
                        #     row['Killed'] = True  # Mark as killed if criteria are met
                        if pre_total_intensity * 0.95 > post_total_intensity:
                            row['Killed'] = True
                        # Append the result to the list
                        res.append(row)
                        
                        # Optional: Free up memory if processing large datasets
                        gc.collect()
        
        # Create a DataFrame from the results
        df = pd.DataFrame(res, columns=['Image', 'Killed'])
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_csv, index=False)
        
        print(f"Processing complete. Results saved to {output_csv}.")