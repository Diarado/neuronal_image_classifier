from skimage import io, color, filters, morphology, measure
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import gc
from skimage import io, color, transform, img_as_float
import os
import pandas as pd

def preprocess_image(image_path, target_size=(512, 512), black_threshold=2500):
    """
    Preprocess the image, label regions, and count cell density.
    
    Parameters:
    - image_path (str): The file path to the image.
    - target_size (tuple): The target size for resizing the image.
    - black_threshold (float): The threshold below which the image is considered mostly black.
    
    Returns:
    - labeled_image (numpy array): The processed image with labeled regions.
    - is_dead (bool): True if the image is mostly black, indicating a dead cell.
    - cell_density (int): The count of desired neuron cells.
    """
    # Load the image
    image = io.imread(image_path)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    
    # Downsample the image before resizing
    downsampled_image = transform.downscale_local_mean(gray_image, (2, 2))
    resized_image = transform.resize(downsampled_image, target_size, anti_aliasing=True)

    mean_intensity = np.mean(resized_image)
    print("image_path: ", image_path)
    print("Mean intensity of the image:", mean_intensity)
    is_dead = mean_intensity < black_threshold
    if is_dead:
        print("Image is mostly black. Marking as dead cell.")
        dummy_labeled_image = np.zeros(target_size, dtype=np.uint8)
        gc.collect()
        return dummy_labeled_image, is_dead, 0, 0  # No cell density for dead images

    # Plot a histogram of the intensity values
    hist, bins = np.histogram(resized_image.ravel(), bins=256)
    
    # Find two peaks in the histogram
    peak1 = np.argmax(hist[:128])  # assuming contamination has lower intensity
    peak2 = np.argmax(hist[128:]) + 128  # assuming desired cells have higher intensity
    
    # Set threshold as the midpoint between the two peaks
    thresh = (bins[peak1] + bins[peak2]) / 2
    print("Computed threshold between peaks:", thresh)

    # Initialize the labeled image with 0 (background)
    labeled_image = np.zeros_like(resized_image, dtype=np.uint8)
    
    # Labeling regions based on intensity and area
    binary_image = resized_image > thresh - 16000

    labeled_regions = label(binary_image)
    regions = regionprops(labeled_regions, intensity_image=resized_image)
    
    cell_density = 0  
    is_round = True
    
    for region in regions:
        
        is_round = region.eccentricity < 0.8
        extent = region.extent > 0.5
        aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
        

        if region.mean_intensity > (thresh-500) and region.area > 10000 or (_isFilled(region) and region.area > 10000):
            labeled_image[region.coords[:, 0], region.coords[:, 1]] = 3  # Peeling
        elif 1 < region.area:
            if region.mean_intensity < thresh-12000 and region.area <= 300: # too dim, contamination automatically
                labeled_image[region.coords[:, 0], region.coords[:, 1]] = 1  # Contamination
            elif (is_round and extent and aspect_ratio < 2) or _hasAxon(region) or region.area > 300:  # if it has axon, then no matter what it's a neuron
                labeled_image[region.coords[:, 0], region.coords[:, 1]] = 2  # Desired neuron cells
                cell_density += 1
            else:
                labeled_image[region.coords[:, 0], region.coords[:, 1]] = 1  # Contamination
        else:
            labeled_image[region.coords[:, 0], region.coords[:, 1]] = 0  # Background
    
    # Debug: Check if the labeling is working
    print(f"Number of pixels labeled as Peeling (3): {np.sum(labeled_image == 3)}")
    print(f"Number of pixels labeled as Desired neuron cells (2): {np.sum(labeled_image == 2)}")
    print(f"Number of pixels labeled as Contamination (1): {np.sum(labeled_image == 1)}")
    print(f"Number of pixels labeled as Background (0): {np.sum(labeled_image == 0)}")
    peeling_degree = 3 # can be 1,2,3
    if np.sum(labeled_image == 3) > 20000:
        peeling_degree = 1
    elif 10000 < np.sum(labeled_image == 3) <= 20000:
        peeling_degree = 2

    # Save memory     
    del gray_image, downsampled_image, resized_image
    gc.collect()

    # Add this to visualize the labeled image, very helpful

    # plt.imshow(labeled_image, cmap='nipy_spectral')  # 'nipy_spectral' gives distinct colors to labels
    # plt.title(f"Labeled Image for {os.path.basename(image_path)}")
    # plt.colorbar()  
    # plt.show()


    return labeled_image, is_dead, peeling_degree, cell_density

from skimage import morphology, measure
import numpy as np

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
    min_spike_size = 5  

    # Filter out small regions that are not likely to be spikes
    spike_regions = [r for r in measure.regionprops(labeled_spikes) if r.area >= min_spike_size]
    
    # Count the number of significant spikes (regions that meet the criteria)
    num_significant_spikes = len(spike_regions)
    
    # a threshold for spikes
    spike_threshold = 10  

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
    solidity_threshold = 0.95  
    
    # Check if both extent and solidity are above their respective thresholds
    is_filled_region = (extent > extent_threshold) and (solidity > solidity_threshold)
    
    return is_filled_region
