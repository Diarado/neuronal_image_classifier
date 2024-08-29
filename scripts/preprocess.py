from skimage import io, color, filters, morphology
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
        return dummy_labeled_image, is_dead, 0  # No cell density for dead images

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
    binary_image = resized_image > thresh
    labeled_regions = label(binary_image)
    regions = regionprops(labeled_regions, intensity_image=resized_image)
    
    cell_density = 0  # Initialize cell density count

    for region in regions:
        if region.mean_intensity > thresh * 1.5 and region.area > 3000:
            labeled_image[region.coords[:, 0], region.coords[:, 1]] = 3  # Peeling
        elif region.mean_intensity > thresh and 1 < region.area:
            labeled_image[region.coords[:, 0], region.coords[:, 1]] = 2  # Desired neuron cells
            cell_density += 1  # Count this region as part of the cell density
        elif region.mean_intensity < thresh:
            labeled_image[region.coords[:, 0], region.coords[:, 1]] = 1  # Contamination
        else:
            labeled_image[region.coords[:, 0], region.coords[:, 1]] = 0  # Background
    
    # Debug: Check if the labeling is working
    print(f"Number of pixels labeled as Peeling (3): {np.sum(labeled_image == 3)}")
    print(f"Number of pixels labeled as Desired neuron cells (2): {np.sum(labeled_image == 2)}")
    print(f"Number of pixels labeled as Contamination (1): {np.sum(labeled_image == 1)}")
    print(f"Number of pixels labeled as Background (0): {np.sum(labeled_image == 0)}")
    
    # Save memory     
    del gray_image, downsampled_image, resized_image
    gc.collect()
    # save_image_to_csv(labeled_image, "labeled_imgs.csv", image_path)


    # Add this to visualize the labeled image, very helpful

    # plt.imshow(labeled_image, cmap='nipy_spectral')  # 'nipy_spectral' gives distinct colors to labels
    # plt.title(f"Labeled Image for {os.path.basename(image_path)}")
    # plt.show()

    return labeled_image, is_dead, cell_density


def save_image_to_csv(labeled_image, csv_path, image_path):
    """
    Save the labeled_image to a CSV file, appending if the file already exists.

    Parameters:
    - labeled_image (numpy array): The labeled_image to save.
    - csv_path (str): The path to the CSV file.
    - image_path (str): The original image path to identify the image in the CSV.
    """

    # Flatten the labeled_image to 1D
    flattened_image = labeled_image.flatten()

    # Prepare a DataFrame for saving
    df = pd.DataFrame([flattened_image])
    df.insert(0, 'image_path', image_path)  # Include image path for reference

    # Append to CSV file
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"Labeled image saved to {csv_path}")
