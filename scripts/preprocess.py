from skimage import io, color, filters, morphology
from skimage.measure import label, regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import gc
from skimage import io, color, transform, img_as_float

def preprocess_image(image_path, target_size=(512, 512), black_threshold=0.01):
    """
    Preprocess the image and check for mostly black images.
    
    Parameters:
    - image_path (str): The file path to the image.
    - target_size (tuple): The target size for resizing the image.
    - black_threshold (float): The threshold below which the image is considered mostly black.

    Returns:
    - binary_image (numpy array): The processed image after thresholding, or a dummy binary image if dead.
    - is_dead (bool): True if the image is mostly black, indicating a dead cell.
    """
    # Load the image
    image = io.imread(image_path)
    print("Image loaded successfully. Shape:", image.shape)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    print("Converted to grayscale.")
    
    # Downsample the image before resizing
    downsampled_image = transform.downscale_local_mean(gray_image, (2, 2))
    print("Image downsampled. Shape:", downsampled_image.shape)

    resized_image = transform.resize(downsampled_image, target_size, anti_aliasing=True)
    print("Image resized.")

    mean_intensity = np.mean(resized_image)
    print("Mean intensity of the image:", mean_intensity)

    is_dead = mean_intensity < black_threshold
    if is_dead:
        print("Image is mostly black. Marking as dead cell.")
        dummy_binary_image = np.zeros(target_size, dtype=np.uint8)
        gc.collect()
        return dummy_binary_image, is_dead

    # Plot a histogram of the intensity values
    plt.figure(figsize=(8, 6))
    hist, bins = np.histogram(resized_image.ravel(), bins=256)
    # plt.plot(bins[:-1], hist, color='black')
    # #plt.xlim(1, 2) 
    # plt.title("Histogram of Image Intensities")
    # plt.xlabel("Intensity")
    # plt.ylabel("Count")
    # plt.show(block=False)

    # Find two peaks in the histogram
    peak1 = np.argmax(hist[:128])  # assuming contamination has lower intensity
    peak2 = np.argmax(hist[128:]) + 128  # assuming desired cells have higher intensity

    print("Peak 1 (contamination):", bins[peak1])
    print("Peak 2 (desired cells):", bins[peak2])

    # Set threshold as the midpoint between the two peaks
    thresh = (bins[peak1] + bins[peak2]) / 2
    print("Computed threshold between peaks:", thresh)

    binary_image = resized_image > thresh

    # Additional processing (e.g., labeling regions)
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image, intensity_image=resized_image)
    
    print("Regions identified:", len(regions))

    for region in regions:
        if region.mean_intensity > thresh * 1.5 and region.area > 100:
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 2
        elif region.mean_intensity > thresh:
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 1
        else:
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 0

    plt.close()      
    del gray_image, downsampled_image, resized_image
    gc.collect()
    return binary_image, is_dead

