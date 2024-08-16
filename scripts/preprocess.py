from skimage import io, color, filters, morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize

def preprocess_image(image_path, target_size=(512, 512)):
    # set the size
    image = io.imread(image_path)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    
    # Resize image
    resized_image = resize(gray_image, target_size, anti_aliasing=True)
    
    # Apply Otsu's thresholding
    thresh = threshold_otsu(resized_image)
    binary_image = resized_image > thresh
    
    # Calculate intensity threshold for contamination
    contamination_thresh = filters.threshold_otsu(resized_image)
    
    # Label regions
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image, intensity_image=resized_image)
    
    # Analyzing regions
    for region in regions:
        # High-intensity regions could be peeling
        if region.mean_intensity > contamination_thresh * 1.5 and region.area > 100:
            # Mark as peeling
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 2  # Example label for peeling
        elif region.mean_intensity > contamination_thresh:
            # Mark as normal neuron cells
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 1  # Example label for normal cells
        else:
            # Mark as contamination
            binary_image[region.coords[:, 0], region.coords[:, 1]] = 0  # Example label for contamination

    return binary_image



