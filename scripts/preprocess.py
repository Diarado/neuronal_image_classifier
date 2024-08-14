from skimage import io, color
from skimage.filters import threshold_otsu
import numpy as np

def preprocess_image(image_path):
    image = io.imread(image_path)
    
    # Check if the image has more than one channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image  # The image is already grayscale
    
    # Binarization using Otsu's thresholding
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh
    
    return binary_image

def extract_features(image):
    # Extract features from the preprocessed image (e.g., density, contamination, peeling)
    # Here we'll use simple statistics as an example
    features = {
        'mean_intensity': np.mean(image),
        'std_intensity': np.std(image),
        'area': np.sum(image > 0)  # Count of non-zero pixels
    }
    
    return features



