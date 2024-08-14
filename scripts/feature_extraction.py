from skimage.measure import label, regionprops
import numpy as np

def extract_features(binary_image):
    labeled_image = label(binary_image)
    properties = regionprops(labeled_image)
    
    feature_dict = {
        "Area": [],
        "Perimeter": [],
        "Eccentricity": [],
        "Intensity": []
    }
    
    for prop in properties:
        feature_dict["Area"].append(prop.area)
        feature_dict["Perimeter"].append(prop.perimeter)
        feature_dict["Eccentricity"].append(prop.eccentricity)
        feature_dict["Intensity"].append(np.mean(binary_image[prop.coords[:, 0], prop.coords[:, 1]]))
    
    return feature_dict
