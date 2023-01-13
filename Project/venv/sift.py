from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from numpy import arctan2, pi, dstack, uint8
import numpy as np
from utils import rgb2gray

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cdist

def orb(img):
    # Define the Harris corner detector kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Compute the Harris corner response
    Ix = convolve(img, np.array([[1, 0, -1]]))
    Iy = convolve(img, np.array([[1], [0], [-1]]))
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    R = convolve(Ixx, kernel) * convolve(Iyy, kernel) - (convolve(Ixy, kernel) ** 2)
    
    # Extract the keypoints
    keypoints = np.column_stack(np.where(R > R.max() * 0.1))

    # Extract the BRIEF descriptor
    patch_size = 32
    patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
    points = np.random.randint(0, patch_size, size=(256, 2))
    descriptor = []
    for kp in keypoints:
        patch = img[kp[0]-patch_size//2:kp[0]+patch_size//2, kp[1]-patch_size//2:kp[1]+patch_size//2]
        descriptor.append((patch[points[:, 0], points[:, 1]] > patch[points[:, 2], points[:, 3]]).astype(np.uint8))
    descriptor = np.array(descriptor)
    return keypoints, descriptor

import numpy as np
from scipy.spatial import cKDTree

def match_features(desc1, desc2, r_threshold=1e-5):
    # Build KD-Tree from second feature set
    kdt = cKDTree(desc2)
    
    # For each feature in desc1, find the closest feature in desc2
    # using the KD-Tree
    _, indices = kdt.query(desc1, distance_upper_bound=r_threshold)
    
    # Filter out any invalid matches (indices of -1)
    indices = indices[indices != -1]
    
    return indices