"""
Find connected components.
"""

import cv2
import numpy as np

def find_connected_components(bin_img, kernel, length_thresh):
    # clean up image
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=5)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=10)
    
    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    large_contours = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length > length_thresh:
            large_contours.append(contour)
    
    hulls = [cv2.convexHull(contour) for contour in large_contours]
    
    enhanced_img = np.zeros(np.shape(bin_img), dtype=np.uint8)
    
    index = -1 # -1 draws all
    colour = 255
    thickness = -1 # negative values mean draw filled
    cv2.drawContours(enhanced_img, hulls, index, colour, thickness)
    
    return enhanced_img
