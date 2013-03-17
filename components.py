"""
Find connected components.
"""

import cv2

def find_connected_components(bin_img, kernel, length_thresh):
    # clean up image
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=15)
    
    cv2.imshow("Segmentation - post processed", bin_img)
    
    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    large_contours = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length > length_thresh:
            large_contours.append(contour)
    
    return large_contours
