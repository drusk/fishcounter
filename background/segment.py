"""
Image segmentation algorithms.
"""

import cv2
import numpy as np

MAX_PIXEL_VAL = 255

def segment_by_velocity(flow, thresh_low, thresh_high, min_frame_portion=0.025):
    flow_mag = np.sum(np.fabs(flow), axis=2)
    
    segmented_flow = hysteresis_thresh(flow_mag, thresh_low, thresh_high)
    
    # Apply morphological closing to combine pieces of the same object
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (35, 35))
    # segmented_flow = cv2.morphologyEx(segmented_flow, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(segmented_flow, cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    frameshape = np.shape(flow)
    framesize = frameshape[0] * frameshape[1]
    
    large_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_frame_portion * framesize:
            large_contours.append(contour)
        
    return large_contours

def hysteresis_thresh(img, thresh_low, thresh_high):
    # Initial segmentation based on the high threshold.  We have high 
    # confidence these are part of the object.
    _, segmented = cv2.threshold(img, thresh_high, MAX_PIXEL_VAL, 
                                 cv2.THRESH_BINARY)
    segmented = segmented.astype(np.uint8)
    
    # Next iteratively add pixels which are above the low threshold but 
    # only if they are connected to the already segmented parts.
    prev_segmented = segmented.copy()

    has_changed = True
    while has_changed:
        for (row, col) in get_outer_border_pts(segmented):
            if img[row, col] > thresh_low:
                segmented[row, col] = MAX_PIXEL_VAL
            
        if (segmented == prev_segmented).all():
            has_changed = False
            
        prev_segmented[:] = segmented

    return segmented
    
def get_outer_border_pts(bin_img):
    # Dilate the objects by 1 pixel
    dilated = cv2.dilate(bin_img, 
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # The contour of the dilated image is the outer border of the original.
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    border_img = np.zeros_like(bin_img)
    
    cv2.drawContours(border_img, contours, -1, MAX_PIXEL_VAL)
    (rows, cols) = np.where(border_img == MAX_PIXEL_VAL)
    
    return zip(rows, cols)
