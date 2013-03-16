"""
Segmentation algorithms.
"""

import cv2
import numpy as np

MAX_PIXEL_VALUE = 255

class MovingAverageBackgroundSubtractor(object):
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.background = None
    
    def segment(self, current_image):
        grayscale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

        if self.background is None:
            self.background = np.zeros(np.shape(grayscale))
            
        self.background = ((1 - self.alpha) * self.background + 
                           self.alpha * grayscale)
        
        moving_pixels = np.abs(self.background - grayscale)
        moving_pixels = moving_pixels.astype(np.uint8)
        
        # when otsu flag specified, the passed in threshold is not used
        threshold, _ = cv2.threshold(moving_pixels, -1, MAX_PIXEL_VALUE, 
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        _, segmented = cv2.threshold(moving_pixels, threshold, MAX_PIXEL_VALUE, 
                                     cv2.THRESH_BINARY)
        return segmented

