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


class MixtureOfGaussiansBackgroundSubtractor(object):
    
    def __init__(self):
        history = 1000 # any value > 0; default 200
        num_gaussians = 5 # can be from 1 - 8; default 5
        background_ratio = 0.8 # any value > 0 and < 1; default 0.7
        noise_sigma = 0.5 # any value > 0; default 15
        self.background_subtractor = cv2.BackgroundSubtractorMOG(history,
                                            num_gaussians, background_ratio,
                                            noise_sigma)

    def segment(self, current_image):
        grayscale = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        learning_rate = -1 # > 0 and < 1; -1 for automatically calculated value
        return self.background_subtractor.apply(grayscale, None, learning_rate)


class CompositeSegmentationAlgorithm(object):
    """
    Performs logical AND operation on the segmentation results of the 
    component segmentation algorithms.
    """
    
    def __init__(self, algorithms):
        self.algorithms = algorithms
        
    def segment(self, current_image):
        overall_segmentation = None
        
        for algorithm in self.algorithms:
            segmentation = algorithm.segment(current_image)

            if overall_segmentation is None:
                overall_segmentation = segmentation
            else:
                overall_segmentation &= segmentation

        return overall_segmentation

