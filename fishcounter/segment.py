# Copyright (C) 2013 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Segmentation algorithms.
"""

import cv2
import numpy as np

MAX_PIXEL_VALUE = 255

class HSVColourSegmenter(object):
    
    def __init__(self):
        # IMPORTANT NOTE:
        #   Hue range:        [0, 180]
        #   Saturation range: [0, 255]
        #   Value range:      [0, 255]
        # This is not the same as programs like gcolor2!
        # TODO an algorithm to dynamically set thresholds
        self.hue_min = 75
        self.hue_max = 100
        self.sat_min = 0.
        self.sat_max = 100
        self.val_min = 100.
        self.val_max = 255.
        
        # For cleaning
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def segment(self, current_image):
        hsv_img = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV) 

        min_vals = np.array([self.hue_min, self.sat_min, self.val_min])
        max_vals = np.array([self.sat_max, self.sat_max, self.val_max])
        bin_img = cv2.inRange(hsv_img, min_vals, max_vals)
        
        return self._clean(bin_img)
        
    def _clean(self, bin_img):
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, self.kernel, 
                                   iterations=17)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, self.kernel, 
                                   iterations=3)
        return bin_img
    

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

