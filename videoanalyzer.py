"""
"""

import cv2

from components import find_connected_components

class Analyzer(object):
    
    def __init__(self, segmenter, tracker):
        self.segmenter = segmenter
        self.tracker = tracker
    
    def analyze(self, previous_image, current_image):
        segmented = self.segmenter.segment(current_image)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        contours = find_connected_components(segmented, kernel, 200)
        
        self.tracker.update(current_image, contours)

