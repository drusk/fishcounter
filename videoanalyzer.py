"""
"""

import cv2

from components import find_connected_components

class Analyzer(object):
    
    def __init__(self, segmenter):
        self.segmenter = segmenter
    
    def analyze(self, previous_image, current_image):
        segmented = self.segmenter.segment(current_image)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected_components = find_connected_components(segmented, kernel, 100)
        
        cv2.imshow("Final segmentation", connected_components)

