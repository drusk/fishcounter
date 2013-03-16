"""
"""

import cv2

class Analyzer(object):
    
    def __init__(self, segmenter):
        self.segmenter = segmenter
    
    def analyze(self, previous_image, current_image):
        segmented = self.segmenter.segment(current_image)
        cv2.imshow("Segmented", segmented)
        