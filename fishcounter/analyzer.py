"""
Analyzers coordinate the analysis performed by segmentation algorithms, 
tracking algorithms, display, etc.
"""

import cv2

from components import find_connected_components

class Analyzer(object):
    
    def __init__(self, segmenter, tracker, display):
        self.segmenter = segmenter
        self.tracker = tracker
        self.display = display
    
    def analyze(self, previous_image, current_image):
        segmented = self.segmenter.segment(current_image)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        contours = find_connected_components(segmented, kernel, 200)
        
        self.tracker.update(current_image, contours)

        self._display_findings(current_image)
        
    def _display_findings(self, current_image):
        self.display.new_frame(current_image)
        self.display.draw_bounding_boxes(self.tracker.potential_objects,
                                         (0, 255, 255))
        self.display.draw_bounding_boxes(self.tracker.moving_objects, 
                                         (255, 0, 0))
        self.display.draw_bounding_boxes(self.tracker.stationary_objects, 
                                         (0, 0, 255))
        self.display.draw_counter(self.tracker.count)

        self.display.display_frame()

