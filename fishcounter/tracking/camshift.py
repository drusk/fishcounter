"""
Tracking based on the Camshift algorithm.
"""

import cv2
import numpy as np

from fishcounter.segment import HSVColourSegmenter

class CamShiftTracker(object):
    """
    Uses colour information to track fish regardless of whether they are 
    moving or not.
    """
    
    def __init__(self):
        self.tracked_objects = []
        self.mask_detector = HSVColourSegmenter()
        
    def track(self, objects):
        self.tracked_objects.extend(objects)

    def update(self, current_image, moving_objects, stationary_objects):
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        mask = self.mask_detector.segment(current_image)
        
        for obj in stationary_objects:
            bbox = obj.bbox
            
            if bbox.has_negative_area:
                print "BBOX has negative area: %s" % bbox
                continue
            
            hsv_roi = hsv[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
            mask_roi = mask[bbox.y0:bbox.y1, bbox.x0:bbox.x1]

            bin_range = [self.mask_detector.hue_min, self.mask_detector.hue_max]
            hist = cv2.calcHist([hsv_roi], # source image(s)
                                [0], # channels to use - just Hue
                                mask_roi, # mask which source pixels to count
                                [16], # number of bins
                                bin_range # first bin min, last bin max
                                )
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            hist = hist.reshape(-1)
            
            prob = cv2.calcBackProject([hsv], # input image
                                       [0], # channels to use - just Hue
                                       hist, # histogram
                                       bin_range, # first bin min, last bin max
                                       1 # scale factor
                                       )
            prob &= mask
            
            stop_criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 
                             10, # max iterationsstop when window center shifts less than this distance
                             1 # desired accuracy of window center 
                             )

            # what is the difference between track_box and track_window?
            track_box, track_window = cv2.CamShift(prob, bbox.cv2rect, stop_criteria)
            
            prev_center = bbox.center
            bbox.update(track_window)
            new_center = bbox.center
            
            displacement = np.sqrt(np.square(prev_center[0] - new_center[0]) + 
                                   np.square(prev_center[1] - new_center[1]))
            
            if displacement > 4:
                stationary_objects.remove(obj)
                moving_objects.append(obj)

        return moving_objects, stationary_objects
    
