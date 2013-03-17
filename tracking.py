"""
Tracking algorithms.
"""

import cv2
import numpy as np

class CamShiftTracker(object):
    """
    Example:
    https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python2/camshift.py?rev=6588
    """
    
    def update(self, current_image, contours):
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), 
                           np.array((180., 255., 255.)))
        
        debug_img = current_image.copy()
        
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            x1, y1, width, height = bounding_rect
            
            x2 = x1 + width
            y2 = y1 + height
            
            hsv_roi = hsv[y1:y2, x1:x2]
            mask_roi = mask[y1:y2, x1:x2]

            hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            hist = hist.reshape(-1)
            
            prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
            prob &= mask
            
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            track_box, track_window = cv2.CamShift(prob, bounding_rect, term_crit)
            tx1, ty1, twidth, theight = track_window
            
            # what is the difference between track_box and track_window?
            
            cv2.rectangle(debug_img, (tx1, ty1), (tx1 + twidth, ty1 + theight),
                           (255, 0, 0))
            
        cv2.imshow("Tracker", debug_img)
            
    