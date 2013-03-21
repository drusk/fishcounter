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
    
    def __init__(self):
        self.tracked_objects = []
        
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

class BoundingBox(object):
    
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        
    @property
    def x1(self):
        return self.x0 + self.width
    
    @property
    def y1(self):
        return self.y0 + self.height
        
    @property
    def center(self):
        return (self.x0 + self.width / 2, self.y0 + self.height / 2)
    
    @property
    def top_left(self):
        return (self.x0, self.y0)
    
    @property
    def bottom_right(self):
        return (self.x1, self.y1)
    
    def contains_point(self, point):
        return (point[0] >= self.x0 and point[0] <= self.x1 and
                point[1] >= self.y0 and point[1] <= self.y1)
        
    def update(self, bbox):
        self.x0 = bbox.x0
        self.y0 = bbox.y0
        self.width = bbox.width
        self.height = bbox.height


class BoundingBoxTracker(object):
    
    def __init__(self):
        self.tracked_objects = []
        
    def draw_tracked_bounding_boxes(self, img):
        for bbox in self.tracked_objects:
            cv2.rectangle(img, bbox.top_left, bbox.bottom_right, (255, 0, 0))
        
        cv2.imshow("Tracker", img)
        
    def _find_matching_objects(self, bbox):
        matches = []
        for obj in self.tracked_objects:
            if obj.contains_point(bbox.center):
                matches.append(obj)
        return matches
        
    def update(self, current_image, contours):
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            bbox = BoundingBox(*bounding_rect)
            
            matches = self._find_matching_objects(bbox)
            
            if len(matches) == 0:
                # This is a new object
                print "New object"
                self.tracked_objects.append(bbox)
            elif len(matches) == 1:
                # Update its location
                print "Already tracked"
                matches[0].update(bbox)
            else:
                # Multiple possible matches - this is either overlapping fish
                # or a bad segmentation.
                print "Multiple possible matches!"
            
        self.draw_tracked_bounding_boxes(current_image.copy())

            
    