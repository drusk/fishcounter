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


class TrackedObject(object):
    
    def __init__(self, bbox, contour):
        self.bbox = bbox
        self.area = cv2.contourArea(contour)
        self.rotated_bbox = cv2.minAreaRect(contour) 
        
        # Keep track of "velocity"
        self.dx = 0
        self.dy = 0
        
    @property
    def center(self):
        return self.bbox.center
    
    @property
    def angle(self):
        # XXX inconsistent, seems to be 0 or -90 on same box
        return self.rotated_bbox[2]
        
    def update(self, new_bbox, contour):
        self.dx = self.bbox.center[0] - new_bbox.center[0]
        self.dy = self.bbox.center[1] - new_bbox.center[1]
        
        self.bbox.update(new_bbox)
        self.area = cv2.contourArea(contour)
        
        self.rotated_bbox = cv2.minAreaRect(contour)
        

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
        

class ShapeFeatureTracker(object):
    
    def __init__(self):
        self.tracked_objects = []
        
        # Thresholds for object similarity
        self.centroid_threshold = 40 # Euclidean distance
        self.area_threshold = 1500
        self.angle_threshold = 30 # in degrees

    @property
    def count(self):
        return len(self.tracked_objects)
        
    def draw_tracked_bounding_boxes(self, img):
        for obj in self.tracked_objects:
            cv2.rectangle(img, obj.bbox.top_left, obj.bbox.bottom_right, (255, 0, 0))
#            rbbox = obj.rotated_bbox
#            points = cv2.cv.BoxPoints(rbbox)         # Find four vertices of rectangle from above rect
#            cv2.polylines(img,np.array([points]),True,(0,0,255),2)# draw rectangle in blue color
        
        cv2.imshow("Tracker", img)
        
    def _find_matching_objects(self, new_obj):
        matches = []
        for obj in self.tracked_objects:
            if self.is_match(new_obj, obj):
                matches.append(obj)
        return matches
    
    def _is_centroid_match(self, obj1, obj2):
        return (np.sqrt(np.square(obj1.center[0] - obj2.center[0]) + 
                        np.square(obj1.center[1] - obj2.center[1]))
                < self.centroid_threshold) 
    
    def _is_area_match(self, obj1, obj2):
        return np.abs(obj1.area - obj2.area) < self.area_threshold
    
    def _is_angle_match(self, obj1, obj2):
        print "Angles: %f %f" % (obj1.angle, obj2.angle)
        return np.abs(obj1.angle - obj2.angle) < self.angle_threshold
    
    def is_match(self, obj1, obj2):
        return (self._is_centroid_match(obj1, obj2) and 
                self._is_area_match(obj1, obj2))
        
    def update(self, current_image, contours):
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            bbox = BoundingBox(*bounding_rect)
            new_obj = TrackedObject(bbox, contour)
            
            matches = self._find_matching_objects(new_obj)
            
            if len(matches) == 0:
                # This is a new object
                print "New object"
                self.tracked_objects.append(new_obj)
                    
            elif len(matches) == 1:
                # Update its location
                print "Already tracked"
                matches[0].update(bbox, contour)
            else:
                # Multiple possible matches - this is either overlapping fish
                # or a bad segmentation.
                print "Multiple possible matches!"
            
        self.draw_tracked_bounding_boxes(current_image.copy())
        print "Count %d" % self.count
            
    