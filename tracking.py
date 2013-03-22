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
    
    def __init__(self, bbox, contour, frame_number, frame_width, frame_height):
        self.bbox = bbox
        self.area = cv2.contourArea(contour)
        self.rotated_bbox = cv2.minAreaRect(contour) 
        
        # Keep track of "velocity"
        self.dx = 0
        self.dy = 0
        
        self.prev_area = 0
        
        self.frames_tracked = 1
        self.last_frame_tracked = frame_number
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.is_frozen = False
        
    @property
    def center(self):
        return self.bbox.center
    
    @property
    def angle(self):
        # raw angle seems inconsistent, can be 0 or -90 on same box
        raw_angle =self.rotated_bbox[2] 
        if raw_angle < -45:
            return raw_angle + 90
        else:
            return raw_angle
        
    def update(self, new_bbox, contour, frame_number):
        self.dx = self.bbox.center[0] - new_bbox.center[0]
        self.dy = self.bbox.center[1] - new_bbox.center[1]
        
        new_area = cv2.contourArea(contour)
        if self.dx < 2 and self.dy < 2 and new_area < self.area:
            # Object is probably stopping, and will fail to be segmented 
            # properly in future frames.  Freeze it. 
            self.is_frozen = True
        else:
            self.is_frozen = False
            
        if not self.is_frozen:
            self.bbox.update(new_bbox)
            self.prev_area = self.area
            self.area = new_area
            self.rotated_bbox = cv2.minAreaRect(contour)
        
        self.frames_tracked += 1
        self.last_frame_tracked = frame_number
    
    def delta_area(self):
        return self.area - self.prev_area
    
    def contains(self, other_obj):
        return self.bbox.contains_bbox(other_obj.bbox)
    
    def bbox_overlap_area(self, other_obj):
        return self.bbox.overlap_area(other_obj.bbox)
    
    def is_new(self):
        return self.frames_tracked < 5
    
    def is_leaving(self, border_thickness):
        if self.delta_area() >= 0:
            # Must be shrinking as it goes off screen.
            return False
        
        leaving_left = self.bbox.x0 < border_thickness and self.dx < 0
        leaving_right = (self.bbox.x1 > (self.frame_width - border_thickness) and
                         self.dx > 0)
        leaving_top = self.bbox.y0 < border_thickness and self.dy < 0
        leaving_bottom = (self.bbox.y1 > (self.frame_height - border_thickness) and
                          self.dy > 0)
        
        return leaving_left or leaving_right or leaving_top or leaving_bottom
    

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
    
    @property
    def area(self):
        return self.width * self.height

    def contains_point(self, point):
        return (point[0] >= self.x0 and point[0] <= self.x1 and
                point[1] >= self.y0 and point[1] <= self.y1)
        
    def contains_bbox(self, other_bbox):
        return (self.contains_point(other_bbox.top_left) and
                self.contains_point(other_bbox.bottom_right))
        
    def overlap_area(self, other_bbox):
        x_overlap = max(0, min(self.x1, other_bbox.x1) - max(self.x0, other_bbox.x0))
        y_overlap = max(0, min(self.y1, other_bbox.y1) - max(self.y0, other_bbox.y0))
        return x_overlap * y_overlap
        
    def update(self, bbox):
        self.x0 = bbox.x0
        self.y0 = bbox.y0
        self.width = bbox.width
        self.height = bbox.height
        

class ShapeFeatureTracker(object):
    
    def __init__(self):
        self.tracked_objects = []
        self.frame_number = 0
        
        self.count = 0
        
        # Thresholds for object similarity
        self.centroid_threshold = 40 # Euclidean distance
        self.area_threshold = 1500
        self.angle_threshold = 30 # in degrees

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
        return np.abs(obj1.angle - obj2.angle) < self.angle_threshold
    
    def is_match(self, obj1, obj2):
        return (self._is_centroid_match(obj1, obj2) and 
                self._is_area_match(obj1, obj2) and
                self._is_angle_match(obj1, obj2))
        
    def _prune_tracks(self):
        self._prune_short_tracks()
        self._prune_sub_tracks()
        self._prune_high_overlap_tracks()
        
    def _prune_short_tracks(self):
        obj_to_prune = []
        for obj in self.tracked_objects:
            if (obj.is_new() and 
                obj.last_frame_tracked < self.frame_number):
                obj_to_prune.append(obj)
                
        for obj in obj_to_prune:
            print "Pruned short track"
            self.tracked_objects.remove(obj)
            
    def _get_other_objects(self, obj):
        other_objs = list(self.tracked_objects)
        other_objs.remove(obj)
        return other_objs
            
    def _prune_sub_tracks(self):
        sub_tracks = []
        for obj in self.tracked_objects:
            for other_obj in self._get_other_objects(obj):
                if other_obj.contains(obj):
                    sub_tracks.append(obj)
                    break # inner loop
                
        for obj in sub_tracks:
            print "Pruned sub track"
            self.tracked_objects.remove(obj)
    
    def _prune_high_overlap_tracks(self):
        obj_to_prune = []
        for obj in self.tracked_objects:
            for other_obj in self._get_other_objects(obj):
                if other_obj.is_new():
                    continue

                if obj.bbox.area > other_obj.bbox.area:
                    # only prune the smaller objects
                    continue
                  
                if obj.bbox_overlap_area(other_obj) > 0.9 * obj.bbox.area:
                    # This object is mostly overlapped with another
                    obj_to_prune.append(obj)
                    break # inner loop
                
        for obj in obj_to_prune:
            print "Pruned high overlap object"
            self.tracked_objects.remove(obj)
    
    def _process_leaving_objects(self):
        leaving_objects = []
        for obj in self.tracked_objects:
            if not obj.is_new() and obj.is_leaving(5):
                leaving_objects.append(obj)
                
        for leaving_object in leaving_objects:
            self.tracked_objects.remove(leaving_object)
            self.count += 1
            print "Count: %d" % self.count
    
    def update(self, current_image, contours):
        self.frame_number += 1
        frame_height, frame_width = current_image.shape[:2]
        
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            bbox = BoundingBox(*bounding_rect)
            new_obj = TrackedObject(bbox, contour, self.frame_number, 
                                    frame_width, frame_height)
            
            matches = self._find_matching_objects(new_obj)
            
            if len(matches) == 0:
                # This is a new object
#                print "New object"
                self.tracked_objects.append(new_obj)
                    
            elif len(matches) == 1:
                # Update its location
#                print "Already tracked"
                matches[0].update(bbox, contour, self.frame_number)
            else:
                # Multiple possible matches - this is either overlapping fish
                # or a bad segmentation.
#                print "Multiple possible matches!"
                pass
            
        self._prune_tracks()
        
#        self._process_leaving_objects() # XXX not working very well yet
            
        self.draw_tracked_bounding_boxes(current_image.copy())
        
    