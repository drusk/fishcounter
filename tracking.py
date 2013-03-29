"""
Tracking algorithms.
"""

import cv2
import numpy as np

from segment import HSVColourSegmenter

class MultistageTracker(object):
    """
    Use ShapeFeatureTracker to find the initial objects until they have been 
    tracked long enough to be considered objects of interest.  Then let
    the CamShiftTracker take over so we can handle when they stop moving.
    Increment counter once the handover has taken place.
    """
    
    def __init__(self):
        self.shape_tracker = ShapeFeatureTracker()
        self.camshift_tracker = CamShiftTracker()
    
    @property
    def count(self):
        return len(self.camshift_tracker.tracked_objects)
    
    def update(self, current_image, contours):
        handoff_objects = self.shape_tracker.update(current_image, contours, 
                                        self.camshift_tracker.tracked_objects)
        
        if handoff_objects:
            self.camshift_tracker.track(handoff_objects)
            print "Fish count: %d" % self.count
            
        self.camshift_tracker.update(current_image)
        
        display = current_image.copy()
        self.draw_tracked_bounding_boxes(display)
        self.draw_counter(display)
        cv2.imshow("Tracker", display)
        
    def draw_tracked_bounding_boxes(self, img):
        min_x = 0
        max_x = img.shape[1]
        min_y = 0
        max_y = img.shape[0]
        
        # Draw potential objects in yellow
        for obj in self.shape_tracker.potential_objects:
            cv2.rectangle(img, obj.bbox.top_left, obj.bbox.bottom_right, (0, 255, 255))
        
        # Draw 'confirmed' objects in red
        for obj in self.camshift_tracker.tracked_objects:
            cv2.rectangle(img, self.restrict_point(obj.bbox.top_left, min_x, max_x, min_y, max_y), 
                          self.restrict_point(obj.bbox.bottom_right, min_x, max_x, min_y, max_y),
                          (0, 0, 255))
            
    def restrict_point(self, point, min_x, max_x, min_y, max_y):
        return (self.restrict_val(point[0], min_x, max_x), 
                self.restrict_val(point[1], min_y, max_y))
            
    def restrict_val(self, val, min_val, max_val):
        if val < min_val:
            return min_val
        elif val > max_val:
            return max_val
        else:
            return val
        
    def draw_counter(self, img):
        padding = 10
        text_bottom_left = (padding, img.shape[0] - padding)

        text = str(self.count)
        color = (0, 255, 255)
        fontFace = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        fontScale = 2
        thickness = 4
        cv2.putText(img, text, text_bottom_left, fontFace, 
                    fontScale, color, thickness)
    

class CamShiftTracker(object):
    """
    Example:
    https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python2/camshift.py?rev=6588
    """
    
    def __init__(self):
        self.tracked_objects = []
        self.mask_detector = HSVColourSegmenter()
        
    def track(self, objects):
        self.tracked_objects.extend(objects)

    def update(self, current_image):
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        mask = self.mask_detector.segment(current_image)
        
#        cv2.imshow("Mask", mask)
        
        for obj in self.tracked_objects:
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
            
            bbox.update(track_window)


class ShapeFeatureTracker(object):
    """
    Performs the early tracking where we determine if an object is of 
    interest for further tracking.
    
    It is based on the following shape features: centroid location, 
    orientation and area.
    """
    
    def __init__(self):
        self.potential_objects = []
        self.frame_number = 0
        
        self.count = 0
        
        # Thresholds for object similarity
        self.centroid_threshold = 40 # Euclidean distance
        self.area_threshold = 1750
        self.angle_threshold = 30 # in degrees

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
        self._prune_super_tracks()
        
    def _prune_short_tracks(self):
        obj_to_prune = []
        for obj in self.potential_objects:
            if (obj.is_new() and
                obj.last_frame_tracked < self.frame_number):
                obj_to_prune.append(obj)
                
        for obj in obj_to_prune:
            print "Pruned short track"
            self.potential_objects.remove(obj)
            
    def _prune_super_tracks(self):
        obj_to_prune = set()
        for potential_object in self.potential_objects:
            for known_object in self.known_objects:
                overlap = potential_object.bbox_overlap_area(known_object) 
                if overlap > 0.5 * known_object.bbox.area:
                    obj_to_prune.add(potential_object)
                    
        for obj in obj_to_prune:
            print "Pruned super track"
            self.potential_objects.remove(obj)
            
    def _get_other_objects(self, obj):
        other_objs = list(self.known_objects)
        other_objs.extend(self.potential_objects)
        other_objs.remove(obj)
        return other_objs
            
    def _prune_sub_tracks(self):
        sub_tracks = []
        for obj in self.potential_objects:
            for other_obj in self._get_other_objects(obj):
                if other_obj.contains(obj):
                    sub_tracks.append(obj)
                    break # inner loop
                
        for obj in sub_tracks:
            print "Pruned sub track"
            self.potential_objects.remove(obj)
    
    def _prune_high_overlap_tracks(self):
        obj_to_prune = []
        for obj in self.potential_objects:
            for other_obj in self._get_other_objects(obj):
                if other_obj.is_new():
                    continue

                if obj.bbox.area > other_obj.bbox.area:
                    # only prune the smaller objects
                    continue
                  
                if obj.bbox_overlap_area(other_obj) > 0.5 * obj.bbox.area:
                    # This object is mostly overlapped with another
                    obj_to_prune.append(obj)
                    break # inner loop
                
        for obj in obj_to_prune:
            print "Pruned high overlap object"
            self.potential_objects.remove(obj)
    
    def handoff_objects_of_interest(self):
        """
        After we have tracked an object for a while, we become more certain 
        it is one of the objects of interest.  Hand it off to the next 
        tracker.
        """
        handoff_objects = []
        for obj in self.potential_objects:
            if not obj.is_new():
                handoff_objects.append(obj)
        
        for obj in handoff_objects:
            self.potential_objects.remove(obj)
                    
        return handoff_objects
    
    def matches_known_object(self, new_obj):
        for obj in self.known_objects:
            if self.is_match(new_obj, obj):
                return True
        return False
    
    def match_potential_objects(self, new_obj):
        matches = []
        for obj in self.potential_objects:
            if self.is_match(new_obj, obj):
                matches.append(obj)
        return matches
    
    def update(self, current_image, contours, known_objects):
        self.known_objects = known_objects
        self.frame_number += 1
        
        frame_height, frame_width = current_image.shape[:2]
        
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            bbox = BoundingBox(*bounding_rect)
            new_obj = TrackedObject(bbox, contour, self.frame_number, 
                                    frame_width, frame_height)
            
            if self.matches_known_object(new_obj):
                self._prune_tracks()
                continue
            
            matches = self.match_potential_objects(new_obj)
            if len(matches) == 0:
                self.potential_objects.append(new_obj)
            else:
                # TODO: what if there are multiple matches?
                matches[0].update(bbox, contour, self.frame_number)
                
        self._prune_tracks()
        
        return self.handoff_objects_of_interest()
        

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
        if abs(self.dx) < 2 and abs(self.dy) < 2 and new_area < self.area:
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
    

class BoundingBox(object):
    
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

    def __str__(self):
        return ("Top left: (%f, %f), "
                "Bottom right: (%f, %f)" % (self.x0, self.y0, 
                                            self.x1, self.y1))

    @property
    def cv2rect(self):
        """
        Represent the bounding box in the format that opencv uses.
        """
        return (self.x0, self.y0, self.width, self.height)
        
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
        
    @property
    def has_negative_area(self):
        return self.x0 < 0 or self.y0 < 0 or self.width < 0 or self.height < 0
        
    def update(self, bbox):
        if isinstance(bbox, BoundingBox):
            self.x0 = bbox.x0
            self.y0 = bbox.y0
            self.width = bbox.width
            self.height = bbox.height
        elif isinstance(bbox, tuple):
            self.x0 = bbox[0]
            self.y0 = bbox[1]
            self.width = bbox[2]
            self.height = bbox[3]
        else:
            raise ValueError("Unknown representation of bounding box.")
        
