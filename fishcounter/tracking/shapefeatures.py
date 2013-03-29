"""
Tracking based on shape features.
"""

import cv2
import numpy as np

from trackables import TrackedObject, BoundingBox

class ShapeFeatureTracker(object):
    """
    Performs the early tracking where we determine if an object is of 
    interest for further tracking.
    
    It is based on the following shape features: centroid location, 
    orientation and area.
    """
    
    def __init__(self):
        self.matcher = ShapeMatcher()
        
        self.potential_objects = []
        self.frame_number = 0
        
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
    
    def update(self, current_image, contours, known_objects):
        self.known_objects = known_objects
        self.frame_number += 1
        
        frame_height, frame_width = current_image.shape[:2]
        
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            bbox = BoundingBox(*bounding_rect)
            new_obj = TrackedObject(bbox, contour, self.frame_number, 
                                    frame_width, frame_height)
            
            if self.matcher.has_match(new_obj, known_objects):
                self._prune_tracks()
                continue
            
            matches = self.matcher.find_matches(new_obj, self.potential_objects)
            if len(matches) == 0:
                self.potential_objects.append(new_obj)
            else:
                # TODO: what if there are multiple matches?
                matches.pop().update(bbox, contour, self.frame_number)
                
        self._prune_tracks()
        
        return self.handoff_objects_of_interest()
    
    
class ShapeMatcher(object):
    
    def __init__(self):
        # Thresholds for object similarity
        self.centroid_threshold = 40 # Euclidean distance
        self.area_threshold = 1750
        self.angle_threshold = 30 # in degrees
    
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
        
    def find_matches(self, target_object, search_objects):
        matches = set()
        for search_object in search_objects:
            if self.is_match(target_object, search_object):
                matches.add(search_object)
        return matches
    
    def has_match(self, target_object, search_objects):
        matches = self.find_matches(target_object, search_objects)
        return len(matches) > 0

