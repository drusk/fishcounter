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
        self.pruner = Pruner()
        
        self.potential_objects = []
        self.frame_number = 0
    
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

    def _prune_tracks(self):
        self.potential_objects = self.pruner.prune_inactive(
                                                self.potential_objects, 
                                                self.frame_number)
        self.potential_objects = self.pruner.prune_subsumed(
                                                self.potential_objects,
                                                self.known_objects)
        self.potential_objects = self.pruner.prune_high_overlap(
                                                self.potential_objects,
                                                self.known_objects)
        self.potential_objects = self.pruner.prune_super_objects(
                                                self.potential_objects,
                                                self.known_objects)
    
    
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


class Pruner(object):
    
    def _get_others(self, excluded_obj, others1, others2):
        all_others = list(others1)
        all_others.extend(others2)
        all_others.remove(excluded_obj)
        return all_others
    
    def prune_inactive(self, tracked_objects, current_frame_number):
        
        def is_active(tracked_object):
            return tracked_object.last_frame_tracked == current_frame_number
        
        return filter(is_active, tracked_objects)

    def prune_subsumed(self, pruneable_objects, other_objects):
        sub_tracks = []
        for obj in pruneable_objects:
            for other_obj in self._get_others(obj, pruneable_objects, other_objects):
                if other_obj.contains(obj):
                    sub_tracks.append(obj)
                    break
        
        for obj in sub_tracks:
            print "Pruned subsumed"
            pruneable_objects.remove(obj)
            
        return pruneable_objects
    
    def prune_high_overlap(self, pruneable_objects, other_objects):
        obj_to_prune = []
        for obj in pruneable_objects:
            for other_obj in self._get_others(obj, pruneable_objects, other_objects):
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
            print "Pruned high overlap"
            pruneable_objects.remove(obj)
            
        return pruneable_objects
    
    def prune_super_objects(self, pruneable_objects, other_objects):
        obj_to_prune = set()
        for obj in pruneable_objects:
            for other_obj in other_objects:
                overlap = obj.bbox_overlap_area(other_obj) 
                if overlap > 0.5 * other_obj.bbox.area:
                    obj_to_prune.add(obj)
                    
        for obj in obj_to_prune:
            print "Pruned super track"
            pruneable_objects.remove(obj)
            
        return pruneable_objects

