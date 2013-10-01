# Copyright (C) 2013 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Tracking based on shape features.
"""

import cv2
import numpy as np

import utils
from trackables import TrackedObject, BoundingBox
from pruning import Pruner

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
        
    def track(self, current_image, frame_number, contours, potential_objects, moving_objects, stationary_objects):
        all_moving_objects = utils.join_lists(potential_objects, moving_objects)
        known_objects = utils.join_lists(moving_objects, stationary_objects)
        
        frame_height, frame_width = current_image.shape[:2]
        
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)
            bbox = BoundingBox(*bounding_rect)
            new_obj = TrackedObject(bbox, contour, frame_number, 
                                    frame_width, frame_height)
            
            if self.matcher.has_match(new_obj, stationary_objects):
                continue
            
            matches = self.matcher.find_matches(new_obj, all_moving_objects)
            if len(matches) == 0:
                if new_obj.is_near_edge():
                    potential_objects.append(new_obj)
            else:
                # TODO: should we only update closest match?
                for match in matches:
                    match.update(bbox, contour, frame_number)
        
        # Prune spurious potential objects
        potential_objects = self.pruner.prune_inactive(potential_objects, 
                                                       frame_number)
        potential_objects = self.pruner.prune_subsumed(potential_objects, 
                                                       known_objects)
        potential_objects = self.pruner.prune_high_overlap(potential_objects, 
                                                           known_objects)
        potential_objects = self.pruner.prune_super_objects(potential_objects, 
                                                            known_objects)
        
        # Identify the potential objects that we are now confident are legitimate
        confirmed_objs = [obj for obj in potential_objects if not obj.is_new()]
        for obj in confirmed_objs:
            potential_objects.remove(obj)
            moving_objects.append(obj)
                
        # Find objects that were moving but have stopped
        stopped_objs = [obj for obj in moving_objects if obj.is_not_moving()]
        for obj in stopped_objs:
            moving_objects.remove(obj)
            stationary_objects.append(obj)
        
        return potential_objects, moving_objects, stationary_objects 
    
    
class ShapeMatcher(object):
    
    def __init__(self):
        # Thresholds for object similarity
        self.centroid_threshold = 50 # Euclidean distance
        self.area_threshold = 0.40 # the percent difference
        self.angle_threshold = 45 # in degrees
    
    def _is_centroid_match(self, obj1, obj2):
        return (np.sqrt(np.square(obj1.center[0] - obj2.center[0]) + 
                        np.square(obj1.center[1] - obj2.center[1]))
                < self.centroid_threshold) 
    
    def _is_area_match(self, obj1, obj2):
        max_area = max(obj1.area, obj2.area)
        min_area = min(obj1.area, obj2.area)
        return (abs(max_area - min_area) / max_area) < self.area_threshold  
#        return np.abs(obj1.area - obj2.area) < self.area_threshold
    
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

