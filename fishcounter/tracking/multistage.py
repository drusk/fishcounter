"""
Multistage tracking utilizing different tracking algorithms for different 
phases of detecting and tracking.
"""

import utils
from shapefeatures import ShapeFeatureTracker
from camshift import CamShiftTracker
from pruning import Pruner

class MultistageTracker(object):
    """
    Use ShapeFeatureTracker to track moving objects (including detection of 
    new objects) and CamshiftTracker to track stationary objects.
    """
    
    def __init__(self):
        self.shape_tracker = ShapeFeatureTracker()
        self.camshift_tracker = CamShiftTracker()
        
        self.potential_objects = []
        self.moving_objects = []
        self.stationary_objects = []
        
        self.frame_number = 1
        self.pruner = Pruner()
    
    @property
    def count(self):
        return len(self.known_objects)
    
    @property
    def known_objects(self):
        return utils.join_lists(self.moving_objects, self.stationary_objects)
    
    def update(self, current_image, contours):
        self.frame_number += 1
        
        # Handle moving objects
        potential, moving, stationary = self.shape_tracker.update(current_image, 
                                                                  self.frame_number,
                                                                  contours, 
                                                                  self.potential_objects, 
                                                                  self.moving_objects,
                                                                  self.stationary_objects)
        self.potential_objects = potential
        self.moving_objects = moving
        self.stationary_objects = stationary
        
        # Handle stationary objects
        moving, stationary = self.camshift_tracker.update(current_image,
                                                          self.frame_number,
                                                          self.moving_objects,
                                                          self.stationary_objects)
        self.moving_objects = moving
        self.stationary_objects = stationary
        
        # Sometimes moving_objects stop getting updated, usually because 
        # they were only a piece of a larger object that emerged.  Prune 
        # back after they are idle too long.
#        self.moving_objects = self.pruner.prune_inactive(self.moving_objects, 
#                                                         self.frame_number,
#                                                         inactive_frames=10)
        
