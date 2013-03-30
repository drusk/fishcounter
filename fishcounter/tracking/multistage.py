"""
Multistage tracking utilizing different tracking algorithms for different 
phases of detecting and tracking.
"""

import utils
from shapefeatures import ShapeFeatureTracker
from camshift import CamShiftTracker

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
    
    @property
    def count(self):
        return len(self.known_objects)
    
    @property
    def known_objects(self):
        return utils.join_lists(self.moving_objects, self.stationary_objects)
    
    def update(self, current_image, contours):
        # Handle moving objects
        potential, moving, stationary = self.shape_tracker.update(current_image, contours, 
                                                                  self.potential_objects, 
                                                                  self.moving_objects,
                                                                  self.stationary_objects)
        self.potential_objects = potential
        self.moving_objects = moving
        self.stationary_objects = stationary
        
        # Handle stationary objects
        moving, stationary = self.camshift_tracker.update(current_image,
                                                          self.moving_objects,
                                                          self.stationary_objects)
        self.moving_objects = moving
        self.stationary_objects = stationary
        
