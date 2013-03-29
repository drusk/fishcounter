"""
Multistage tracking utilizing different tracking algorithms for different 
phases of detecting and tracking.
"""

from shapefeatures import ShapeFeatureTracker
from camshift import CamShiftTracker

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
    
    @property
    def known_objects(self):
        return self.camshift_tracker.tracked_objects
        
    @property
    def potential_objects(self):
        return self.shape_tracker.potential_objects
    
    def update(self, current_image, contours):
        handoff_objects = self.shape_tracker.update(current_image, contours, 
                                        self.camshift_tracker.tracked_objects)
        
        if handoff_objects:
            self.camshift_tracker.track(handoff_objects)
            print "Fish count: %d" % self.count
            
        self.camshift_tracker.update(current_image)

