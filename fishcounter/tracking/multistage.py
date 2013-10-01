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

    def track(self, current_image, contours):
        self.frame_number += 1

        # Handle moving objects
        potential, moving, stationary = self.shape_tracker.track(current_image,
                                                                 self.frame_number,
                                                                 contours,
                                                                 self.potential_objects,
                                                                 self.moving_objects,
                                                                 self.stationary_objects)
        self.potential_objects = potential
        self.moving_objects = moving
        self.stationary_objects = stationary

        # Handle stationary objects
        moving, stationary = self.camshift_tracker.track(current_image,
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
        
