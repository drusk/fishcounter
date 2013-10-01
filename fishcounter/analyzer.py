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
Analyzers coordinate the analysis performed by segmentation algorithms, 
tracking algorithms, display, etc.
"""

import cv2

from components import find_connected_components


class Analyzer(object):
    def __init__(self, segmenter, tracker, display):
        self.segmenter = segmenter
        self.tracker = tracker
        self.display = display

    def analyze(self, previous_image, current_image):
        segmented = self.segmenter.segment(current_image)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # TODO an algorithm to dynamically set threshold
        contours = find_connected_components(segmented, kernel, 200)

        self.tracker.track(current_image, contours)

        self._display_findings(current_image)

    def _display_findings(self, current_image):
        self.display.new_frame(current_image)
        self.display.draw_bounding_boxes(self.tracker.potential_objects,
                                         (0, 255, 255))
        self.display.draw_bounding_boxes(self.tracker.moving_objects,
                                         (255, 0, 0))
        self.display.draw_bounding_boxes(self.tracker.stationary_objects,
                                         (0, 0, 255))
        self.display.draw_counter(self.tracker.count)

        self.display.display_frame()

