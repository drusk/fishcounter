"""
Tests for segment module.
"""

import unittest

import numpy as np
from hamcrest import assert_that
from hamcrest import contains_inanyorder

import segment

class SegmentTest(unittest.TestCase):

    def test_get_outer_border_pts(self):
        bin_img = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0], 
                            [0, 0, 255, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]
                            ], dtype=np.uint8)
        
        border_pts = segment.get_outer_border_pts(bin_img)
        
        assert_that(border_pts, 
                    contains_inanyorder((1, 1), (1, 2), (1, 3), (2, 3), 
                                        (3, 3), (3, 2), (3, 1), (2, 1)))

    def test_hysteresis_thresh(self):
        img = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], 
                        [0, 0, 0, 0, 0],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1, 1]
                        ], dtype=np.float32)
        
        segmented = segment.hysteresis_thresh(img, 0.25, 0.75)

        # NOTE that the boundary pixels on the far left and right 
        # unfortunately aren't found by findContours so don't get included.
        # Ideally they would be included, but for larger images the effect 
        # will be fairly minimal.
        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 255, 255, 255, 0],
                             [255, 255, 255, 255, 255]
                             ], dtype=np.float32)
        
        assert_that((segmented == expected).all())
        
    def test_hysteresis_thresh_2_iterations(self):
        img = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], 
                        [0, 0, 0, 0, 0],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1, 1]
                        ], dtype=np.float32)
        
        segmented = segment.hysteresis_thresh(img, 0.25, 0.75)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [255, 255, 255, 255, 255]
                             ], dtype=np.float32)
        
        assert_that((segmented == expected).all())
        
    def test_hysteresis_thresh_3_iterations(self):
        img = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], 
                        [0, 0, 0, 0, 0],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1, 1]
                        ], dtype=np.float32)
        
        segmented = segment.hysteresis_thresh(img, 0.25, 0.75)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [255, 255, 255, 255, 255]
                             ], dtype=np.float32)
        
        assert_that((segmented == expected).all())
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    