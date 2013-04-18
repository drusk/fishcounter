import unittest

from hamcrest import assert_that
from hamcrest import equal_to

from fishcounter.tracking.trackables import BoundingBox

class BBoxTest(unittest.TestCase):

    def test_overlap_area(self):
        bbox1 = BoundingBox(0, 0, 2, 2)
        bbox2 = BoundingBox(0, 1, 2, 2)
        
        assert_that(bbox1.overlap_area(bbox2), equal_to(2))

    def test_overlap_area_zero(self):
        bbox1 = BoundingBox(10, 10, 2, 2)
        bbox2 = BoundingBox(0, 0, 3, 3)
        
        assert_that(bbox1.overlap_area(bbox2), equal_to(0))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()