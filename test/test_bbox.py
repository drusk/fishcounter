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