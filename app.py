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
Main module of the application performing fish counting.
"""

import argparse

from fishcounter.display import DisplayManager
from fishcounter.segment import (CompositeSegmentationAlgorithm,
                                 MovingAverageBackgroundSubtractor,
                                 MixtureOfGaussiansBackgroundSubtractor,
                                 HSVColourSegmenter)
from fishcounter.tracking.multistage import MultistageTracker
from fishcounter.analyzer import Analyzer
from fishcounter.videoreader import VideoReader

def run(video_path, skip=0):
    segmenter = CompositeSegmentationAlgorithm([
        MovingAverageBackgroundSubtractor(0.05),
        MixtureOfGaussiansBackgroundSubtractor(),
        HSVColourSegmenter()])
    display = DisplayManager("Fish Counter")
    tracker = MultistageTracker()
    analyzer = Analyzer(segmenter, tracker, display)
    VideoReader(video_path, analyzer).start(skip)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video", help="the video to analyze")
    parser.add_argument("-s", "--skip", type=int, default=0,
                        help="skip to start analyzing at a later frame")

    args = parser.parse_args()

    run(args.video, args.skip)
