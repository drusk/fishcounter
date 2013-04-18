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
