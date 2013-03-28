"""
Main module of the application performing fish counting.
"""
import sys

from segment import (CompositeSegmentationAlgorithm, 
                     MovingAverageBackgroundSubtractor,
                     MixtureOfGaussiansBackgroundSubtractor,
                     HSVColourSegmenter)
from tracking import MultistageTracker
from videoanalyzer import Analyzer
from videoreader import VideoReader

def run(video_path, skip=0):
    segmenter = CompositeSegmentationAlgorithm([
                    MovingAverageBackgroundSubtractor(0.05),
                    MixtureOfGaussiansBackgroundSubtractor(),
                    HSVColourSegmenter()])
    tracker = MultistageTracker()
    analyzer = Analyzer(segmenter, tracker)
    VideoReader(video_path, analyzer).start(skip)

if __name__ == "__main__":
    skip = 0
    if len(sys.argv) == 2:
        skip = int(sys.argv[1])
    run("data/fish_video.mp4", skip)
