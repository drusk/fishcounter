"""
Main module of the application performing fish counting.
"""

from segment import (CompositeSegmentationAlgorithm, 
                     MovingAverageBackgroundSubtractor,
                     MixtureOfGaussiansBackgroundSubtractor)
from tracking import ShapeFeatureTracker
from videoanalyzer import Analyzer
from videoreader import VideoReader

def run(video_path):
    segmenter = CompositeSegmentationAlgorithm(
                    MovingAverageBackgroundSubtractor(0.05),
                    MixtureOfGaussiansBackgroundSubtractor())
    tracker = ShapeFeatureTracker()
    analyzer = Analyzer(segmenter, tracker)
    VideoReader(video_path, analyzer).start()

if __name__ == "__main__":
    run("data/fish_video.mp4")
