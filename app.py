"""
Main module of the application performing fish counting.
"""

import segment
import videoanalyzer
import videoreader

def run(video_path):
    analyzer = videoanalyzer.Analyzer(segment.MovingAverageBackgroundSubtractor(0.05)) 
    videoreader.VideoReader(video_path, analyzer).start()

if __name__ == "__main__":
    run("data/fish_video.mp4")
