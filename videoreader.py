"""
Read and process video files.
"""

import cv2

class VideoReader(object):
    """
    Reads a video file and analyzes each frame with a specified video 
    analyzer.
    """
    
    def __init__(self, video_path, video_analyzer):
        self.capture = cv2.VideoCapture(video_path)
        self.video_analyzer = video_analyzer
        
    def start(self):
        frame_was_read, current_image = self.capture.read()
        
        while frame_was_read:
            previous_image = current_image
            frame_was_read, current_image = self.capture.read()
            
            self.video_analyzer.analyze(previous_image, current_image)
            
            # Exit if user presses the Escape key
            if cv2.waitKey(10) == 27:
                break


class FrameDisplayAnalyzer(object):
    """
    A video analyzer which does nothing but display the current frame.
    """
    
    def analyze(self, previous_image, current_image):
        cv2.imshow("Frame", current_image)
    

if __name__ == "__main__":
    VideoReader("data/fish_video.mp4", FrameDisplayAnalyzer()).start()
