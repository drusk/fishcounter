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

    def start(self, skip=0):
        frame_was_read, current_image = self.capture.read()

        while frame_was_read:
            previous_image = current_image
            frame_was_read, current_image = self.capture.read()

            if skip > 0:
                skip -= 1
                continue

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
    VideoReader("data/fish_video.mp4", FrameDisplayAnalyzer()).start(skip=500)
