"""
Try Gaussian smoothing before edge detection.
"""

import cv2
import numpy as np

def get_frame(videocapture):
    # Read one frame of the video
    _, frame = videocapture.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def main(videofile):
    # Initialize video stream
    cap = cv2.VideoCapture(videofile)
    
    while True:
        frame = get_frame(cap)
        cv2.imshow("Original", frame)
        smoothed = cv2.GaussianBlur(frame, (0, 0), 1)
        
        cv2.imshow("Smoothed", smoothed)
        
        edge_img = cv2.Canny(smoothed, 40, 80)
        cv2.imshow("Edges", edge_img)
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("screenshot.jpg", edge_img)

if __name__ == "__main__":
    main("data/fish_video.mp4")
