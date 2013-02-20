"""
Counts the number of fish that appear in a video segment.
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
    
    frame = get_frame(cap)
    edge_img = cv2.Canny(frame, 40, 100)
    
    bg = np.uint8(edge_img)
    num_bg_samples = 20
    for _ in xrange(num_bg_samples):
        frame = get_frame(cap)
        edge_img = cv2.Canny(frame, 40, 100)
        bg[edge_img.nonzero()] = 255
    
    cv2.imshow("BG", bg)
    while True:
        frame = get_frame(cap)
        cv2.imshow("Original", frame)
        
        edge_img = cv2.Canny(frame, 40, 100)
        cv2.imshow("Raw edges", edge_img)
        
        edge_nobg = cv2.convertScaleAbs(edge_img - bg)
        cv2.imshow("Edges no bg", edge_nobg) 
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("screenshot.jpg", edge_img)

if __name__ == "__main__":
    main("data/fish_video.mp4")
