"""
Counts the number of fish that appear in a video segment.
"""

import cv2
import numpy as np

def get_frame(videocapture):
    # Read one frame of the video
    _, frame = videocapture.read()
        
    # Red channel seems to have the best contrast so grab it
    _, _, redchannel = cv2.split(frame)
        
    # Histogram equalize to improve contrast further
    # Actually, this doesn't look too great...
    #return cv2.equalizeHist(redchannel)
    
    return redchannel

def main(videofile):
    # Initialize video stream
    cap = cv2.VideoCapture(videofile)
    
    # Get video stream width and height
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    
    # Accumulators to be used for background subtraction
    accum = np.zeros((height, width), np.float32)
    sqaccum = np.zeros((height, width), np.float32)
    
    num_bg_samples = 20
    for _ in xrange(num_bg_samples):
        frame = get_frame(cap)
        cv2.accumulate(frame, accum)
        cv2.accumulateSquare(frame, sqaccum)

    avg_bg = accum / float(num_bg_samples)
    mu, sigma = cv2.meanStdDev(avg_bg)
    
    while True:
        frame = get_frame(cap)
        
        frame -= mu
        
        cv2.imshow("Video test", frame)
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("screenshot.jpg", frame)

if __name__ == "__main__":
    main("data/fish_video.mp4")
