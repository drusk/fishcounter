"""
Experimenting with background subtraction.
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
    
    avg1 = np.float32(frame)
    avg2 = np.float32(frame)
    
    accum = np.float32(frame)
    
    num_bg_samples = 20
    for _ in xrange(num_bg_samples):
        frame = get_frame(cap)
        cv2.accumulate(frame, accum)
        
    bg = cv2.convertScaleAbs(accum / float(num_bg_samples))
    bgmean = bg.mean()
    bgstd = bg.std()
    
    cv2.imshow("Background", bg)
    while True:
        frame = get_frame(cap)
        
        fg = cv2.convertScaleAbs(bg - frame)
        _, fgbin = cv2.threshold(fg, bgmean + bgstd, 255, cv2.THRESH_BINARY)
#        cv2.accumulateWeighted(frame, avg1, 0.1)
#        cv2.accumulateWeighted(frame, avg2, 0.01)
        
#        res1 = cv2.convertScaleAbs(avg1)
#        res2 = cv2.convertScaleAbs(avg2)
        
        cv2.imshow('img',frame)
        cv2.imshow("FG", fg)
        cv2.imshow("FG bin", fgbin)
#        cv2.imshow('avg1',res1)
#        cv2.imshow('avg2',res2)
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("screenshot.jpg", frame)

if __name__ == "__main__":
    main("data/fish_video.mp4")
