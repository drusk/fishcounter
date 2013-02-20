"""
Foreground/background edge model.
"""

import cv2
import numpy as np

def get_frame(videocapture):
    # Read one frame of the video
    _, frame = videocapture.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def get_edge_frame(videocapture):
    return cv2.Canny(get_frame(videocapture), 40, 100)

def main(videofile):
    # Initialize video stream
    cap = cv2.VideoCapture(videofile)
    
    # Get video stream width and height
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    
    edge_counts = np.zeros((height, width))
    
    num_bg_samples = 50
    for _ in xrange(num_bg_samples):
        cv2.accumulate(get_edge_frame(cap)/255.0, edge_counts)

    bg_prob = edge_counts / float(num_bg_samples)
    print bg_prob.min()
    print bg_prob.max()
    
    bg_thresh = 0.5
    bg_pixels = bg_prob > bg_thresh
    
    while True:
        frame = get_frame(cap)
        cv2.imshow("Original", frame)
        
        edge_img = cv2.Canny(frame, 40, 100)
        cv2.imshow("Raw edges", edge_img)
        
        edge_nobg = edge_img.copy()
        edge_nobg[bg_pixels] = 0
        cv2.imshow("Edges no bg", edge_nobg) 
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("screenshot.jpg", edge_img)

if __name__ == "__main__":
    main("data/fish_video.mp4")

