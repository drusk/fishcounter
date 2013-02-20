"""
Counts the number of fish that appear in a video segment.
"""

import cv2

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
    
    while True:
        frame = get_frame(cap)
        
        edge_img = cv2.Canny(frame, 50, 80)
        
        cv2.imshow("Video test", edge_img)
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("screenshot.jpg", edge_img)

if __name__ == "__main__":
    main("data/fish_video.mp4")
