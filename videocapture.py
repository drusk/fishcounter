"""
Example of using video capture.
"""

import cv2

def main(videosource=0):
    cap = cv2.VideoCapture(videosource)
    
    while True:
        _, im = cap.read()
        cv2.imshow("Video test", im)
        
        key = cv2.waitKey(10)
        
        if key == 27:
            break
        if key == ord(" "):
            cv2.imwrite("vid_result.jpg", im)
            

if __name__ == "__main__":
    main("data/fish_video.mp4")
#    main()
