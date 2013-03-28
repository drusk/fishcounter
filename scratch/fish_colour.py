"""
Experimenting with fish colours.
"""

import cv2
import numpy as np

def hsv_test(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    hue_min = 75
    hue_max = 100
    sat_min = 0.
    sat_max = 100
    val_min = 100.
    val_max = 255.
    mask = cv2.inRange(hsv, np.array((hue_min, sat_min, val_min)), 
                            np.array((hue_max, sat_max, val_max)))

    cv2.imshow("Raw colour segmentation", mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=17)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    
    cv2.imshow("Postprocessed", mask)
    
    cv2.waitKey()


if __name__ == "__main__":
    img = cv2.imread("data/fish_ss.png")
    hsv_test(img)
    
    shadow_img = cv2.imread("data/fish_and_shadow.jpg")
    hsv_test(shadow_img)
