"""
Experimenting using Python OpenCV binding.
"""

import cv2

def separate_rgb(filename):
    image = cv2.imread(filename)
    blue, green, red = cv2.split(image)
    
#    img_gray = cv2.cvtColor(image, cv2.CV_)
    
    cv2.imshow("Red channel", red)
    cv2.imshow("Green channel", green)
    cv2.imshow("Blue channel", blue)
    cv2.waitKey()
    
def histeq(filename):
    im_gray = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imshow("Original image as grayscale", im_gray)

    img_eq = cv2.equalizeHist(im_gray)
    cv2.imshow("Equalized image", img_eq)
    
    cv2.waitKey()

if __name__ == "__main__":
    filename = "data/fish_ss.png"
    
    separate_rgb(filename)
    histeq(filename)
    