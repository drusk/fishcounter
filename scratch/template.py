import cv2
from cv2 import cv

def template_match(image_path, template_path):
    img = cv2.imread(image_path)
    templ = cv2.imread(template_path)

    r1 = cv2.matchTemplate(img, templ, cv.CV_TM_SQDIFF_NORMED) 
    r2 = cv2.matchTemplate(img, templ, cv.CV_TM_CCORR_NORMED) 

    _, _, pos1, _ = cv2.minMaxLoc(r1)
    _, _, _, pos2 = cv2.minMaxLoc(r2)

    cv2.circle(img, pos1, 5, (0, 0, 255))
    cv2.circle(img, pos2, 5, (255, 0, 0))

    #import ipdb; ipdb.set_trace()
    cv2.imshow("foo", img)
    #cv2.imshow("a", (1 - r1)*255 )
    cv2.imshow("b", r2**(20))
    cv2.waitKey()
    
if __name__ == "__main__":
    image = "data/fish_ss.png"
    template_eye = "data/eye.png"
    template_head = "data/head.png"

    template_match(image, template_eye)
