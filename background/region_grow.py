import cv2
import numpy as np
from scipy.ndimage.filters import correlate

VIDEO = "data/fish_video.mp4"

class Fish(object):
    
    def __init__(self, control_pts):
        self.control_pts = control_pts
        
    def in_region(self, contour):
        hits = 0
        for (x, y) in self.control_pts:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                hits += 1
        
        # do majority vote of points
        return hits > len(self.control_pts) / 2

def draw_flow(im,flow,step=16):
    """ Plot optical flow at sample points spaced step pixels apart. """
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    lines = [((x1,y1),(x2,y2)) for ((x1,y1),(x2,y2)) in lines if abs(x1-x2) + abs(x1-x2) > 2 ]

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)

    return vis

def segment_by_velocity(flow, l_thresh=1.5, min_frame_portion=0.025):
    mag = np.sum(np.fabs(flow), axis=2)
    _, magbin = cv2.threshold(mag, l_thresh, 255, cv2.THRESH_BINARY)
    magbin = magbin.astype(np.uint8)
    
    # Apply morphological closing to combine pieces of same fish
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (35, 35))
    magbin = cv2.morphologyEx(magbin, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(magbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    frameshape = np.shape(flow)
    framesize = frameshape[0] * frameshape[1]
    
    large_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_frame_portion * framesize:
            large_contours.append(contour)
        
    return large_contours

def get_control_pts(img, contour):
    x, y, width, height = cv2.boundingRect(contour)
            
    sub_img = img[y:y+height, x:x+width]
    features = cv2.goodFeaturesToTrack(sub_img, 5, 0.5, 1)
    
    if features is None:
        return None
    
    # Get the points back in the coordinates of the original image
    norm_pts = [(int(pt[0][0]) + x, int(pt[0][1]) + y) for pt in features]
    
    return np.array(norm_pts, dtype=np.float32)

def draw_frame_with_tracked_pts(img, fishes):
    for fish in fishes:
        for (x, y) in fish.control_pts:
            cv2.circle(img, (int(x), int(y)), 7, (255, 0, 0))
        
    cv2.imshow("Frame", img)

# setup video capture
cap = cv2.VideoCapture(VIDEO)
ret, img = cap.read()

prev_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

skip = 950

fishes = []

while True:
    ret, img = cap.read()
    if skip > 0:
        skip -= 1
        continue
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Update fish control points
#    import pdb; pdb.set_trace()
    for fish in fishes:
        fish.control_pts, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, fish.control_pts, None)

    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    contours = segment_by_velocity(flow)
    
    # Find contours that are new fishes
    untracked_contours = []
    for contour in contours:
        add_to_list = True
        for fish in fishes:
            if fish.in_region(contour):
                add_to_list = False
        if add_to_list:
            untracked_contours.append(contour)
    
    for contour in untracked_contours:
        control_pts = get_control_pts(gray, contour)
        if control_pts is not None:
            fishes.append(Fish(control_pts))

    prev_gray = gray
    
    draw_frame_with_tracked_pts(img, fishes)
    
    # Debugging - draw contours
    tmp = gray.copy()
    cv2.drawContours(tmp, contours, -1, 255)
    cv2.imshow("Contours", tmp)
    
    print "Number of fish: %d" % len(fishes)

    if cv2.waitKey(10) == 27:
        break

