import cv2
import numpy as np
from scipy.ndimage.filters import correlate

VIDEO = "data/fish_video.mp4"

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

def segment_by_velocity(img, flow, l_thresh=1.5):
    mag = np.sum(np.fabs(flow), 2)
    mag[mag < l_thresh] = 0
    _, magbin = cv2.threshold(mag, l_thresh, 255, cv2.THRESH_BINARY)
    magbin = magbin.astype(np.uint8)
    contours, _ = cv2.findContours(magbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    large_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            large_contours.append(contour)
        
    return large_contours

def get_control_pts(im, contours):
    boxed_img = im.copy()
    boxed_gray = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2GRAY)
    
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
            
        sub_img = boxed_gray[y:y+height, x:x+width]
    
    cv2.imshow("Sub img", sub_img)
            
    features = cv2.goodFeaturesToTrack(sub_img, 5, 0.5, 1)
    pts = []
    for pt in features:
        normalized_pt = (int(pt[0][0]) + x, int(pt[0][1]) + y)
        cv2.circle(boxed_img, normalized_pt, 7, (255, 0, 0))
        pts.append(normalized_pt)
        
    cv2.imshow("Boxed img", boxed_img)
    return pts

def segment_by_velocity2(im, flow, l_thresh=1.5, n=30):
    mag = np.sum(np.fabs(flow), 2)
    mag[mag < l_thresh] = 0

    _, magbin = cv2.threshold(mag, l_thresh, 255, cv2.THRESH_BINARY)
    magbin = magbin.astype(np.uint8)
    contours, _ = cv2.findContours(magbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros(magbin.shape)
    
    large_contours = []
    boxed_img = im.copy()
    boxed_gray = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2GRAY)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            large_contours.append(contour)
            
    for i in xrange(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 3000:
            large_contours.append(contours[i])
            cv2.drawContours(contour_img, contours, i, 255)

            x, y, width, height = cv2.boundingRect(contours[i])
            
            sub_img = boxed_gray[y:y+height, x:x+width]
            cv2.imshow("Sub img", sub_img)
            
            features = cv2.goodFeaturesToTrack(sub_img, 5, 0.5, 1)
            for pt in features:
                cv2.circle(boxed_img, (int(pt[0][0]) + x, int(pt[0][1]) + y), 7, (255, 0, 0))
            break

    cv2.imshow("Contours", contour_img)
    
    cv2.imshow("Bounded", boxed_img)
    
#    print np.max(mag)
    kernel = np.ones((n,n))
    mag_accum = correlate(mag, kernel)
    return mag_accum > n*n

def draw_region(im, regions):
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    vis[regions] = (255,0,0)
    return vis

# setup video capture
cap = cv2.VideoCapture(VIDEO)
ret,im = cap.read()
prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

skip = 950

while True:
    ret,im = cap.read()
    if skip > 0:
        skip -= 1
        continue
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray

    

    contours = segment_by_velocity(im, flow)
    control_pts = get_control_pts(im, contours)
    
#    regions = segment_by_velocity2(im, flow)

    # plot the flow vectors
    #cv2.imshow('Optical flow', draw_flow(gray,flow))
#    cv2.imshow('Optical flow', draw_region(gray,regions))
    if cv2.waitKey(10) == 27:
        break

