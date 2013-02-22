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

def segment_by_velocity(im, flow, l_thresh=1.5, n=30):
    mag = np.sum(np.fabs(flow), 2)
    mag[mag < l_thresh] = 0
    print np.max(mag)
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

skip = 230
while True:
    ret,im = cap.read()
    if skip > 0:
        skip -= 1
        continue
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray

    regions = segment_by_velocity(im, flow)

    # plot the flow vectors
    #cv2.imshow('Optical flow', draw_flow(gray,flow))
    cv2.imshow('Optical flow', draw_region(gray,regions))
    if cv2.waitKey(10) == 27:
        break

