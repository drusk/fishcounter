"""
Representations of things that can be tracked.
"""

import cv2

class TrackedObject(object):
    
    def __init__(self, bbox, contour, frame_number, frame_width, frame_height):
        self.bbox = bbox
        self.area = cv2.contourArea(contour)
        self.rotated_bbox = cv2.minAreaRect(contour) 
        
        # Keep track of "velocity"
        self.dx = 0
        self.dy = 0
        
        self.prev_area = 0
        
        self.frames_tracked = 1
        self.last_frame_tracked = frame_number
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.is_frozen = False
        
    @property
    def center(self):
        return self.bbox.center
    
    @property
    def angle(self):
        # raw angle seems inconsistent, can be 0 or -90 on same box
        raw_angle =self.rotated_bbox[2]
        if raw_angle < -45:
            return raw_angle + 90
        else:
            return raw_angle
        
    def update(self, new_bbox, contour, frame_number):
        self.dx = self.bbox.center[0] - new_bbox.center[0]
        self.dy = self.bbox.center[1] - new_bbox.center[1]
        
        new_area = cv2.contourArea(contour)
        self.bbox.update(new_bbox)
        self.prev_area = self.area
        self.area = new_area
        self.rotated_bbox = cv2.minAreaRect(contour)
        
        self.frames_tracked += 1
        self.last_frame_tracked = frame_number
    
    def delta_area(self):
        return self.area - self.prev_area
    
    def contains(self, other_obj):
        return self.bbox.contains_bbox(other_obj.bbox)
    
    def bbox_overlap_area(self, other_obj):
        return self.bbox.overlap_area(other_obj.bbox)
    
    def is_new(self):
        return self.frames_tracked < 5
    
    def is_not_moving(self):
        return abs(self.dx) < 2 and abs(self.dy) < 2 and self.delta_area() < 0
    

class BoundingBox(object):
    
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        
    def __str__(self):
        return ("Top left: (%f, %f), "
                "Bottom right: (%f, %f)" % (self.x0, self.y0, 
                                            self.x1, self.y1))

    @property
    def cv2rect(self):
        """
        Represent the bounding box in the format that opencv uses.
        """
        return (self.x0, self.y0, self.width, self.height)
        
    @property
    def x1(self):
        return self.x0 + self.width
    
    @property
    def y1(self):
        return self.y0 + self.height
        
    @property
    def center(self):
        return (self.x0 + self.width / 2, self.y0 + self.height / 2)
    
    @property
    def top_left(self):
        return (self.x0, self.y0)
    
    @property
    def bottom_right(self):
        return (self.x1, self.y1)
    
    @property
    def area(self):
        return self.width * self.height

    def contains_point(self, point):
        return (point[0] >= self.x0 and point[0] <= self.x1 and
                point[1] >= self.y0 and point[1] <= self.y1)
        
    def contains_bbox(self, other_bbox):
        return (self.contains_point(other_bbox.top_left) and
                self.contains_point(other_bbox.bottom_right))
        
    def overlap_area(self, other_bbox):
        x_overlap = max(0, min(self.x1, other_bbox.x1) - max(self.x0, other_bbox.x0))
        y_overlap = max(0, min(self.y1, other_bbox.y1) - max(self.y0, other_bbox.y0))
        return x_overlap * y_overlap
        
    @property
    def has_negative_area(self):
        return self.x0 < 0 or self.y0 < 0 or self.width < 0 or self.height < 0
        
    def update(self, bbox):
        if isinstance(bbox, BoundingBox):
            self.x0 = bbox.x0
            self.y0 = bbox.y0
            self.width = bbox.width
            self.height = bbox.height
        elif isinstance(bbox, tuple):
            self.x0 = bbox[0]
            self.y0 = bbox[1]
            self.width = bbox[2]
            self.height = bbox[3]
        else:
            raise ValueError("Unknown representation of bounding box.")

