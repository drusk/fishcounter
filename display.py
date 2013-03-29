"""
Manages displaying results to user.
"""

import cv2

class DisplayManager(object):
    
    def __init__(self, display_name):
        self.display_name = display_name
        self.current_frame = None
    
    def _frame_height(self):
        return self.current_frame.shape[0]
    
    def _frame_width(self):
        return self.current_frame.shape[1]
    
    def new_frame(self, new_frame):
        self.current_frame = new_frame.copy()
    
    def draw_bounding_boxes(self, tracked_objects, colour):
        min_x = 0
        max_x = self._frame_width()
        min_y = 0
        max_y = self._frame_height()
        
        for obj in tracked_objects:
            cv2.rectangle(self.current_frame, 
                          self._restrict_point(obj.bbox.top_left, 
                                              min_x, max_x, min_y, max_y), 
                          self._restrict_point(obj.bbox.bottom_right, 
                                              min_x, max_x, min_y, max_y),
                          colour)
    
    def draw_counter(self, count):
        padding = 10
        text_bottom_left = (padding, self._frame_height() - padding)

        text = str(count)
        color = (0, 255, 255)
        fontFace = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        fontScale = 2
        thickness = 4
        cv2.putText(self.current_frame, text, text_bottom_left, fontFace, 
                    fontScale, color, thickness)
    
    def display_frame(self):
        cv2.imshow(self.display_name, self.current_frame)

    def _restrict_point(self, point, min_x, max_x, min_y, max_y):
        return (self._restrict_val(point[0], min_x, max_x), 
                self._restrict_val(point[1], min_y, max_y))
            
    def _restrict_val(self, val, min_val, max_val):
        if val < min_val:
            return min_val
        elif val > max_val:
            return max_val
        else:
            return val

