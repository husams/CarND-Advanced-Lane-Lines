from image_processing import *
from lanes_and_curvatures import *
from calibration import *

import numpy as np

class LanesDetector(object):
    def __init__(self, camera):
        self.nframes        = 5
        self.camera         = camera
        self.M              = None
        self.Minv           = None
        self.src            = np.float32([[(200, 720), (570, 465), (730, 465), (1200, 720)]]) 
        self.dst            = np.float32([[(320, 720), (320, 0), (980, 0), (980, 720)]])
        #Avarage Fit
        self.detected       = False
        self.left_best_fit  = None
        self.right_best_fit = None
        # Hold  last n fits 
        self.left_current_fit  =  []
        self.right_current_fit = []

    
    
    def sanity_check(self, binary_warped, left_fit, right_fit):
        """
            Check if the Second degree polynomial for left  and rightlines
            should be accepted.
        Args:
            binary_warped(np.array)   - (heigt x width x channels) matrix represnt binary warped image.
            left_fit(np.array)        - Second degree polynomial for left line.
            right_fit(np.array)       - Second degree polynomial for right line.
        Returns:
            True if the left and right fit should be accepted otherwise False.
        """ 
        if left_fit is None or right_fit is None:
            # No lane detected.
            return False
        
        # 1. Conpute distence betweem left and right points
        y      = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        leftx  = left_fit[0] * y**2  + left_fit[1] * y  + left_fit[2]
        rightx = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        diff   = rightx - leftx
        
        # 2. Compute the STD for distance between the points 
        std    = np.std(diff)
    
        # 3. only accept lane with STD < 40 given the
        #    the distance between left and right points
        #    greater then zero 
        return (std < 40 and np.all(diff > 0))

    def mark_lane(self, image):
        """
            Detect and mark the lanes
        Args:
            image(np.array)   - (heigt x width x channels) matrix represnt colored image.
        Returns:
            (heigt x width x channels) matrix represnt colored image with lane marked.            
        """ 
        # Calculate matrix transformation
        if self.M is None:
            self.M, self.Minv  = get_perspective_transform(image, self.src, self.dst)
            
        # 1. undistort image
        undistorted_image = self.camera.undistort(image)

        # 2. Create binary mask using color thresholding
        binary = binary_mask(undistorted_image)

        # 3. warp
        binary_warped = warp(binary, self.M)

        # 4. Find lane
        if self.detected:
            left_fit, right_fit, leftx, lefty, rightx, righty = find_lane_using_previous_fit(binary_warped, 
                                                                                             self.left_current_fit[-1], 
                                                                                             self.right_current_fit[-1])
        else:
            left_fit, right_fit, leftx, lefty, rightx, righty = find_lines(binary_warped)
       
       # 5. Check the new fitted lines
        if self.sanity_check(binary_warped,left_fit,right_fit):
            # Mark lane is detevcted and and left abd right fits to
            # the current list
            self.detected = True
            self.left_current_fit.append(left_fit)
            self.right_current_fit.append(right_fit)
            
            # Only keep n frames
            if len(self.left_current_fit) > self.nframes:
                self.left_current_fit  = self.left_current_fit[1:]
                self.right_current_fit = self.right_current_fit[1:]
            # Compute the mean
            self.left_best_fit = np.mean(self.left_current_fit, axis=0)
            self.right_best_fit = np.mean(self.right_current_fit, axis=0)
        elif self.left_best_fit is None:
            # We don't have fit, rturn the same image
            return image

        # 5. Mark lane
        marked_image =  mark_lane(undistorted_image, binary_warped, 
                                  self.left_best_fit, self.right_best_fit, 
                                  self.Minv)
        # 6. Compute the curvature and distance with respect to center 
        return draw_curvatures_and_distance(marked_image, self.left_best_fit, self.right_best_fit)