import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as im
import pickle



class Camera(object):
    def __init__(self):
        self.mtx  = None
        self.dist = None

    def calibrate(self, imagesPath):
        # Object points
        objp        = np.zeros((6*9, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        objpoints = []
        imgpoints = []

        images = glob.glob(imagesPath)

        for idx, filename in enumerate(images):
            # Read image
            image = im.imread(filename)
            # Convert to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find image points
            isFound, points = cv2.findChessboardCorners(gray, (9,6), None)
            
            # Check if we found an image
            if isFound == True:
                objpoints.append(objp)
                imgpoints.append(points)

        cv2.destroyAllWindows()

        # Read test image
        image = im.imread(images[0])
        size  = (image.shape[1], image.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

        self.mtx  = mtx
        self.dist = dist
    
    def  undistort(self, image):
        if self.mtx is None or self.dist is None:
            raise Exception("You need to calibrate the camera first")

        return cv2.undistort(image, self.mtx,self.dist, None, self.mtx)


def gradients(image):
    # Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,1]
    # Take gradint x direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1)
    # ABS value
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    # scale gradint
    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))

    thresh_max = 170
    thresh_min = 26
    sxbinary   = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    thresh = (120,255)
    HLS    = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S      = HLS[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    combined_binary = np.zeros_like(binary)
    combined_binary[(binary==1) | (sxbinary==1)] = 1

    return combined_binary

def find_lane(image, nwindows=9):
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)

    # Find left and right peak
    midpoint = np.int(histogram.shape[0]/2) # Mid point
    left_base= np.argmax(histogram[:midpoint])
    right_base= np.argmax(histogram[midpoint:]) + midpoint

    # Height of window
    window_height = np.int(image.shape[0]/nwindows)
    # Identify the x, y points of all . onseros pixls
    nonzero  = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions of all none zeros pixels
    left_current  = left_base
    right_current = right_base
    # Set width of window +/- margin
    margin = 100
    # set number of pixels found to recenter window
    minpix = 50

    left_lane_inds  = []
    right_lane_inds = []

    for window in range(nwindows):
        # identify window boundaries x/y and Left/Right
        win_y_low       = image.shape[0] - (window+1)*window_height
        win_y_high      = image.shape[0] - window*window_height
        win_xleft_low   = left_current - margin
        win_xleft_high  = left_current + margin
        win_xright_low  = right_current - margin
        win_xright_high = right_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds= ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & 
                                                     (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append new indices
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            left_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current= np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit= np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def mark_lane(image, binary_warped, left_fit, right_fit, Minv):
    # Fit polynomial for the left and right lanes 
    ploty      = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx  = left_fit[0] * ploty**2+left_fit[1]*ploty+left_fit[2]
    right_fitx = right_fit[0] * ploty**2+right_fit[1]*ploty+right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts       = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

def get_perspective_transform(image_size):
    src = np.float32([[(215, 720), (565, 470), (720, 470), (1105, 720)]])
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])

    M    = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

class LaneDetection(object):
    def __init__(self, camera):
        self.camera   = camera
        self.M       = None
        self.Minv    = None
        
    def mark_lane(self, image):
        # Calculate matrix transformation
        image_size    = (image.shape[1], image.shape[0])
        if self.M is None:
            self.M, self.Minv = get_perspective_transform(image_size)
            
        # 1. undistort image
        undistorted_image = self.camera.undistort(image)

        # 2. Gradients
        binary = gradients(undistorted_image)

        # 3. warp
        binary_warped = cv2.warpPerspective(binary, self.M, image_size, flags=cv2.INTER_LINEAR)

        # 4. Find lane
        left_fit, right_fit = find_lane(binary_warped)

        # 5. Mark lanes
        return mark_lane(undistorted_image, binary_warped, left_fit, right_fit, self.Minv)