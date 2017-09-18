import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import cv2


def draw_lanes(binary_warped,left_lane_inds, right_lane_inds, left_fitx, right_fitx, margin, ploty,nonzeroy,nonzerox):
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.figure()
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def append(self, fit, xfitted):
        self.current_fit.append(fit)
        if len(self.current_fit) > 10:
            self.current_fit = self.current_fit[:-9]
        self.best_fit = np.average(self.current_fit, axis=0)
        self.recent_xfitted.append(xfitted)
        if len(self.recent_xfitted) > 10:
            self.recent_xfitted = self.recent_xfitted[:-9]
        self.bestx = np.average(self.recent_xfitted, axis=0)
        #print(self.bestx)


class SlidingWindow(object):
    def __init__(self, nwindows=9):
        self.nwindows   = nwindows
        self.left_lane  = Line()
        self.right_lane = Line()
        self.nframes    = 10
        
    def find_lane(self, image):
        histogram = np.sum(image[image.shape[0]//2:,:], axis=0)

        # Find left and right peak
        midpoint = np.int(histogram.shape[0]/2) # Mid point
        left_base= np.argmax(histogram[:midpoint])
        right_base= np.argmax(histogram[midpoint:]) + midpoint

        # Height of window
        window_height = np.int(image.shape[0]/self.nwindows)
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

        for window in range(self.nwindows):
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

        #print("leftx : {}".format(leftx.size))
        #print("rightx : {}".format(rightx.size))
        self.left_lane.detected  = (leftx.size > 5)
        self.right_lane.detected = (rightx.size > 5)


        if self.left_lane.detected & self.right_lane.detected:
            left_fit  = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            ploty      = np.linspace(0, image.shape[0]-1, image.shape[0])
            left_fitx  = left_fit[0] * ploty**2+left_fit[1]*ploty+left_fit[2]
            right_fitx = right_fit[0] * ploty**2+right_fit[1]*ploty+right_fit[2]


            draw_lanes(image,left_lane_inds, right_lane_inds, left_fitx, right_fitx, margin, ploty,nonzeroy,nonzerox)

            self.left_lane.append(left_fit, left_fitx)
            self.right_lane.append(right_fit,right_fitx)
    
        return self.left_lane, self.right_lane