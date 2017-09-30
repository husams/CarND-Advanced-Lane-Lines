import numpy as np
import cv2

# Global variable
ym_per_pix = 20/720
xm_per_pix = 3.7/700

def draw_polyline(img, fit):
    """
            Draw second degree polynomial line on an image
        Args:
            binary_warped(np.array)   - (heigt x width x channels) matrix represnt image.
            fit(np.array)             - Second degree polynomial.
        Returns:
            True if the left and right fit should be accepted otherwise False.
    """
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    x = fit[0] * y**2 + fit[1] * y + fit[2]
    pts = np.array([np.transpose(np.vstack([x, y]))])
    cv2.polylines(img, np.int_(pts), isClosed=False, color=(255,255,0), thickness=3)
    return img


def draw_fit(binary_image, leftx, lefty,rightx,righty,left_fit, right_fit, margin=60):
    """
            Draw second degree polynomial line and swarxh window on an image
        Args:
            binary_warped(np.array)   - (heigt x width x channels) matrix represnt image.
            lefty(np.array)           - list of y coordinates for detected points in the
                                        left line.
            leftx(np.array)           - list of x coordinates for detected points in the
                                        left line.
            righty(np.array)          - list of y coordinates for detected points in the
                                        right line.
            rightx(np.array)          - list of x coordinates for detected points in the
                                        right line.
            left_fit(np.array)        - Second degree polynomial for left line.
            right_fit(np.array)       - Second degree polynomial for right line.
            margin(int)               - Width for the search window.
        Returns:
            Image which shows the lines and detected points.
    """
    y = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0])
   
    left_fitx  = left_fit[0] * y**2+left_fit[1]*y+left_fit[2]
    right_fitx = right_fit[0] * y**2+right_fit[1]*y+right_fit[2]

    # Create output image
    out_image = np.dstack((binary_image, binary_image, binary_image))*255
    window_img = np.zeros_like(out_image)
    
    out_image[lefty,  leftx]  = [255,0,0]
    out_image[righty, rightx] = [0,0,255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  y])))])

    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_image, 1, window_img, 0.3, 0)
    
    draw_polyline(result, left_fit)
    draw_polyline(result, right_fit)

    return result

def compute_curvature_and_distance(image, y, left_fit, right_fit):
    """
            Draw second degree polynomial line and swarxh window on an image
        Args:
            image(np.array)           - (heigt x width x 1) matrix represnt image.
            left_fit(np.array)        - Second degree polynomial for left line.
            right_fit(np.array)       - Second degree polynomial for right line.
        Returns:
            curvature and distance from center.
    """
    y      = np.linspace(0, image.shape[0]-1, image.shape[0])
    leftx  = left_fit[0]  * y**2 + left_fit[1]  * y + left_fit[2]
    rightx = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
    y_eval = np.max(y)
    height = image.shape[0] # Image height

    # Compute curvatures
    left_fit_cr  = np.polyfit(y*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(y*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curvature  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Compute curvature
    curvature = (left_curvature + right_curvature) / 2


    # Distance from the center
    car_center  = image.shape[1] / 2  # Computer center of the car
    xleft       = left_fit[0]*height**2  + left_fit[1]*height  + left_fit[2]
    xright      = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    lane_center = (xleft + xright)/2 # Compute the center of the lane
    distance    = (car_center - lane_center) * xm_per_pix # Convert from pxiels to meters

    return curvature, distance

def draw_curvatures_and_distance(image, left_fit, right_fit):
    """
            Draw curvature and distance on the output image
        Args:
            image(np.array)           - (heigt x width x channels) matrix represnt colored image.
            left_fit(np.array)        - Second degree polynomial for left line.
            right_fit(np.array)       - Second degree polynomial for right line.
        Returns:
            Image which shows the curvature and distance,
    """
    y = np.linspace(0, image.shape[0]-1, image.shape[0])

    # Calculate curvatures and distance
    curvature, distance = compute_curvature_and_distance(image, y, left_fit, right_fit)

    # Font and color
    font  = cv2.FONT_HERSHEY_DUPLEX
    color = (255,255,255)

    curvature_text = "Radius of curvature : {0:0.2f}m ".format(curvature) 
    distance_text  = "Vehicle is {0:0.3f}m {1} of the center".format(abs(distance),
                                                                         "Right" if distance > 0 else "Left")

    cv2.putText(image, curvature_text, (20,60),  font, 1.5, color, 2, cv2.LINE_AA)
    cv2.putText(image, distance_text,  (20,110), font, 1.5, color, 2, cv2.LINE_AA)

    return image

def mark_lane(image, binary_warped, left_fit, right_fit, Minv):
    """
            Draw the lane  in the output image image
        Args:
            image(np.array)           - (heigt x width x channels) matrix represnt orignal image.
            binary_warped(np.array)   - (heigt x width x 1) matrix represnt image.
            left_fit(np.array)        - Second degree polynomial for left line.
            right_fit(np.array)       - Second degree polynomial for right line.
            Minv(p.array)             - Inverse transformation matrix.
        Returns:
            Final output image with lane marked and .
    """
    # Fit polynomial for the left and right lines 
    y      = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    xleft  = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
    xright = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
  
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([xleft, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([xright, y])))])
    pts       = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    marked_image =  cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return marked_image


def find_lane_using_previous_fit(image, current_left_fit, current_right_fit, margin = 60):
    """
            Search for left and right lines based on last valid fitted second degree polynomials
        Args:
            image(np.array)            - (heigt x width x channels) matrix represnt orignal image.
            binary_warped(np.array)     - (heigt x width x 1) matrix represnt image.
            current_left_fit(np.array)  - The last valid Second degree polynomial for left line.
            current_right_fit(np.array) - The last valid Second degree polynomial for right line.
            margin(p.array)             - width of the search window.
        Returns:
            fitting for left and right lanes along with x,y coordinates for
            the discovred points.
    """
    nonzero = image.nonzero()
    y       = np.array(nonzero[0])
    x       = np.array(nonzero[1])
    

    left_lane_inds = ((x > (current_left_fit[0]*(y**2) + current_left_fit[1]*y +  current_left_fit[2] - margin)) & 
                      (x < (current_left_fit[0]*(y**2) + current_left_fit[1]*y + current_left_fit[2] + margin))) 

    right_lane_inds = ((x > (current_right_fit[0]*(y**2) + current_right_fit[1]*y + current_right_fit[2] - margin)) & 
                       (x < (current_right_fit[0]*(y**2) + current_right_fit[1]*y + current_right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = x[left_lane_inds]
    lefty = y[left_lane_inds] 
    rightx = x[right_lane_inds]
    righty = y[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, leftx, lefty, rightx, righty


def find_lines(image, nwindows=9):
    """
            Search for left and right lines using sliding windows
        Args:
            image(np.array)    - (heigt x width x 1) matrix represnt binray image.
            nwindows(int)      - number of search windows.
        Returns:
            fitting for left and right lanes along with x,y coordinates for
            the discovred points.
    """
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
    margin = 60
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
    
    return left_fit, right_fit, leftx, lefty, rightx, righty
