**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Undistorted"
[image2]: ./output_images/undistorted.png "Road Transformed"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/lane_boundary.png "Fit Visual"
[image6]: ./output_images/final_image.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 11 through 45 of the file called `calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline 

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Thresholded binary image.

I used only color thresholds based on HSV and LUV color space to generate a binary image (thresholding steps at lines 20 through 30 in `image_processing.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Perspective transform.

The code for my perspective transform includes a function called `warp()`, which appears in lines 41 through 43 in the file `image_processing.py` .  The `warp()` function takes as inputs an image (`(binary.`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src            = np.float32([[(200, 720), 
                              (570, 465), 
                              (730, 465),
                              (1200, 720)]]) 
dst            = np.float32([[(320, 720),
                              (320, 0),
                              (980, 0), 
                              (980, 720)]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 20, 720       | 
| 570, 465      | 320, 0        |
| 730, 465      | 980, 0        |
| 1200, 720     | 9980, 720     |

And here is an example showing the warped image and binary version.

![alt text][image4]

#### 4.Lane line pixels.

I used sliding window algorithm to  search the first lanes, then I used information  to search for the reset of lanes.

The code for the sliding window apper in line 152 through 216 in file `lanes_and_curvatures.py`. And the code for the second function which search based on information for last detected lines apper in line 127 through 149.

The sliding window algorithm first compute the histogram for the lower half of the binray image, to identify the start for left and right line using location for the peak value. And then  walk up the image from the bottom uing fixed sized window to identify the reset the lines. 


![alt text][image5]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.


First I fitted second degree polynomials in world coordinates space, by computing and converting list of image coordinates to world coordinates. Then used the coefficient for the fiited second degree polynomials along with equation provided in the lecture to compute curvature for left and right line.

Which allow me to compute the road curvature by taking the avarage, and the position of the vehicle with respect to center by subtracting center of the lane from the center of the car.

the code is in lines 51 through 77 in my code in `lanes_and_curvatures.py`

#### 6. Example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in  `mark_lane()` function lines 98 through 124, and function `draw_curvatures_and_distance()` lines 79 through in `lanes_and_curvatures.py`.  Here is an example of my result on a test image:


![alt text][image6]

---

### Pipeline (video)


Here's a video 

[![ADVANCE LANE DETECTION](https://img.youtube.com/vi/sp2L505b6f0/0.jpg)](https://youtu.be/sp2L505b6f0 "Advance Lane Detection")

---

### Discussion

I started by butting the first stage if pipeline which takes an input image and preduce warped binray image with littel tunning, and then tested it with sample images just to see how all fit togther.

Then I start tunining individual parts separately, where the easiest was the camera Calibration  and the hardest is thresholding where most of my time is spent. I found apply the changes in range of test images and displaying image for all stages help with testing and debuging including video.

For color thresholding I found trying to detect white and yellow lines separately works better then trying to detect both od thwn ar rhe same time, basiclly have range for yellow and one for the white then combain them.

I tried different color spaces and I had better result with using S,V from HSV and L channal from Luv.

After tunning the image procssing I start implementing sliding window algorithm and the rseet of the pipeline.

My current pipeline not robust because I'm using fix points to compute perspective transformation matrix, and I'm using global range for color thresholding 

So the pipeline will most likely fail for roads with different lane size because of the fix points, and the fix range will likely intrduce an outliers when the pioeline  process frames with shadows, different lighting condition and texture (dirt and old roads).

So to imporve the pipeline I think I need to first intrduce preprocessing stange to remove shadows, background and applying something like histogram equilization or CLAHE to handle different lighting condition and texture.

And to handle different lane size we could compute the point using ratio instead of hard coded value.