## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell after the initial imports of the IPython notebook located in "./main.ipynb".

We first need to correct for camera distortion, the effect in which objects, especially those close to the edges, can be stretched or skewed in various ways. This can change the apparent shape, size and location of various objects in the image.

To correct the distortion, I utilize an OpenCV function that calculates r, the known distance between a point in an undistorted image and the center of image distortion. To accomplish this, we'll read in a series of chessboard images at various shapes and angles. I then call the OpenCV functions findChessboardCorners() and drawChessboardCorners() to automatically find and draw corners in an image of a chessboard pattern. I map the corners of the 2D image, called image points, to the 3D corners of the real undistorted image, object points.

I then used the output `object points` and `image points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. For the gradient, I specifically applied the Sobel operator in the x-direction and computed the resulting magnitude. For the color gradient, I first converted to HLS color format to utilize the S channel values, which do a better job of retaining lane lines under different color and contrast conditions.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the 4th code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane-line pixels, I initially plotted a histogram of binary activations along the x-columns of the image to clearly identify the left and right lane x positions. From this starting point, I used a sliding window technique to move upward along the image and define the rest of the lane pixel positions. I also periodically recalculated the center position of the lanes when sufficient pixels were found. 

I then fitted a polynomial function given the lane line pixels I found. To improve the performance of the algorithm, after I found the polynomial fit from the initial frame, I used these values in subsequent frames to conduct a targeted search for lane line pixels around said line. 

However, this technique will fail when we arrive at the left or right edge of the image. To account for this, I implemented a recalibration test every frame in which I compared the lane width and radius with sensible values. If the values fail the test, I revert to the first step of the polynomial fit. 


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This code is computed in the `calculate_curve_in_meters` and `calculate_distance_width` function in the notebook. Specifically, to compute the lane radius of the polynomial, I took advantage of the radius formula given the polynomial fit. 
​   

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this is in the `draw_lanes_on_image` function in the notebook. It takes advantage of the inverse matrix function provided by the cv2 getPerspectiveTransform function. 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


I ran into the expected issue of the algorithm failing given changing lighting conditions and also when the road took steep turns. One feature that improved robustness was implementing a recalibration_test for every frame. If the values for lane radius or width did not seem sensible, I recomputed the entire polynomial fit from scratch. Another feature that improved robustness was implementing a weighted average of the most recent x-values when using the polynomial fit from the previous frame. This was done in the `update_smooth_fit` function. 

However, in the harder_challenge_video, the algorithm still fails for very sharp turns. There seems to be a distinct lag in recalculating the changing radius curvature. 