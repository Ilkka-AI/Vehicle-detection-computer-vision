
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.PNG
[image2]: ./examples/hogfeatures.PNG
[image3]: ./examples/sliding_windows.PNG

[image5]: ./examples/bboxes_and_heat.PNG
[image6]: ./examples/integrateddetection.PNG

[video1]: ./project_output.mp4



### Histogram of Oriented Gradients (HOG)


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Example of a vehicle image and a non-vehicle image][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSL` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Examples of HOG features][image2]

### Classification and parameter selection

I tried various combinations of the HOG parameters and color space. I got the best classification accuracy using the HSL space, HOG features together with spatially binned color and histograms of color in the feature vector with parameter values `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. With this setup I reached the best classification accuracy with the linear SVM, 99.23% accuracy. Features were scaled to unit scale and zero-mean. 


### Sliding Window Search

I implemented a sliding window search for finding cars to go through all possible positions in an image where a car may be located. I experimented with scale values, 1,1.5,2 and 3. I chose the value 1.5 as it detected the cars best. With this value, it was possible to obtain a good thresholding for the heatmap. I searched only in the lower part of the image as this is where the cars are assumed to be located. 

![Example of car matches found by a sliding window search][image3]

I then extended the method to search in three different parts of the image using three different scales, 1,1.5 and 1.7. The three areas were:

ystart = 350 ystop = 450 scale = 0.8

ystart = 400 ystop = 600 scale = 1.5

ystart = 400 ystop = 656 scale = 1.8


### Video Implementation

Here's a [link to my video result](./project_output.mp4)

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Thresholding also removed outliers, since multiple detections were required to discover a car.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


Here are six frames and their corresponding heatmaps:

![Six frames from the video and corresponding heatmaps][image5]

### Integrating consecutive frames

In order to further improve vechile detection and outlier removal, I then combined the heatmaps from 10 consecutive frames and used a stronger threshold. I optimized the performance of my detection method by studying different thresholds in integrating consecutive frames. A threshold of 20 gave a convincing performance.

Here is the integrated heatmap from 6 frames and the output of `scipy.ndimage.measurements.label()` plotted with the corresponding bounding box:

![Integrated heatmap and bounding box representing the detected vechile][image6]


### Discussion

The method worked reasonably in detecting a vechile. I reached a 99.23% classification accuracy in the training phase, however, I still found false positive and false negative detections. False positivies came out sometimes, even after a stringent thresholding. On the other hand, cars were not detected perfectly in all frames. The stringency of the threshold is a compromize between false positives and false negatives. Averaging over the frames helped a lot in both problems but still isn't perfect. 


