# Vehicle Detection

## Feature Vector
 
In cell #3 of vehicle-detect.ipynb I created the feature vector that would be later used by vehicle detection classifier.  The feature vector was created by first converting color channels from RGB to YCrCb, and then calculating three components.

* A spatial color bin vector, which is a downsampled image 32x32 pixels where each color channel is converted to a 1D vector and appended together with the other two channels.

* A histogram of color channels, which is where we take each color channel separately and create a histogram (32 bins) of the color channel values.  The three resulting histograms are appended together to create one vector.

* A histogram of gradients (HOG) feature, which is a ???

The three vectors above are then concatenated together to create one single feature vector (8460 features) used for classification.


## Classifier Training

Next I trained a Linear SVM classifier using the feature vector created above on a labeled training set of vehicle and non-vehicle images.  In cell #5, I perform the following steps.

* Create a feature vectors for all of the training images
* Normalize the feature vectors to be from -0.5 - 0.5
* Create a vector (y) to store the labels 0=non-car, 1=car

Then I train the classifier on the above data.  In order to avoid overfitting I reserve 20% of data for test validation during training.  I also shuffle the data during each epoch to avoid the data ording affecting the training.

With the feature vector described above and the Linear SVM, I was able to achieve accuracy of ~99.24% on the test set.

## Sliding Window

Next I defined a function that could create windows of multiple scales (64x64, 96x96, and 128x128) and slide them over the image in an overlapping fashion.  These windows would later be used in the detection pipeline to detect vehicles of different scales and locations in the image.

The method scans the entire X range of the image since vehicles were observed in the training set throughout the entire X range, but the Y range was limited to 400-550.  By limiting the range to the vertical field of view where vehicles were likely to be we decreased the runtime of the detection pipeline.  

I ended up deciding to use three scales to search for (64x64, 96x96, and 128x128).  I arrived at these values somewhat empirically.  I viewed the supplied test images and found what the scale of the vehicles in the image were.  These three scales were a good mix to capture vehicles in the near to middle range.  They were also large enough that the runtime wasn't serverely impacted due to the creation of many small windows.  The window sizes were also have the same aspect ratio as the original training images.

I also adjusted the overlapping percentages from the default (50%) to 75%.  This was in response to missing some detections of vehicles at a medium distance.  It seems the patch would slide too far between iterations and miss the vehicle.

# False Positive Reduction
In order to avoid false positives I added a method of combining multiple overlapping detections using a 'heatmap'.  The heatmap method counts the number of detections for a region of the image, and then applies a threshold function to filter out regions that have a number of detections less than the threshold.  Once the heatmap was thresholded, I used the scipy.ndimage.measurements.label function to create a bounding box to the remaining heatmap regions.

For the video, I also added a function to accumulate the heatmaps over a series of frames, add them together, and then apply the threshold.  This served to minimize the impact of spurious detections that lasted only a single frame.

# Discussion
The main challenge I faced in this project was an oversight in the difference of range for PNG and JPG images.  Due to training on PNG images and then using JPG images for the video I found my detector performing terribly.  Once I realized the difference and scaled the JPG values from a range of (0-255) to a range of (0-1) things looked much better.

Another issue I faced is in the runtime of the pipeline.  Due to the structure of the code there are many overlapping HOG features being calculated.  This is very slow.  To increase the performance I could extract the HOG calculation to occur once per image and then just sample from that for each sliding window.  

I also notice that vehicles in the far distance are not being detected.  Maybe this is okay for some use cases, but if they need to be detected it could be done by possibly trying smaller scale windows in the sliding window function.  This would of course decrease performance substantially.


