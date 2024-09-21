# Course Assignments

This repository contains the implementation of Homework 1 for the **Introduction to Image Processing, Computer Vision, and Deep Learning** course at NCKU CSIE, 2023.
More infomation :[Slide](https://github.com/Iane14093051/OpenCVDL_Hw1_2023/raw/refs/heads/main/OpenCv_Hw_1_Q_20231024_V1B4.pptx)

## Environment

- OS: Windows 10
- Python Version: 3.8

## Setup Instructions

1. Clone the repository:
   ```bash
   $ git clone https://github.com/Iane14093051/OpenCVDL_Hw1_2023.git
   ```
2. Install the required dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   $ python hw1.0.py
   ```

## Application Features

Once the application (`hw1.0.py`) is running, the UI is divided into five sections: **Image Processing**, **Image Smoothing**, **Edge Detection**, **Transforms**, and **VGG19**.

## Train VGG19 model
   ```bash
   $ python py.py
   ```


### 1. Image Processing
Offers three features:
- **Color Separation**: Extract and display the three BGR channels of "rgb.jpg" as separate grayscale images.
- **Color Transformation**: Convert "rgb.jpg" to grayscale (I1) using the OpenCV function cv2.cvtColor(), and then create an averaged image (I2) by merging the separated BGR channels.
- **Color Extraction**: Transform "rgb.jpg" to HSV format, extract a Yellow-Green mask, and create a modified image without yellow and green regions.

### 2. Image Smoothing
Provides three smoothing filters:
- **Gaussian Blur**: Adjust the Gaussian blur radius using a track bar.
- **Bilateral Filter**: Adjust the bilateral filter's radius using a track bar.
- **Median Filter**: Adjust the radius of the median filter using a track bar.

### 3. Edge Detection
Four options for edge detection:
- **Sobel X**: Detects vertical edges using the Sobel X operator.
- **Sobel Y**: Detects horizontal edges using the Sobel Y operator.
- **Combination and Threshold**: Combines Sobel X and Sobel Y images, applying a threshold to the result.
- **Gradient Angle**: Displays image areas where the gradient angle is between 120-180° and 210-330°.

### 4. Transforms
Performs image transformations:
- **Rotation, Scaling, and Translation**: Apply rotation, scaling, and translation to an image with adjustable parameters.

### 5. VGG19
Deep learning features using the VGG19 model:
- **Show Augmented Images**: Displays nine augmented images from the `Q5_image/Q5_1` directory.
- **Show VGG19 Model Structure**: Displays the architecture of the VGG19 with Batch Normalization.
- **Show Training/Validation Accuracy and Loss**: Displays a graph of training and validation accuracy/loss.
- **Inference**: Load an image and use the trained model to classify it, displaying the predicted class label and probability distribution.
