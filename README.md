**Dehazing of Images** 
This project implements a Convolutional Neural Network (CNN) model to dehaze images. The model is trained on a dataset of hazy and clear image pairs and is capable of learning to remove haze from images.

Project Overview
This project focuses on removing haze from images using a deep learning approach. We have developed a CNN-based model that learns the mapping between hazy and clear images. The model architecture includes several convolutional, pooling, and upsampling layers, along with batch normalization, to ensure effective learning and dehazing.

Dataset
The dataset used in this project consists of paired hazy and clear images. The hazy images are used as input, while the corresponding clear images serve as the target output.

Preprocessing
Image Resizing: All images are resized to 256x256 pixels.
Normalization: Pixel values are scaled to the range [0, 1].
Directory Structure
The dataset is organized into two folders:

hazy/: Contains the hazy images.
clear/: Contains the clear (dehazed) images.
Model Architecture
The model is a deep Convolutional Neural Network (CNN) with the following layers:

Input Layer: Accepts images of size 256x256x3.
Convolutional Layers: Several layers with filter sizes of 32, 64, 80, 128, and 256.
Max Pooling Layers: Applied after each convolutional block.
Batch Normalization: To stabilize and speed up the training process.
Upsampling Layers: To gradually reconstruct the image back to its original resolution.
Output Layer: A convolutional layer with a sigmoid activation function, producing the dehazed image.
The model is compiled using the Adam optimizer and mean squared error (MSE) as the loss function.

Dependencies
Python 3.x
TensorFlow 2.x
NumPy
OpenCV
Matplotlib
Scikit-learn
You can install the required packages using the following command:
pip install -r requirements.txt

