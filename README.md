# Semantic Segmentation
### Introduction
This project uses Fully Convolutional Networks (FCNs) to classify pixels as belonging to a road. It uses a segmentation architecture inspired by [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) developed at UC Berkeley.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

 For simplest setup use the environment-gpu.yml virtual environment configuration found at [Udacity's starter kit repo](https://github.com/udacity/CarND-Term1-Starter-Kit) along with anaconda. 
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

