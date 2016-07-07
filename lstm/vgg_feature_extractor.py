## analogicalnexus159@gmail.com
import numpy as np
import matplotlib.pyplot as plt


# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/analogicalnexus/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import cv2

# use caffe.set_mode_gpu() - use this for gpu processing
caffe.set_mode_cpu()

model_def = caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt'
model_weights = '/media/analogicalnexus/Data/VGG_ILSVRC_16_layers.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input 2
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 224x224 for vgg16 model
#image = caffe.io.load_image(caffe_root + 'examples/images/dataset_224/09.jpg') - default caffe io
image = cv2.imread(caffe_root + 'examples/images/dataset_224/09.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
plt.show()
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform feature extraction
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch


for layer_name, blob in net.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))

for layer_name, param in net.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))

feat = net.blobs['fc7'].data[0] # features from fc7 layer
print(feat.shape) # size of the feature

## plotting 4096 features in histogram
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()