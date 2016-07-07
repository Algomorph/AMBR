"""
FeatExtractor is a feature extraction specialization of Net.
"""

import numpy as np
import cv2
import caffe
import time

import getpass
import os.path


def transform_image(img, over_sample=False, mean_pix=[103.939, 116.779, 123.68], image_dim=256, crop_dim=224):
    # convert to BGR
    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
    # resize image, the shorter side is set to image_dim
    if img.shape[0] < img.shape[1]:
        # Note: OpenCV uses width first...
        dsize = (int(np.floor(float(image_dim) * img.shape[1] / img.shape[0])), image_dim)
    else:
        dsize = (image_dim, int(np.floor(float(image_dim) * img.shape[0] / img.shape[1])))
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)

    # convert to float32
    img = img.astype(np.float32, copy=False)

    if over_sample:
        imgs = np.zeros((10, crop_dim, crop_dim, 3), dtype=np.float32)
    else:
        imgs = np.zeros((1, crop_dim, crop_dim, 3), dtype=np.float32)

    # crop
    indices_y = [0, img.shape[0] - crop_dim]
    indices_x = [0, img.shape[1] - crop_dim]
    center_y = np.floor(indices_y[1] / 2)
    center_x = np.floor(indices_x[1] / 2)

    imgs[0] = img[center_y:center_y + crop_dim, center_x:center_x + crop_dim, :]
    if over_sample:
        curr = 1
        for i in indices_y:
            for j in indices_x:
                imgs[curr] = img[i:i + crop_dim, j:j + crop_dim, :]
                imgs[curr + 5] = imgs[curr, :, ::-1, :]
                curr += 1
        imgs[5] = imgs[0, :, ::-1, :]

    # subtract mean
    for c in range(3):
        imgs[:, :, :, c] = imgs[:, :, :, c] - mean_pix[c]
    # reorder axis
    return np.rollaxis(imgs, 3, 1)


class VGGFeatureExtractor(caffe.Net):
    """
    Calls caffe_io to convert video/images
    and extract embedding features
    """
    default_vgg_model_path = "/media/" + getpass.getuser() + "/Data/AMBR_data/ml_models"
    default_model_definition_file = "VGG_ILSVRC_16_layers_deploy.prototxt"
    default_pretrained_model_file = "VGG_ILSVRC_16_layers.caffemodel"
    jesu9_mean = [103.939, 116.779, 123.68]
    ilsvrc_2012_mean = [104.00698793, 116.66876762, 122.67891434]

    def __init__(self, model_file=os.path.join(default_vgg_model_path, default_model_definition_file),
                 pretrained_file=os.path.join(default_vgg_model_path, default_model_definition_file), img_dim=256,
                 crop_dim=224, mean=ilsvrc_2012_mean, oversample=False):
        super().__init__(self, model_file, pretrained_file, caffe.TEST)
        self.img_dim = img_dim
        self.crop_dim = crop_dim
        self.mean = mean
        self.oversample = oversample
        self.batch_size = 10  # hard coded, same as oversample patches

    def extract(self, images, blobs=['fc6', 'fc7']):
        features = {}
        for blob in blobs:
            features[blob] = []
        for img in images:
            data = transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)
            # Use forward all to do the padding
            out = self.forward_all(**{self.inputs[0]: data, 'blobs': blobs})
            for blob in blobs:
                feat = out[blob]
                if self.oversample:
                    feat = feat.reshape((len(feat) / self.batch_size, self.batch_size, -1))
                    feat = feat.mean(1)
                features[blob].append(feat.flatten())
        return features

    def extract_single(self, img, blobs=['fc6', 'fc7']):
        features = {}
        for blob in blobs:
            features[blob] = []
        data = transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)
        out = self.forward(**{self.inputs[0]: data, 'blobs': blobs})
        for blob in blobs:
            feat = out[blob]
            if self.oversample:
                feat = feat.reshape((len(feat) / self.batch_size, self.batch_size, -1))
                feat = feat.mean(1)
            features[blob].append(feat.flatten())
        return features

    def _process_batch(self, data, feats, blobs):
        if data is None:
            return
        out = self.forward_all(**{self.inputs[0]: data, 'blobs': blobs})
        for blob in blobs:
            feat = out[blob]
            for i in range(data.shape[0]):
                feats[blob].append(feat[:, i, :].flatten())

    def extract_batch(self, images, blobs=['fc6', 'fc7']):
        if self.oversample:  # Each oversampled image is a batch
            return self.extract(images, blobs)
        features = {}
        for blob in blobs:
            features[blob] = []
        data = None
        for img in images:
            if data is None:
                data = transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)
            else:
                data = np.vstack((data, transform_image(img, self.oversample, self.mean, self.img_dim, self.crop_dim)))
            if data.shape[0] == self.batch_size:
                self._process_batch(data, features, blobs)
                data = None
        self._process_batch(data, features, blobs)
        return features


if __name__ == '__main__':
    caffe.set_mode_gpu()
    model_file = './caffe_models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt'
    pretrained_file = './caffe_models/vgg_16/VGG_ILSVRC_16_layers.caffemodel'
    img_list = 'flickr8k_images_fullpath.lst'
    start = time.time()
    extractor = VGGFeatureExtractor(model_file, pretrained_file, oversample=False)
    print('intitialization time(s):', time.time() - start)
    with open(img_list) as f:
        img_names = [l.rstrip() for l in f]
    images = []
    for i in range(13):
        img_name = img_names[i]
        img = cv2.imread(img_name)
        images.append(img)
    start = time.time()
    feats1 = extractor.extract(images)
    print('non-batch extraction time(s):', time.time() - start)
    start = time.time()
    feats2 = extractor.extract_batch(images)
    print('batch extraction time(s):', time.time() - start)

    print(len(feats1['fc6']), len(feats2['fc6']))
    for i in range(len(feats1['fc6'])):
        print
        feats1['fc6'][i].shape
        print(feats1['fc6'][i] == feats2['fc6'][i]).all()
