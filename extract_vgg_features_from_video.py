#!/usr/bin/python3

from video_processor import VideoProcessor
from subtract_background_from_video import BaseVideoBackgroundSubtractor
from background_subtractor import BackgroundSubtractor
from vgg_utils import VGGFeatureExtractor
import sys
import cv2
import numpy as np
import os.path
import caffe
import getpass


class VideoVGGFeatureExtractor(BaseVideoBackgroundSubtractor):
    DRAW_COMPONENTS = False
    DRAW_BBOX = True
    DRAW_FRAME_N = True
    DRAW_VALUE = True

    @staticmethod
    def make_parser(help_string):
        parser = VideoProcessor.make_parser(help_string, with_output=False)
        BackgroundSubtractor.prep_parser(parser)
        parser.add_argument("-cpu", "--caffe_cpu", action="store_true", help="Use Caffe in CPU mode.", default=False)
        parser.add_argument("-od", "--output_datafile", default=None)
        parser.add_argument("-bc", "--boundary_check", action="store_true",
                            help="Whether to mark frame as 'subject out-of-view' for frames when the" +
                                 " subject's bounding box intersects with the frame's bounding box.")
        parser.add_argument("-v", "--vgg_model_path", type=str, default=None,
                            help="Path to the vgg model file.")
        parser.add_argument("-vm", "--vgg_model_filename", type=str, default="VGG_ILSVRC_16_layers_deploy.prototxt",
                            help="Path to the vgg model file.")
        parser.add_argument("-vp", "--vgg_pretrained_filename", type=str, default="VGG_ILSVRC_16_layers.caffemodel",
                            help="Path to the vgg model file.")
        parser.add_argument("-aug", "--augment_file", action="store_true",
                            help="Augment exisiting file instead of overwriting " +
                                 "(useful when not all features are collected)", default=False)
        parser.add_argument("-nv", "--no_vgg", action="store_true",
                            help="skip actual vgg feature extraction", default=False)
        return parser

    def __init__(self, args):
        super().__init__(args, with_video_output=False)
        if self.vgg_model_path is None:
            self.vgg_model_path = "/media/" + getpass.getuser() + "/Data/AMBR_data/ml"
        self.vgg_model_filename = os.path.join(self.vgg_model_path, self.vgg_model_filename)
        self.vgg_pretrained_filename = os.path.join(self.vgg_model_path, self.vgg_pretrained_filename)

        if self.output_datafile is None:
            self.output_datafile = "{:s}_features.npz".format(self.in_video[:-4])
        self.prev_frame_centroid = None
        if self.caffe_cpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()

        self.extractor = None
        self.blank_features = None
        if not self.no_vgg:
            self.extractor = VGGFeatureExtractor(model_file=self.vgg_model_filename,
                                                 pretrained_file=self.vgg_pretrained_filename)
            self.blank_features = self.extractor.extract_single(np.zeros((256, 256, 3), dtype=np.uint8), blobs=['fc7'])[
                'fc7']
        self.features = []
        self.present_flags = []

    def save_results(self):
        feature_results = np.array(self.features)
        present_flag_results = np.array(self.present_flags)
        output_file_path = os.path.join(self.datapath, self.output_datafile)
        if self.augment_file and os.path.isfile(output_file_path):
            archive = np.load(output_file_path)
            archive_dict = {key: value for key, value in archive.items()}
        else:
            archive_dict = {}
        if not self.no_vgg:
            archive_dict["features"] = feature_results
        archive_dict["present"] = present_flag_results
        np.savez_compressed(output_file_path, **archive_dict)

    def intersects_frame_boundary(self, x1, x2, y1, y2):
        return x1 == 0 or y1 == 0 or x2 == self.frame.shape[1] or y2 == self.frame.shape[0]

    def process_frame(self):
        mask, bin_mask, contour_found, dist, tracked_px_count, tracked_object_stats, largest_centroid \
            = self.background_subtractor.extract_tracked_object(self.frame,
                                                                self.prev_frame_centroid)
        if contour_found:
            self.prev_frame_centroid = largest_centroid
            foreground = self.frame.copy()
            foreground[bin_mask == 0] = 0
            x1 = tracked_object_stats[cv2.CC_STAT_LEFT]
            x2 = x1 + tracked_object_stats[cv2.CC_STAT_WIDTH] + 1
            y1 = tracked_object_stats[cv2.CC_STAT_TOP]
            y2 = y1 + tracked_object_stats[cv2.CC_STAT_HEIGHT] + 1
            if self.boundary_check and self.intersects_frame_boundary(x1, x2, y1, y2):
                if not self.no_vgg:
                    features = self.blank_features.copy()
                self.present_flags.append(False)
            else:
                # slice out the bounding box
                input_image = foreground[y1:y2, x1:x2]
                if not self.no_vgg:
                    features = self.extractor.extract_single(input_image, blobs=['fc7'])['fc7']
                self.present_flags.append(True)
        else:
            self.prev_frame_centroid = None
            if not self.no_vgg:
                features = self.blank_features.copy()
            self.present_flags.append(False)
        if not self.no_vgg:
            self.features.append(features)


def main():
    parser = VideoVGGFeatureExtractor.make_parser("Extract VGG features from tracked moving object in video.")
    args = parser.parse_args()
    app = VideoVGGFeatureExtractor(args)
    app.initialize()
    retval = app.run()
    if retval != 0:
        return retval
    return app.save_results()


if __name__ == '__main__':
    sys.exit(main())
