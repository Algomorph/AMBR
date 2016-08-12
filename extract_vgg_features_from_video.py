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

        return parser

    def __init__(self, args):
        super().__init__(args, with_video_output=False)
        if self.output_datafile is None:
            self.output_datafile = "{:s}_vgg.npz".format(self.in_video[:-4])
        self.prev_frame_centroid = None
        if self.caffe_cpu:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        self.extractor = VGGFeatureExtractor()
        self.blank_features = self.extractor.extract_single(np.zeros((256, 256, 3), dtype=np.uint8), blobs=['fc7'])[
            'fc7']
        self.features = []

    def save_results(self):
        results = np.array(self.features)
        np.savez_compressed(os.path.join(self.datapath, self.output_datafile),
                            **{self.in_video[:-4]: results})

    def process_frame(self):
        mask, bin_mask, contour_found, dist, tracked_px_count, tracked_object_stats, largest_centroid \
            = self.background_subtractor.extract_tracked_object(self.frame,
                                                                self.prev_frame_centroid)
        if contour_found:
            self.prev_frame_centroid = largest_centroid
            foreground = self.frame.copy()
            foreground[bin_mask == 0] = 0
            x1 = tracked_object_stats[cv2.CC_STAT_LEFT]
            x2 = x1 + tracked_object_stats[cv2.CC_STAT_WIDTH]
            y1 = tracked_object_stats[cv2.CC_STAT_TOP]
            y2 = y1 + tracked_object_stats[cv2.CC_STAT_HEIGHT]
            input_image = foreground[y1:y2, x1:x2]
            features = self.extractor.extract_single(input_image, blobs=['fc7'])['fc7']
        else:
            self.prev_frame_centroid = None
            features = self.blank_features.copy()
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
