#!/usr/bin/python3

from video_processor import VideoProcessor
from subtract_background_from_video import BaseVideoBackgroundSubtractor
from background_subtractor import BackgroundSubtractor
from vgg_utils import VGGFeatureExtractor
import sys
import cv2
import numpy as np
import os.path
from enum import Enum


class VGGFeatureExtractor(BaseVideoBackgroundSubtractor):
    DRAW_COMPONENTS = False
    DRAW_BBOX = True
    DRAW_FRAME_N = True
    DRAW_VALUE = True

    @staticmethod
    def make_parser(help_string):
        parser = VideoProcessor.make_parser(help_string)
        BackgroundSubtractor.prep_parser(parser)
        return parser

    def __init__(self, args):
        super().__init__(args, with_output_video=False)
        self.prev_frame_centroid = None
        self.extractor = VGGFeatureExtractor()

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
            input_image = foreground[x1:x2, y1:y2]
        else:
            self.prev_frame_centroid = None
            total_darkness = np.zeros((256, 256, 3), dtype=np.uint8)
            input_image = total_darkness
        self.extractor.extract_single(blobs=['fc7'])


def main():
    parser = VGGFeatureExtractor.make_parser("Extract VGG features from tracked moving object in video.")
    args = parser.parse_args()
    app = VGGFeatureExtractor(args)
    app.initialize()
    retval = app.run()
    if retval != 0:
        return retval
    return app.save_results(True)


if __name__ == '__main__':
    sys.exit(main())
