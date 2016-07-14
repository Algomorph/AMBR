#!/usr/bin/python3
from subtract_background_from_video import VideoBackgroundSubtractor, MaskLabel
import sys
import cv2
import numpy as np
import os.path
from enum import Enum


class SilhouetteExtractor(VideoBackgroundSubtractor):
    DRAW_COMPONENTS = False
    DRAW_BBOX = True
    DRAW_FRAME_N = True
    DRAW_VALUE = True

    @staticmethod
    def make_parser(help_string):
        parser = VideoBackgroundSubtractor.make_parser(help_string)
        parser.add_argument("-od", "--output_datafile", default="result.npz")
        return parser

    def __init__(self, args):
        super().__init__(args, "silhouettes")
        self.result_data = []
        self.prev_frame_centroid = None

    def process_frame(self):
        mask, bin_mask, contour_found, dist, tracked_px_count, tracked_object_stats, largest_centroid \
            = self.background_subtractor.extract_tracked_object(self.frame,
                                                                self.prev_frame_centroid)
        self.mask = mask
        foreground = super().extract_foreground()
        if contour_found:
            self.draw_silhouette(foreground, bin_mask, tracked_object_stats, largest_centroid, dist)
            self.prev_frame_centroid = largest_centroid
            bbox_w_h_ratio = tracked_object_stats[cv2.CC_STAT_WIDTH] / tracked_object_stats[cv2.CC_STAT_HEIGHT]
            self.result_data.append((self.cur_frame_number, tracked_px_count, bbox_w_h_ratio, largest_centroid[0],
                                     largest_centroid[1], float(contour_found)))
        else:
            self.prev_frame_centroid = None
            self.result_data.append((self.cur_frame_number, 0, -1.0, -1., -1., 0.0))
        self.mask_writer.write(mask)
        self.foreground_writer.write(foreground)

    @staticmethod
    def __to_int_tuple(np_array):
        return int(round(np_array[0])), int(round(np_array[1]))

    def draw_silhouette(self, foreground, bin_mask, tracked_object_stats, centroid, value):
        contours = cv2.findContours(bin_mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[1]
        for i_contour in range(0, len(contours)):
            cv2.drawContours(foreground, contours, i_contour, (0, 255, 0))

        if SilhouetteExtractor.DRAW_BBOX:
            x1 = tracked_object_stats[cv2.CC_STAT_LEFT]
            x2 = x1 + tracked_object_stats[cv2.CC_STAT_WIDTH]
            y1 = tracked_object_stats[cv2.CC_STAT_TOP]
            y2 = y1 + tracked_object_stats[cv2.CC_STAT_HEIGHT]
            cv2.rectangle(foreground, (x1, y1), (x2, y2), color=(0, 0, 255))
            cv2.drawMarker(foreground, SilhouetteExtractor.__to_int_tuple(centroid), (0, 0, 255), cv2.MARKER_CROSS, 11)
            bbox_w_h_ratio = tracked_object_stats[cv2.CC_STAT_WIDTH] / tracked_object_stats[cv2.CC_STAT_HEIGHT]
            cv2.putText(foreground, "BBOX w/h ratio: {0:.4f}".format(bbox_w_h_ratio), (x1, y1 - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        #
        # if SilhouetteExtractor.DRAW_COMPONENTS:
        #     stats = self.cc_stats
        #     for i_comp in range(1, len(stats)):
        #         x1 = stats[i_comp, cv2.CC_STAT_LEFT]
        #         x2 = x1 + stats[i_comp, cv2.CC_STAT_WIDTH]
        #         y1 = stats[i_comp, cv2.CC_STAT_TOP]
        #         y2 = y1 + stats[i_comp, cv2.CC_STAT_HEIGHT]
        #         cv2.rectangle(foreground, (x1, y1), (x2, y2), color=(0, 0, 255))

        if SilhouetteExtractor.DRAW_FRAME_N:
            cv2.putText(foreground, str(self.cur_frame_number), (0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0))
        if SilhouetteExtractor.DRAW_VALUE:
            cv2.putText(foreground, str(value), (0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0))

    def save_results(self, verbose=False):
        largest_component_sizes = np.array(self.result_data)
        np.savez_compressed(os.path.join(self.datapath, self.output_datafile), frame_data=largest_component_sizes)
        return 0


def main():
    parser = SilhouetteExtractor.make_parser("Extract foreground silhouettes using a combination of computer vision" +
                                             " techniques.")
    args = parser.parse_args()
    app = SilhouetteExtractor(args)
    app.initialize()
    retval = app.run()
    if retval != 0:
        return retval
    return app.save_results(True)


if __name__ == '__main__':
    sys.exit(main())
