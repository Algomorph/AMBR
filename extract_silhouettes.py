#!/usr/bin/python3
from subtract_background_from_video import VideoBackgroundSubtractor, Label
import sys
import cv2
import numpy as np
import os.path
from enum import Enum


class CcThreshold(Enum):
    HIDDEN = 1200
    BBOX_THRESH = 6500
    TRACK_DIST_THRESH = 60


class SilhouetteExtractor(VideoBackgroundSubtractor):
    DRAW_COMPONENTS = False
    DRAW_BBOX = True
    DRAW_FRAME_N = True

    @staticmethod
    def make_parser(help_string):
        parser = VideoBackgroundSubtractor.make_parser(help_string)
        parser.add_argument("-od", "--output_datafile", default="result.npz")
        return parser

    def __init__(self, args):
        super().__init__(args, "silhouettes")
        self.result_data = []
        self.prev_frame_success = False

    def extract_foreground_mask(self):
        super().extract_foreground_mask()
        bin_mask = self.mask.copy()
        bin_mask[bin_mask < Label.PERSISTENCE_LABEL.value] = 0
        bin_mask[bin_mask > 0] = 1
        labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, ltype=cv2.CV_16U)[1:4]
        self.has_contour = False
        if len(stats) > 1:
            # initially, just grab the biggest connected component
            ix_of_tracked_component = np.argmax(stats[1:, 4]) + 1
            centroid_slice = centroids[ix_of_tracked_component]
            cur_largest_centroid = (int(round(centroid_slice[0])), int(round(centroid_slice[1])))
            tracking_ok = True
            if self.prev_frame_success:
                a = self.largest_cc_centroid
                b = cur_largest_centroid
                dist = np.linalg.norm((a[0] - b[0], a[1] - b[1]))
                # check to make sure we're not too far from the previously-detected blob
                if dist > 50:
                    dists = np.linalg.norm(centroids - b, axis=1)
                    ix_of_tracked_component = np.argmin(dists)
                    if dists[ix_of_tracked_component] > CcThreshold.TRACK_DIST_THRESH:
                        self.prev_frame_success = False
                        tracking_ok = False
                    centroid_slice = centroids[ix_of_tracked_component]
                    cur_largest_centroid = (int(round(centroid_slice[0])), int(round(centroid_slice[1])))

            tracked_px_count = stats[ix_of_tracked_component, 4]
            self.largest_cc_stats = stats[ix_of_tracked_component]
            bbox_w_h_ratio = self.largest_cc_stats[cv2.CC_STAT_WIDTH] / self.largest_cc_stats[cv2.CC_STAT_HEIGHT]
            self.has_contour = tracked_px_count > CcThreshold.HIDDEN.value and tracking_ok
            self.result_data.append(
                (self.cur_frame_number, tracked_px_count, bbox_w_h_ratio, centroid_slice[0], centroid_slice[1],
                 int(self.has_contour)))
            if self.has_contour:
                bin_mask[labels != ix_of_tracked_component] = 0
                self.bin_mask = bin_mask
                self.mask[bin_mask == 0] = 0
                self.cc_stats = stats
                self.largest_cc_centroid = cur_largest_centroid
                self.prev_frame_success = True
            else:
                self.prev_frame_success = False
        else:
            self.prev_frame_success = False
            self.result_data.append((self.cur_frame_number, 0, -1.0, -1., -1., 0.0))

    def extract_foreground(self):
        super().extract_foreground()
        foreground = self.foreground
        if self.has_contour:
            contours = cv2.findContours(self.bin_mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[1]
            for i_contour in range(0, len(contours)):
                cv2.drawContours(foreground, contours, i_contour, (0, 255, 0))

            if SilhouetteExtractor.DRAW_BBOX:
                x1 = self.largest_cc_stats[cv2.CC_STAT_LEFT]
                x2 = x1 + self.largest_cc_stats[cv2.CC_STAT_WIDTH]
                y1 = self.largest_cc_stats[cv2.CC_STAT_TOP]
                y2 = y1 + self.largest_cc_stats[cv2.CC_STAT_HEIGHT]
                cv2.rectangle(foreground, (x1, y1), (x2, y2), color=(0, 0, 255))
                cv2.drawMarker(foreground, self.largest_cc_centroid, (0, 0, 255), cv2.MARKER_CROSS, 11)
                bbox_w_h_ratio = self.largest_cc_stats[cv2.CC_STAT_WIDTH] / self.largest_cc_stats[cv2.CC_STAT_HEIGHT]
                cv2.putText(foreground, "BBOX w/h ratio: {0:.4f}".format(bbox_w_h_ratio), (x1, y1 - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

            if SilhouetteExtractor.DRAW_COMPONENTS:
                stats = self.cc_stats
                for i_comp in range(1, len(stats)):
                    x1 = stats[i_comp, cv2.CC_STAT_LEFT]
                    x2 = x1 + stats[i_comp, cv2.CC_STAT_WIDTH]
                    y1 = stats[i_comp, cv2.CC_STAT_TOP]
                    y2 = y1 + stats[i_comp, cv2.CC_STAT_HEIGHT]
                    cv2.rectangle(foreground, (x1, y1), (x2, y2), color=(0, 0, 255))

            if SilhouetteExtractor.DRAW_FRAME_N:
                cv2.putText(foreground, str(self.cur_frame_number), (0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0))

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
