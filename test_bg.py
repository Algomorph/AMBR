#!/usr/bin/python3

from subtract_background_from_video import VideoBackgroundSubtractor, Label
import cv2
import sys
import numpy as np
import cve
import os.path
import os
import re
import math
import sys
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def error_by_dist(pos1, pos2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


def generate_dist_kernel(size):
    k = np.zeros((size, size))
    for row in range(k.shape[0]):
        for col in range(k.shape[1]):
            k[row, col] = error_by_dist((col, row), (3, 3))
    # make kernel 0 in the center and approach 1 toward the edges
    k = k.max() - k
    return k


class VideoBgsTester(VideoBackgroundSubtractor):
    @staticmethod
    def make_parser(help_string):
        parser = VideoBackgroundSubtractor.make_parser(help_string)
        parser.add_argument("ground_truth_folder")
        parser.add_argument("-of", "--output_file", default="test_bg_out.yaml")
        return parser

    def __init__(self, args):
        super().__init__(args, "foreground")
        if not os.path.exists(self.ground_truth_folder):
            raise "Folder with ground truth images ({:s}) not found.".format(self.ground_truth_folder)
        # get test frame filenames

        files = os.listdir(self.ground_truth_folder)
        files.sort()
        # get file sizes
        for i_file in range(len(files)):
            files[i_file] = (files[i_file], os.path.getsize(os.path.join(self.ground_truth_folder, files[i_file])))
        # filter ready test frames out (these will have bytesize below 10000
        self.ground_truth_frame_filenames = [file[0] for file in files if file[1] < 10000]
        self.ground_truth_frame_numbers = [int(re.search(r'\d\d\d\d\d\d', file).group(0)) for file
                                           in self.ground_truth_frame_filenames]
        self.gt_frame_ix = 0
        self.smoothing_kernel = generate_dist_kernel(7)
        self.cum_fp = 0.
        self.cum_fn = 0.
        self.cum_wfp = 0.
        self.cum_wfn = 0.
        self.tested_frame_coutner = 0
        self.args_dict = vars(args)

    def __del__(self):
        super().__del__()
        self.mask_writer.release()

    def process_frame(self):
        super().process_frame()
        if self.cur_frame_number == self.ground_truth_frame_numbers[self.gt_frame_ix]:
            # we have struck upon a frame we can evaluate against ground truth
            gt_file_path = os.path.join(self.ground_truth_folder, self.ground_truth_frame_filenames[self.gt_frame_ix])
            gt_mask = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)
            self.gt_frame_ix += 1  # advance for next hit
            test_mask = self.mask.copy()
            test_mask[test_mask < Label.PERSISTENCE_LABEL.value] = 0
            test_mask[test_mask >= Label.PERSISTENCE_LABEL.value] = 1
            gt_mask[gt_mask == 255] = 1
            test_mask = test_mask.astype(np.int8)  # to allow subtraction
            errors = test_mask - gt_mask
            false_positives = errors.copy()
            false_negatives = errors.copy()
            false_positives[false_positives == -1] = 0
            false_negatives[false_negatives == 1] = 0
            n_fp = false_positives.sum()
            n_fn = -false_negatives.sum()

            penalty_map = cv2.filter2D(gt_mask, cv2.CV_32FC1, self.smoothing_kernel)
            cv2.normalize(penalty_map, penalty_map, 0, 1.0, cv2.NORM_MINMAX)
            weighted_fn = (penalty_map[false_negatives == -1]).sum()
            penalty_map = penalty_map.max() - penalty_map  # invert
            weighted_fp = (penalty_map[false_positives == 1]).sum()

            self.cum_fp += n_fp
            self.cum_fn += n_fn
            self.cum_wfn += weighted_fn
            self.cum_wfp += weighted_fp
            self.tested_frame_coutner += 1

    def save_results(self, verbose):
        if self.tested_frame_coutner > 0:
            ave_fp = self.cum_fp / self.tested_frame_coutner
            ave_fn = self.cum_fn / self.tested_frame_coutner
            ave_wfp = self.cum_wfp / self.tested_frame_coutner
            ave_wfn = self.cum_wfn / self.tested_frame_coutner
        else:
            ave_fp = 0.
            ave_fn = 0.
            ave_wfp = 0.
            ave_wfn = 0.
        if verbose:
            print("Tested frame count: {:d}".format(self.tested_frame_coutner))
            print("Avg. false positives: {:.2f}\nAvg. false negatives: {:.2f}".format(
                ave_fp, ave_fn))
            print("Avg. weighted false positives: {:.2f}\nAvg. weighted false negatives: {:.2f}".format(
                ave_wfp, ave_wfn))
        out = {"average_false_positives": float(ave_fp),
               "average_false_negatives": float(ave_fn),
               "average_weighted_false_positives": float(ave_wfp),
               "average_weighted_false_negatives": float(ave_wfn),
               "tested_frame_count": self.tested_frame_coutner,
               "args": self.args_dict}
        out_file = open(self.output_file, "w", encoding="utf_8")
        dump(out, out_file, Dumper=Dumper)
        out_file.close()
        return 0


def main():
    parser = VideoBgsTester.make_parser("Subtract background from video and evaluate against the " +
                                        "provided still silhouettes.")
    args = parser.parse_args()
    app = VideoBgsTester(args)
    app.initialize()
    retval = app.run()
    if retval != 0:
        return retval
    return app.save_results(True)


if __name__ == '__main__':
    sys.exit(main())
