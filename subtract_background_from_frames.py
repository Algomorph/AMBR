#!/usr/bin/python3

import cv2
import cve
import sys
import os
import os.path
import argparse as ap
import re

parser = ap.ArgumentParser("Run background subtraction on frames in a folder.")
parser.add_argument("in_folder")
parser.add_argument("-o", "--out_folder", default="")
parser.add_argument("-s", "--start_from", type=int, default=0)
parser.add_argument("-e", "--end_with", type=int, default=-1)
parser.add_argument("--fps", type=float, default=60)
parser.add_argument("--fg_threshold", type=int, default=15)
parser.add_argument("--association_threshold", type=int, default=5)
parser.add_argument("--sampling_period", type=float, default=250.)
parser.add_argument("--min_bin_height", type=int, default=2)
parser.add_argument("--num_samples", type=int, default=30)
parser.add_argument("--alpha", type=int, default=30)
parser.add_argument("--beta", type=int, default=30)

if __name__ == '__main__':
    args = parser.parse_args()

    file_list = []

    for filename in os.listdir(args.in_folder):
        if filename.endswith(".png"):
            file_list.append(os.path.join(args.in_folder, filename))

    file_list.sort()
    last_frame = int(re.search(r"\d\d\d\d\d\d", os.path.basename(file_list[-1])[:-4]).group(0))

    if args.end_with == -1:
        args.end_with = last_frame

    if args.out_folder == "":
        args.out_folder = args.in_folder + "_fgmask"

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    imbs = cve.BackgroundSubtractorIMBS()
    for filename in file_list:
        frame = cv2.imread(filename)
        mask = imbs.apply(frame)
        cv2.imwrite(os.path.join(args.out_folder, "{:s}_mask.png".format(os.path.basename(filename)[:-4])), mask)
