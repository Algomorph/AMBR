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
parser.add_argument("mask_file")
parser.add_argument("-o", "--out_folder", default="")
parser.add_argument("-s", "--start_from", type=int, default=0)
parser.add_argument("-e", "--end_with", type=int, default=-1)
parser.add_argument("--frame_suffix", default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    mask = cv2.imread(args.mask_file, cv2.IMREAD_COLOR)

    file_list = []

    for filename in os.listdir(args.in_folder):
        if filename.endswith(".png"):
            file_list.append(os.path.join(args.in_folder, filename))

    file_list.sort()
    last_frame = int(re.search(r"\d\d\d\d\d\d", os.path.basename(file_list[-1])[:-4]).group(0))

    if args.end_with == -1:
        args.end_with = last_frame

    if args.out_folder == "":
        args.out_folder = args.in_folder + "_masked"

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    for filename in file_list:
        frame = cv2.imread(filename)
        masked = mask & frame
        if args.frame_suffix is not None:
            out_name = "{:s}_mask.png".format(os.path.basename(filename)[:-4])
        else:
            out_name = os.path.basename(filename)
        out_path = os.path.join(args.out_folder, out_name)
        cv2.imwrite(out_path, masked)

