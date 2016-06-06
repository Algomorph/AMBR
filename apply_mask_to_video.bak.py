#!/usr/bin/python3

import cv2
import cve
import sys
import os
import os.path
import argparse as ap
import re
from multiprocessing import cpu_count

parser = ap.ArgumentParser("Run background subtraction on frames in a folder.")
parser.add_argument("in_video")
parser.add_argument("mask_file")
parser.add_argument("-o", "--out_video", default="")
parser.add_argument("-s", "--start_from", type=int, default=0)
parser.add_argument("-e", "--end_with", type=int, default=-1)
parser.add_argument("-c", "--frame_count", type=int, default=-1)


def process_frame(cap, writer, mask):
    read_succeeded, frame = cap.read()
    masked = mask & frame
    writer.write(masked)

def update_progress(amount_done):
    sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amount_done * 50), amount_done * 100))


def main():
    args = parser.parse_args()

    mask = cv2.imread(args.mask_file, cv2.IMREAD_COLOR)

    cap = cv2.VideoCapture(args.in_video)
    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

    if args.end_with == -1:
        args.end_with = last_frame
    else:
        if args.end_with > last_frame:
            print(
                "Warning: specified end frame ({:d})is beyond the last video frame ({:d}). Stopping after last frame.".format(
                    args.end_with, last_frame))
            args.end_with = last_frame

    if args.out_video == "":
        args.out_video = args.in_video[:-4] + "_masked.mp4"

    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)
    writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, cpu_count())

    if args.start_from > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_from)

    total_frame_span = args.end_with - args.start_from
    frame_counter = 0
    if args.frame_count == -1:
        cur_frame_number = args.start_from
        while cur_frame_number < args.end_with:
            process_frame(cap, writer, mask)
            frame_counter += 1
            amount_done = frame_counter / total_frame_span
            update_progress(amount_done)
            cur_frame_number += 1
    else:
        frame_interval = total_frame_span // args.frame_count
        for i_frame in range(args.start_from, args.end_with, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            process_frame(cap, writer, mask)
            frame_counter += 1
            amount_done = frame_counter / args.frame_count
            update_progress(amount_done)


    cap.release()
    writer.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
