from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
from multiprocessing import cpu_count
import sys
import time
import datetime
import os.path
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class VideoProcessor(metaclass=ABCMeta):
    @staticmethod
    def update_progress(amount_done, start_time):
        current_time = time.time()
        elapsed_time = current_time - start_time
        amount_remaining = 1.0 - amount_done
        est_time_remaining = elapsed_time / amount_done * amount_remaining
        eta = datetime.datetime.fromtimestamp(current_time + est_time_remaining)
        sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%; ETA: {2:s}".format('#' * int(amount_done * 50),
                                                                             amount_done * 100,
                                                                             eta.strftime("%H:%M:%S")))

    @staticmethod
    def make_parser(help_string):
        parser = ArgumentParser(help_string, formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument("in_video")
        parser.add_argument("-o", "--out_video", default="")
        parser.add_argument("-s", "--start_from", type=int, default=0)
        parser.add_argument("-e", "--end_with", type=int, default=-1)
        parser.add_argument("-c", "--frame_count", type=int, default=-1)
        parser.add_argument("-np", "--no-progress-bar", action='store_true', default=False)
        return parser

    def __init__(self, args, out_postfix):
        self.datapath = "./"
        if os.path.exists("settings.yaml"):
            stream = open("settings.yaml", mode='r')
            self.settings = load(stream, Loader=Loader)

            stream.close()
            self.datapath = self.settings['datapath']

        self.cap = cv2.VideoCapture(os.path.join(self.datapath, args.in_video))
        last_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

        if args.end_with == -1:
            args.end_with = last_frame
        else:
            if args.end_with > last_frame:
                print(("Warning: specified end frame ({:d})is beyond the last video frame" +
                       " ({:d}). Stopping after last frame.")
                      .format(args.end_with, last_frame))
                args.end_with = last_frame

        if args.out_video == "":
            args.out_video = args.in_video[:-4] + "_" + out_postfix + ".mp4"

        self.writer = cv2.VideoWriter(os.path.join(self.datapath, args.out_video),
                                      cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                                      self.cap.get(cv2.CAP_PROP_FPS),
                                      (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      True)
        self.writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, cpu_count())
        self.__dict__.update(vars(args))
        self.frame = None
        self.cur_frame_number = None

    def __del__(self):
        self.cap.release()
        self.writer.release()

    def run(self, verbose=True):
        if verbose:
            print("Processing video...")
        start_time = time.time()
        total_frame_span = self.end_with - self.start_from
        frame_counter = 0

        if self.start_from > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_from)

        if self.frame_count == -1:
            cur_frame_number = self.start_from
            while cur_frame_number < self.end_with:
                self.frame = self.cap.read()[1]
                self.cur_frame_number = cur_frame_number
                self.process_frame()
                frame_counter += 1
                amount_done = frame_counter / total_frame_span
                cur_frame_number += 1
                if not self.no_progress_bar:
                    VideoProcessor.update_progress(amount_done, start_time)
        else:
            frame_interval = total_frame_span // self.frame_count
            for i_frame in range(self.start_from, self.end_with, frame_interval):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                self.frame = self.cap.read()[1]
                self.cur_frame_number = i_frame
                self.process_frame()
                frame_counter += 1
                amount_done = frame_counter / self.frame_count
                if not self.no_progress_bar:
                    VideoProcessor.update_progress(amount_done, start_time)
        if not self.no_progress_bar:
            sys.stdout.write("\n")  # terminate progress bar
        return 0

    @abstractmethod
    def process_frame(self):
        pass
