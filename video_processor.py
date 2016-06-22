from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
from multiprocessing import cpu_count
import sys
import time
import datetime
import os.path
from getpass import getuser
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
        self.global_video_offset = 0
        self.flip_video = False
        self.datapath = "./"
        self.__dict__.update(vars(args))
        self.writer = None

        if os.path.exists("settings.yaml"):
            stream = open("settings.yaml", mode='r')
            self.settings = load(stream, Loader=Loader)
            stream.close()
            self.datapath = self.settings['datapath'].replace("<current_user>", getuser())
            print("Processing path: ", self.datapath)
            if 'raw_options' in self.settings:
                raw_options = self.settings['raw_options']
                if self.in_video in raw_options:
                    self.global_video_offset = raw_options[args.in_video]['global_offset']
                    self.flip_video = raw_options[args.in_video]['flip']

        self.cap = None
        self.reload_video()

        last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

        if self.end_with == -1:
            self.end_with = last_frame
        else:
            if self.end_with > last_frame:
                print(("Warning: specified end frame ({:d}) is beyond the last video frame" +
                       " ({:d}). Stopping after last frame.")
                      .format(self.end_with, last_frame))
                self.end_with = last_frame

        if self.out_video == "":
            self.out_video = args.in_video[:-4] + "_" + out_postfix + ".mp4"

        self.writer = cv2.VideoWriter(os.path.join(self.datapath, self.out_video),
                                      cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                                      self.cap.get(cv2.CAP_PROP_FPS),
                                      (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      True)
        self.writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, cpu_count())

        self.frame = None
        self.cur_frame_number = None

    def go_to_frame(self, i_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.global_video_offset + i_frame))

    def get_last_frame(self):
        if self.cap is None:
            return -1
        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.global_video_offset) - 1

    def reload_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(os.path.join(self.datapath, self.in_video))
        if not self.cap.isOpened():
            raise ValueError("Could not open video!")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.global_video_offset)

    def __del__(self):
        self.cap.release()
        if self.writer is not None:
            self.writer.release()

    def run(self, verbose=True):
        if verbose:
            print("Processing video...")
        start_time = time.time()
        total_frame_span = self.end_with - self.start_from
        frame_counter = 0

        if self.start_from > 0:
            self.go_to_frame(self.start_from)

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
                self.go_to_frame(i_frame)
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
