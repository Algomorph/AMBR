#!/usr/bin/python3

from video_processor import VideoProcessor
from multiprocessing import cpu_count
import cv2
import sys
import cve
import os.path
import time
from enum import Enum


class Label(Enum):
    FOREGROUND_LABEL = 255
    PERSISTENCE_LABEL = 180
    SHADOW_LABEL = 80
    BACKGROUND_LABEL = 0


class VideoBackgroundSubtractor(VideoProcessor):
    @staticmethod
    def make_parser(help_string):
        parser = VideoProcessor.make_parser(help_string)
        parser.add_argument("-mo", "--mask_output_video", default="")

        # ====================IMBS parameters========================== #
        parser.add_argument("--fps", type=float, default=60.0)

        parser.add_argument("--fg_threshold", type=int, default=15)
        parser.add_argument("--association_threshold", type=int, default=5)

        parser.add_argument("-sis", "--sampling_interval_start_frame", type=int, default=-1)
        parser.add_argument("-sie", "--sampling_interval_end_frame", type=int, default=-1)

        parser.add_argument("--sampling_interval", type=float, default=250.0,
                            help="Duration, in ms, of the sampling interval between frames"
                                 " used to build initial background model")
        parser.add_argument("--num_samples", type=int, default=30)

        parser.add_argument("--min_bin_height", type=int, default=2)

        # *** shadow ***
        parser.add_argument("--alpha", type=float, default=0.65,
                            help="Lower bound on the value ratio between image & " +
                                 "model for the pixel to be labeled shadow")
        parser.add_argument("--beta", type=float, default=1.15,
                            help="Upper bound on the value ratio between image & " +
                                 "model for the pixel to be labeled shadow")
        parser.add_argument("--tau_s", type=float, default=60.,
                            help="Upper bound on the saturation difference between image & model "
                                 "for the pixel to be labeled as shadow")
        parser.add_argument("--tau_h", type=float, default=40.,
                            help="Upper bound on the hue difference between image & model "
                                 "for the pixel to be labeled as shadow")
        # ***************

        parser.add_argument("--min_area", type=float, default=1.15,
                            help="")
        parser.add_argument("--persistence_period", type=float, default=10000.0,
                            help="Duration of the persistence period in ms")
        parser.add_argument("-m", "--use_morphological_filtering", help="Use morphological filtering (open, close) on "
                                                                        "the result.", default=False,
                            action='store_true')
        # ============================================================= #

        parser.add_argument("-mf", "--mask_file", default=None)
        return parser

    def __init__(self, args, main_out_vid_name="foreground"):
        super().__init__(args, main_out_vid_name)
        if args.mask_output_video == "":
            args.mask_output_video = args.in_video[:-4] + "_bs_mask.mp4"

        self.mask_writer = cv2.VideoWriter(os.path.join(self.datapath, args.mask_output_video),
                                           cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                                           self.cap.get(cv2.CAP_PROP_FPS),
                                           (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                           False)

        self.mask_writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, cpu_count())
        self.foreground_writer = self.writer
        self.subtractor = cve.BackgroundSubtractorIMBS(self.fps, self.fg_threshold, self.association_threshold,
                                                       self.sampling_interval, self.min_bin_height, self.num_samples,
                                                       self.alpha, self.beta, self.tau_s, self.tau_h,
                                                       self.min_area, self.persistence_period,
                                                       self.use_morphological_filtering, False)
        if self.mask_file is not None:
            self.mask_file = os.path.join(self.datapath, self.mask_file)
            if not os.path.isfile(self.mask_file):
                raise ValueError("Could not find preliminary mask file at {:s}.".format(self.mask_file))
            self.prelim_mask = cv2.imread(self.mask_file, cv2.IMREAD_COLOR)
        else:
            self.prelim_mask = None
        self.mask = None

    def __del__(self):
        super().__del__()
        self.mask_writer.release()

    def initialize(self, verbose=True):
        start = self.sampling_interval_start_frame
        if start < 0:
            return
        end = self.sampling_interval_end_frame
        last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        if end <= start:
            raise ValueError("Sampling interval end frame (currently set to {:d}) should be greater than the " +
                             "sampling interval start frame (currently set to {:d}).".format(end, start))

        if start > last_frame or end > last_frame:
            raise ValueError("The sampling interval start & end frame (currently set to {:d} and {:d}, " +
                             "respectively) should be within [0,{:d}] as dictated by length of video {:s}."
                             .format(start, end, last_frame, self.in_video))

        max_sampling_duration_frames = int(self.sampling_interval * (self.num_samples-1) / 1000 * self.fps) + 1

        max_end = start + max_sampling_duration_frames - 1
        if end > max_end:
            print(("Notice: sampling_interval_end_frame is set to {0:d}, which is beyond the limit imposed by " +
                   "sampling interval ({1:f}), fps {2:.2f}, and number of samples ({3:d}). " +
                   "Changing it to {4:d} to save time.")
                  .format(start, self.sampling_interval, self.fps, self.num_samples, max_end))
            end = max_end

        if verbose:
            print("Initializing from frame {:d} to frame {:d}...".format(start, end))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        start_time = time.time()
        total_frames = end - start + 1
        fc = 1
        for i_frame in range(start, end + 1):
            frame = self.cap.read()[1]
            # apply preliminary mask if at all present
            if self.prelim_mask is not None:
                frame = self.prelim_mask & frame
            # builds up the background model
            self.subtractor.apply(frame)
            if not self.no_progress_bar:
                VideoProcessor.update_progress(fc / total_frames, start_time)
            fc += 1
        sys.stdout.write("\n")  # terminate progress bar

        # re-open video
        self.cap.release()
        self.cap = cv2.VideoCapture(os.path.join(self.datapath, self.in_video))

    def extract_foreground_mask(self):
        if self.prelim_mask is not None:
            frame = self.prelim_mask & self.frame
        else:
            frame = self.frame
        mask = self.subtractor.apply(frame)
        self.mask = mask

    def extract_foreground(self):
        foreground = self.frame.copy()
        foreground[self.mask < Label.PERSISTENCE_LABEL.value] = (0, 0, 0)
        self.foreground = foreground

    def process_frame(self):
        self.extract_foreground_mask()
        self.extract_foreground()
        self.mask_writer.write(self.mask)
        self.foreground_writer.write(self.foreground)


def main():
    help_string = "Subtract background from every frame in the video " + \
                  "and write both masks and masked video as two output videos."
    parser = VideoBackgroundSubtractor.make_parser(help_string)
    args = parser.parse_args()
    app = VideoBackgroundSubtractor(args)
    app.initialize()
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
