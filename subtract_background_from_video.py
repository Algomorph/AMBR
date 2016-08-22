#!/usr/bin/python3

from video_processor import VideoProcessor
from multiprocessing import cpu_count
import cv2
import sys
import os.path
import time
from background_subtractor import BackgroundSubtractor, MaskLabel


class BaseVideoBackgroundSubtractor(VideoProcessor):
    def __init__(self, args, main_out_vid_name="", with_video_output=True):
        super().__init__(args, main_out_vid_name, with_video_output)
        args.datapath = self.datapath
        self.background_subtractor = BackgroundSubtractor(args)

    def initialize(self, verbose=True):
        start = self.sampling_interval_start_frame
        if start < 0:
            return
        end = self.sampling_interval_end_frame
        last_frame = self.get_last_frame()

        if end <= start:
            raise ValueError("Sampling interval end frame (currently set to {:d}) should be greater than the " +
                             "sampling interval start frame (currently set to {:d}).".format(end, start))

        if start > last_frame or end > last_frame:
            raise ValueError("The sampling interval start & end frame (currently set to {:d} and {:d}, " +
                             "respectively) should be within [0,{:d}] as dictated by length of video {:s} " +
                             "(and global offset, if present)."
                             .format(start, end, last_frame, self.in_video))

        max_sampling_duration_frames = int(self.sampling_interval * (self.num_samples - 1) / 1000 * self.fps) + 1

        max_end = start + max_sampling_duration_frames - 1
        if end > max_end:
            print(("Notice: sampling_interval_end_frame is set to {0:d}, which is beyond the limit imposed by " +
                   "sampling interval ({1:f}), fps {2:.2f}, and number of samples ({3:d}). " +
                   "Changing it to {4:d} to save time.")
                  .format(start, self.sampling_interval, self.fps, self.num_samples, max_end))
            end = max_end

        if verbose:
            print("Initializing from frame {:d} to frame {:d}...".format(start, end))

        self.go_to_frame(start)
        start_time = time.time()
        total_frames = end - start + 1
        fc = 1
        for i_frame in range(start, end + 1):
            frame = self.cap.read()[1]
            # apply preliminary mask if at all present
            # build up the background model
            self.background_subtractor.pretrain(frame)
            if not self.no_progress_bar:
                VideoProcessor.update_progress(fc / total_frames, start_time)
            fc += 1
        sys.stdout.write("\n")  # terminate progress bar

        self.reload_video()


class VideoBackgroundSubtractor(BaseVideoBackgroundSubtractor):
    @staticmethod
    def make_parser(help_string):
        parser = VideoProcessor.make_parser(help_string)
        parser.add_argument("-mo", "--mask_output_video", default="")
        BackgroundSubtractor.prep_parser(parser)
        return parser

    def __init__(self, args, main_out_vid_name="foreground"):
        self.mask_writer = None
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
        self.foreground = None
        self.mask = None

    def __del__(self):
        super().__del__()
        if self.mask_writer is not None:
            self.mask_writer.release()

    def extract_foreground(self):
        foreground = self.frame.copy()
        foreground[self.mask < MaskLabel.PERSISTENCE_LABEL.value] = (0, 0, 0)
        return foreground

    def process_frame(self):
        self.mask = self.background_subtractor.extract_foreground_mask(self.frame)
        self.foreground = self.extract_foreground()
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
