#!/usr/bin/python3
from video_processor import VideoProcessor
import cv2
import sys


class VideoMasker(VideoProcessor):
    def __init__(self, args):
        super().__init__(args, "masked")
        self.mask = cv2.imread(args.mask_file, cv2.IMREAD_COLOR)

    def process_frame(self):
        masked = self.mask & self.frame
        self.writer.write(masked)


def main():
    parser = VideoProcessor.make_parser("Apply mask to every frame of a video and save as a new video.")
    parser.add_argument("mask_file")
    args = parser.parse_args()
    app = VideoMasker(args)
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
