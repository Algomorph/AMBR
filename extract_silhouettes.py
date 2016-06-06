#!/usr/bin/python3
from subtract_background_from_video import VideoBackgroundSubtractor, Label
import sys
import cv2


class SilhouetteExtractor(VideoBackgroundSubtractor):
    @staticmethod
    def make_parser(help_string):
        parser = VideoBackgroundSubtractor.make_parser(help_string)
        return parser

    def __init__(self, args):
        super().__init__(args, "silhouettes")

    def extract_foreground_mask(self):
        super().extract_foreground_mask()
        bmask = self.mask.copy()
        bmask[bmask < Label.PERSISTENCE_LABEL.value] = 0
        bmask[bmask > 0] = 1
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(bmask, ltype=cv2.CV_16U)
        self.stats = stats

    def extract_foreground(self):
        super().extract_foreground()
        foreground = self.foreground
        stats = self.stats
        for i_comp in range(1, len(stats)):
            x1 = stats[i_comp, cv2.CC_STAT_LEFT]
            x2 = x1 + stats[i_comp, cv2.CC_STAT_WIDTH]
            y1 = stats[i_comp, cv2.CC_STAT_TOP]
            y2 = y1 + stats[i_comp, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(foreground, (x1, y1), (x2, y2), color=(0, 0, 255))


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
