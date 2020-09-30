#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from scipy.interpolate import griddata

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli
from _corners import _to_int_tuple


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class Tracker:
    def __init__(self, frame_0, treshold=200):
        self.frame_0 = frame_0
        self.treshold = treshold
        self.corners, self.corner_sizes = self.find_corners(frame_0)
        self.ids = np.arange(self.corners.shape[0])
        self.track_length = np.ones((self.corners.shape[0]), dtype='int')
        self.max_length = 0

    def find_flow(self, frame_1):
        tracked, mask, mask1 = self.use_optflow(self.frame_0, frame_1,
                                                self.corners)
        new_corners, new_sizes = self.find_corners(
            frame_1, mask=self.get_corner_cover()
        )

        m = (mask1 == 1) & (mask == 1)

        self.frame_0 = frame_1
        self.corners = np.concatenate(
            (tracked[m], new_corners),
            axis=0
        )

        new_m = m.reshape((-1,))

        self.corner_sizes = np.concatenate(
            (self.corner_sizes[new_m], new_sizes),
            axis=0
        )

        last_id = self.ids[-1]
        new_ids = np.arange(last_id, last_id + new_corners.shape[0])

        self.ids = np.concatenate(
            (self.ids[new_m], new_ids),
            axis=0
        )

        self.track_length[new_m] += 1
        self.track_length = np.concatenate(
            (self.track_length[new_m], np.ones(new_corners.shape[0],)),
            axis=0
        )
        cur_max_lenght = self.track_length.max()
        if cur_max_lenght > self.max_length:
            self.max_length = cur_max_lenght

        '''
        Следующие строки кода позволяют фильтровать уголки по
        минимальным собственным значениям, оставляя только нужный
        квантиль.
        Но из-за них обработка видео 1024x576 вместо 2 минут заняла
        25.
        '''

        # quality_mask = self.filter_corners_mask(self.frame_0)
        # self.corners = self.corners[quality_mask]
        # self.corner_sizes = self.corner_sizes[quality_mask]
        # self.ids = self.ids[quality_mask]
        # self.track_length = self.track_length[quality_mask]

    def find_layer_corners(self, frame, max_corners=10000,
                           quality=0.01, dist=10, mask=None, blockSize=10):
        corners = cv2.goodFeaturesToTrack(
            frame, max_corners, quality, dist, mask=mask, blockSize=blockSize
        )
        if corners is None:
            return np.empty((0, 2), dtype=float)
        corners = corners[:, 0, :]
        return corners

    def find_corners(self, frame, steps=5, mask=None):
        layer = frame.copy()
        k = 1
        corners = np.empty((0, 2), dtype=float)
        sizes = np.empty((0, ), dtype=float)

        for _ in range(steps):
            layer_corners = self.find_layer_corners(layer, mask=mask)
            corners = np.concatenate((corners, layer_corners * k), axis=0)
            sizes = np.concatenate(
                (sizes, np.array([k*3] * layer_corners.shape[0])),
                axis=0
            )
            layer = cv2.pyrDown(layer)
            if mask is not None:
                mask = cv2.pyrDown(mask)
            k *= 2

        return corners, sizes

    def get_corner_cover(self):
        cover = np.full(self.frame_0.shape, 255, dtype='uint8')
        for corner, rad in zip(self.corners, self.corner_sizes):
            coord = _to_int_tuple(corner)
            cover = cv2.circle(cover, coord, int(rad), color=0, thickness=-1)
        return cover

    def use_optflow(self, frame_0, frame_1, points, eps=1e-2):
        new_points, mask, _ = cv2.calcOpticalFlowPyrLK(
            (frame_0*255).astype(np.uint8),
            (frame_1*255).astype(np.uint8),
            points.astype('float32').reshape((-1, 1, 2)),
            None,
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      10, 0.03)
        )

        _, mask1, _ = cv2.calcOpticalFlowPyrLK(
            (frame_1*255).astype(np.uint8),
            (frame_0*255).astype(np.uint8),
            new_points.astype('float32').reshape((-1, 1, 2)),
            None,
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      10, 0.03)
        )

        return new_points, mask, mask1

    def filter_corners_mask(self, frame):
        eignvals = cv2.cornerMinEigenVal(frame, 10)
        xx, yy = np.meshgrid(
            np.arange(frame.shape[1]),
            np.arange(frame.shape[0])
        )
        coords = np.array((xx.ravel(), yy.ravel())).T

        corn_eig = griddata(
            coords,
            eignvals.flatten(),
            self.corners,
            fill_value=0
        )
        mask = corn_eig > np.quantile(corn_eig, 0.5)
        return mask

    def get_corners(self):
        return FrameCorners(
            self.ids[:],
            self.corners[:],
            self.corner_sizes[:]
        )


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    tracker = Tracker(frame_sequence[0])  # tracks and store corners on frames
    builder.set_corners_at_frame(0, tracker.get_corners())
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        tracker.find_flow(image_1)  # find new corners and track them
        builder.set_corners_at_frame(frame, tracker.get_corners())
    print('\n', tracker.max_length)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
