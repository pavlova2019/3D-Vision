#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


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


def set_mask(corners: FrameCorners, img):
    mask = (np.ones(img.shape)*255).astype(np.uint8)
    for p, sz in zip(corners.points.astype(int), corners.sizes):
        cv2.circle(mask, (p[0], p[1]), sz[0], 0, -1)
    return mask


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.1,
                          minDistance=10,
                          blockSize=10)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    image_0 = (frame_sequence[0] * 255.0).astype(np.uint8)
    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params).reshape((-1, 2))
    corners = FrameCorners(np.arange(p0.shape[0]), p0, np.full(p0.shape[0], feature_params['blockSize']))
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = (image_1 * 255.0).astype(np.uint8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners.points, None, **lk_params,
                                             minEigThreshold=1.0*1e-3)
        stt = np.hstack((st, st))
        corners = FrameCorners(corners.ids[st == 1], p1[stt == 1].reshape((-1, 2)), corners.sizes[st == 1])
        if frame % 5:
            mask = set_mask(corners, image_1)
            p2 = cv2.goodFeaturesToTrack(image_1, mask=mask, **feature_params)
            if p2 is not None:
                p2 = p2.reshape((-1, 2))
                mx = np.max(corners.ids)
                corners = FrameCorners(
                    np.vstack((corners.ids, np.arange(mx, mx+p2.shape[0]).reshape((-1, 1)) + 1)),
                    np.vstack((corners.points, p2.reshape((-1, 2)))),
                    np.vstack((corners.sizes, np.full(p2.shape[0], feature_params['blockSize']).reshape((-1, 1))))
                )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


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
