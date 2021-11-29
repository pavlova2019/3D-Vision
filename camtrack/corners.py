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


def _to_int_tuple(point):
    return tuple(map(int, np.round(point)))


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
    mask = (np.ones(img.shape) * 255).astype(np.uint8)
    for p, sz in zip(corners.points.astype(int), corners.sizes):
        cv2.circle(mask, _to_int_tuple(p), int(sz * 2), 0, thickness=-1)
    return mask


def add_corners(cnt, image, feature_params, corners, max_id):
    mask = set_mask(corners, image)
    p2 = cv2.goodFeaturesToTrack(image, mask=mask, **feature_params)
    if p2 is not None:
        p2 = p2.reshape((-1, 2))
        p2 = p2[:cnt, :]
        corners = FrameCorners(
            np.vstack((corners.ids, np.arange(max_id, max_id + p2.shape[0]).reshape((-1, 1)))),
            np.vstack((corners.points, p2)),
            np.vstack((corners.sizes, np.full(p2.shape[0], feature_params['blockSize']).reshape((-1, 1))))
        )
        max_id += p2.shape[0]
    return corners, max_id


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = (frame_sequence[0] * 255.0).astype(np.uint8)
    h, w = image_0.shape
    max_corners = int((h * w) // 4000)
    print("\n", max_corners)
    feature_params = {'maxCorners': max_corners, 'qualityLevel': 0.2, 'minDistance': 16, 'blockSize': 8}
    lk_params = {'winSize': (15, 15), 'maxLevel': 2,
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)}
    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params).reshape((-1, 2))
    corners = FrameCorners(np.arange(p0.shape[0]), p0, np.full(p0.shape[0], feature_params['blockSize']))
    max_id = corners.ids.shape[0]
    corners, max_id = add_corners(max_corners - max_id, image_0, feature_params, corners, max_id)

    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = (image_1 * 255.0).astype(np.uint8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners.points, None, **lk_params,
                                             minEigThreshold=2.0 * 1e-3)
        stt = np.hstack((st, st))
        corners_alive = np.sum(st).astype(int)
        corners = FrameCorners(corners.ids[st == 1], p1[stt == 1].reshape((-1, 2)),
                               corners.sizes[st == 1])
        if corners_alive < max_corners:
            corners, max_id = add_corners(max_corners - corners_alive, image_1, feature_params, corners, max_id)
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
