#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    build_correspondences,
    build_correspondences_corners_cloud,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)


def triangulate_from_2_pos(cor1: FrameCorners, cor2: FrameCorners, mat1, mat2, intrinsic_mat, param, pos1, pos2,
                           point_cloud_builder: PointCloudBuilder):
    correspondences = build_correspondences(cor1, cor2)
    points3d, ids, cos = triangulate_correspondences(correspondences, mat1, mat2,
                                                     intrinsic_mat, param)
    point_cloud_builder.add_points(ids, points3d)
    print("Triangulating cloud points for frames:", pos1, pos2)
    print("Current size of the cloud is:", point_cloud_builder.ids.shape[0], ", median cos is:", cos)


def count_view_mat(corners: FrameCorners, point_cloud_builder: PointCloudBuilder, intrinsic_mat, pos):
    correspondences_corners_cloud = build_correspondences_corners_cloud(corners,
                                                                        point_cloud_builder)
    # Solving PnP
    _, r_vec, t_vec, inliers = cv2.solvePnPRansac(correspondences_corners_cloud.points_2,
                                                  correspondences_corners_cloud.points_1, intrinsic_mat, None)
    r_vec, t_vec = cv2.solvePnPRefineLM(correspondences_corners_cloud.points_2[inliers],
                                        correspondences_corners_cloud.points_1[inliers], intrinsic_mat, None,
                                        r_vec, t_vec)
    if inliers is None:
        inliers = np.empty(1)
    print("Frame", pos, ": calculating camera pos --", inliers.shape[0], "inliers found")
    return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # Initial setup
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    left = min(known_view_1[0], known_view_2[0])
    right = max(known_view_1[0], known_view_2[0])
    point_cloud_builder = PointCloudBuilder()
    triangulate_from_2_pos(corner_storage[left], corner_storage[right],
                           view_mats[left], view_mats[right],
                           intrinsic_mat, TriangulationParameters(0.5, 0, 0),
                           left, right, point_cloud_builder)

    initial_length = right - left + 1
    switch = 1
    frames_done = 2
    delta = 9
    min_delta = 4

    # Filling the middle
    mid_right = right
    mid_left = left
    while frames_done < initial_length:
        # Triangulation setup
        done = False
        if switch == 1:
            # Right side
            switch = -1
            new_frame = max(mid_right - delta, mid_left)
            if new_frame - mid_left < min_delta:
                new_frame = mid_left
                done = True
            cur_left = new_frame
            cur_right = mid_right
            mid_right = new_frame
        else:
            # Left side
            switch = 1
            new_frame = min(mid_left + delta, mid_right)
            if mid_right - new_frame < min_delta:
                new_frame = mid_right
                done = True
            cur_left = mid_left
            cur_right = new_frame
            mid_left = new_frame

        if not done:
            view_mats[new_frame] = count_view_mat(corner_storage[new_frame], point_cloud_builder,
                                                  intrinsic_mat, new_frame)
            frames_done += 1

        # Adding 3d points
        triangulate_from_2_pos(corner_storage[cur_left], corner_storage[cur_right],
                               view_mats[cur_left], view_mats[cur_right],
                               intrinsic_mat, TriangulationParameters(0.5, 0, 0),
                               cur_left, cur_right, point_cloud_builder)

        # Counting view_mats in (cur_left, cur_right)
        for mid_frame in range(cur_left + 1, cur_right):
            view_mats[mid_frame] = count_view_mat(corner_storage[mid_frame], point_cloud_builder,
                                                  intrinsic_mat, mid_frame)
        frames_done += cur_right - cur_left - 1

    delta = 8

    while frames_done < frame_count:
        # Triangulation setup
        if switch == 1:
            # Right side
            switch = -1
            if right == frame_count - 1:
                continue
            new_frame = min(right + delta, frame_count - 1)
            if frame_count - 1 - new_frame < min_delta:
                new_frame = frame_count - 1
            cur_left = right
            cur_right = new_frame
            right = new_frame
        else:
            # Left side
            switch = 1
            if left == 0:
                continue
            new_frame = max(left - delta, 0)
            if new_frame - 0 < min_delta:
                new_frame = 0
            cur_left = new_frame
            cur_right = left
            left = new_frame
        view_mats[new_frame] = count_view_mat(corner_storage[new_frame], point_cloud_builder,
                                              intrinsic_mat, new_frame)
        frames_done += 1

        # Adding 3d points
        triangulate_from_2_pos(corner_storage[cur_left], corner_storage[cur_right],
                               view_mats[cur_left], view_mats[cur_right],
                               intrinsic_mat, TriangulationParameters(0.5, 0, 0),
                               cur_left, cur_right, point_cloud_builder)

        # Counting view_mats in (cur_left, cur_right)
        for mid_frame in range(cur_left + 1, cur_right):
            view_mats[mid_frame] = count_view_mat(corner_storage[mid_frame], point_cloud_builder,
                                                  intrinsic_mat, mid_frame)
        frames_done += cur_right - cur_left - 1

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
