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
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    calc_m,
    to_opencv_camera_mat4x4
)


MAX_REP_ERROR = 2.5


def get_new_points(cor1: FrameCorners, cor2: FrameCorners, mat1, mat2, intrinsic_mat, pos1, pos2,
                   point_cloud_builder: PointCloudBuilder):
    correspondences = build_correspondences(cor1, cor2)
    param = TriangulationParameters(MAX_REP_ERROR, 0.2, 0.2)
    points3d, ids, cos = triangulate_correspondences(correspondences, mat1, mat2, intrinsic_mat, param)
    point_cloud_builder.add_points(ids, points3d)
    print("Triangulating cloud points for frames:", pos1, pos2)
    print("Current size of the cloud is:", point_cloud_builder.ids.shape[0], ", median cos is:", cos)


def count_view_mat(corners: FrameCorners, point_cloud_builder: PointCloudBuilder, intrinsic_mat, proj_mat, pos):
    print("Frame", pos, ": calculating camera pos --", end="")
    correspondences_corners_cloud = build_correspondences_corners_cloud(corners,
                                                                        point_cloud_builder)
    # Solving PnP
    pnp_params = {"reprojectionError": MAX_REP_ERROR, "confidence": 0.99, "iterationsCount": 1000}
    res, r_vec, t_vec, inliers = cv2.solvePnPRansac(correspondences_corners_cloud.points_cloud,
                                                    correspondences_corners_cloud.points_corners,
                                                    intrinsic_mat, None, **pnp_params)
    if inliers is None:
        raise RuntimeError("Can't calculate view_mat")
    print(inliers.shape[0], "inliers found")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    r_vec, t_vec = cv2.solvePnPRefineLM(correspondences_corners_cloud.points_cloud[inliers],
                                        correspondences_corners_cloud.points_corners[inliers], intrinsic_mat, None,
                                        r_vec, t_vec, criteria)
    view_mat3x4 = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
    m_view_mat3x4 = calc_m(correspondences_corners_cloud.points_corners, correspondences_corners_cloud.points_cloud,
                           view_mat3x4, proj_mat)
    return m_view_mat3x4


def pose_to_mat(pose: Pose):
    return np.hstack((pose.r_mat, pose.t_vec))


def get_positions(cor1: FrameCorners, cor2: FrameCorners, intrinsic_mat):
    correspondences = build_correspondences(cor1, cor2)
    tr_params = TriangulationParameters(MAX_REP_ERROR, 0.2, 0.2)

    if correspondences.ids.shape[0] < 5:
        return eye3x4(), 0

    e_params = {"method": cv2.RANSAC, "prob": 0.99, "threshold": 1.0, "maxIters": 1000}
    mat, mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2, intrinsic_mat, **e_params)

    if not np.any(mask) or mat.shape != (3, 3):
        return eye3x4(), 0

    r_mat1, r_mat2, t_vec0 = cv2.decomposeEssentialMat(mat)
    r_mat_s = [r_mat1, r_mat2]
    t_vec_s = [t_vec0, -t_vec0]
    poses = [Pose(r_mat, t_vec) for r_mat in r_mat_s for t_vec in t_vec_s]
    new_points = [len(triangulate_correspondences(correspondences, eye3x4(),
                                                  pose_to_mat(pose), intrinsic_mat,
                                                  tr_params)[1]) for pose in poses]
    best_pose = np.argmax(new_points)

    h_params = {"method": cv2.RANSAC, "confidence": 0.99, "ransacReprojThreshold": 5.0, "maxIters": 1000}
    _, h_mask = cv2.findHomography(correspondences.points_1, correspondences.points_2, **h_params)

    _, ids, cos = triangulate_correspondences(correspondences, eye3x4(),
                                              pose_to_mat(poses[best_pose]), intrinsic_mat, tr_params)
    inliers_num = ids.shape[0]
    print("+++ ", np.sum(mask)/np.sum(h_mask), cos)
    print(new_points)
    return pose_to_mat(poses[best_pose]), inliers_num/(np.sum(h_mask)*cos)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    proj_mat = to_opencv_camera_mat4x4(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frame_count = len(corner_storage)

    delta = 8
    min_delta = 4

    if known_view_1 is None or known_view_2 is None:
        # raise NotImplementedError()
        best = []
        best_cnt = 0
        st = 5
        for now in range(0, frame_count, st):
            for nxt in range(now + delta + min_delta//2, frame_count, st):
                print("** ", now, nxt)

                mat, cnt = get_positions(corner_storage[now], corner_storage[nxt], intrinsic_mat)
                if cnt > best_cnt:
                    print(cnt)
                    best_cnt = cnt
                    best = [(now, view_mat3x4_to_pose(eye3x4())), (nxt, view_mat3x4_to_pose(mat))]
        known_view_1 = best[0]
        known_view_2 = best[1]

    print("--- ", known_view_1, "\n--- ", known_view_2)

    # Initial setup
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    left = min(known_view_1[0], known_view_2[0])
    right = max(known_view_1[0], known_view_2[0])
    point_cloud_builder = PointCloudBuilder()
    get_new_points(corner_storage[left], corner_storage[right],
                   view_mats[left], view_mats[right],
                   intrinsic_mat, left, right, point_cloud_builder)

    initial_length = right - left + 1
    switch = 1
    frames_done = 2

    # Filling the middle
    mid_right = right
    mid_left = left
    while frames_done < initial_length:
        # Triangulation setup
        done = False
        new_frame = max(mid_right - delta, mid_left)
        if new_frame - mid_left < min_delta:
            new_frame = mid_left
            done = True
        cur_left = new_frame
        cur_right = mid_right
        mid_right = new_frame

        if not done:
            view_mats[new_frame] = count_view_mat(corner_storage[new_frame], point_cloud_builder,
                                                  intrinsic_mat, proj_mat, new_frame)
            frames_done += 1

        # Adding 3d points
        get_new_points(corner_storage[cur_left], corner_storage[cur_right],
                       view_mats[cur_left], view_mats[cur_right],
                       intrinsic_mat, cur_left, cur_right, point_cloud_builder)

        # Counting view_mats in (cur_left, cur_right)
        for mid_frame in range(cur_left + 1, cur_right):
            view_mats[mid_frame] = count_view_mat(corner_storage[mid_frame], point_cloud_builder,
                                                  intrinsic_mat, proj_mat, mid_frame)
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
                                              intrinsic_mat, proj_mat, new_frame)
        frames_done += 1

        # Adding 3d points
        get_new_points(corner_storage[cur_left], corner_storage[cur_right],
                       view_mats[cur_left], view_mats[cur_right],
                       intrinsic_mat, cur_left, cur_right, point_cloud_builder)

        # Counting view_mats in (cur_left, cur_right)
        for mid_frame in range(cur_left + 1, cur_right):
            view_mats[mid_frame] = count_view_mat(corner_storage[mid_frame], point_cloud_builder,
                                                  intrinsic_mat, proj_mat, mid_frame)
        frames_done += cur_right - cur_left - 1

    '''for new_frame in range(frame_count):
        view_mats[new_frame] = count_view_mat(corner_storage[new_frame], point_cloud_builder,
                                              intrinsic_mat, proj_mat, new_frame)'''

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
