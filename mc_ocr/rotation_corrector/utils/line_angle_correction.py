import os
import numpy as np
import cv2
from scipy.spatial import ConvexHull


def minimum_bounding_box(points):
    pi2 = np.pi / 2.
    hull_points = points[ConvexHull(points).vertices]
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    # rotations = np.vstack([
    #     np.cos(angles),
    #     -np.sin(angles),
    #     np.sin(angles),
    #     np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    rot_points = np.dot(rotations, hull_points.T)

    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def rotate_and_crop(img, points, debug=False, rotate=True, extend=True,
                    extend_x_ratio=1, extend_y_ratio=0.01,
                    min_extend_y=1, min_extend_x=2):
    rect = cv2.minAreaRect(points)

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if debug:
        print("shape of cnt: {}".format(points.shape))
        print("rect: {}".format(rect))
        print("bounding box: {}".format(box))
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    height = int(rect[1][0])
    width = int(rect[1][1])

    if extend:
        if width > height:
            w, h = width, height
        else:
            h, w = width, height
        ex = min_extend_x if (extend_x_ratio * w) < min_extend_x else (extend_x_ratio * w)
        ey = min_extend_y if (extend_y_ratio * h) < min_extend_y else (extend_y_ratio * h)
        ex = int(round(ex))
        ey = int(round(ey))
        if width < height:
            ex, ey = ey, ex
    else:
        ex, ey = 0, 0
    src_pts = box.astype("float32")
    # width = width + 10
    # height = height + 10
    dst_pts = np.array([
        [width - 1 + ex, height - 1 + ey],
        [ex, height - 1 + ey],
        [ex, ey],
        [width - 1 + ex, ey]
    ], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    # print(M)
    warped = cv2.warpPerspective(img, M, (width + 2 * ex, height + 2 * ey))
    h, w, c = warped.shape
    rotate_warped = warped
    if w < h and rotate:
        rotate_warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    if debug:
        print('ex, ey', ex, ey)
        cv2.imshow('before rotated', warped)
        cv2.imshow('rotated', rotate_warped)
        cv2.waitKey(0)
    return rotate_warped


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    os.environ["DISPLAY"] = ":12.0"
    # for n in range(10):
    #     points = np.random.rand(6, 2)
    #     # print(points)
    #     # plt.scatter(points[:, 0], points[:, 1])
    #     bbox = minimum_bounding_box(points)
    #     print(bbox)
    #     # plt.fill(bbox[:, 0], bbox[:, 1], alpha=0.2)
    #     # plt.axis('equal')
    #     # plt.show()
    input = '281,221,299,221,299,329,281,329'
    cnt = [int(f) for f in input.split(',')]
    cnt = np.array(cnt).astype(np.int32).reshape(-1, 1, 2)
    # cnt = np.array([[[469, 83]],
    #                 [[232, 81]],
    #                 [[232, 108]],
    #                 [[469, 109]]])
    path_im = '/home/cuongnd/PycharmProjects/aicr/PaddleOCR/inference_results/server_DB/viz_images_val/det_res_mcocr_val_145115bfzur.jpg'
    image = cv2.imread(path_im)
    rotate_and_crop(image, cnt, True, True)
