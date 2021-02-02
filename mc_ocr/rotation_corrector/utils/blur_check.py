import numpy as np
import cv2
import os


def normalize_scale(image, new_long_edge, debug=True, ):
    h, w = image.shape[:2]
    # if w > h:
    #     scale = new_long_edge / w
    #     new_h = int(h * scale)
    #     image = cv2.resize(image, (new_long_edge, new_h))
    # else:
    scale = new_long_edge / h
    new_w = int(w * scale)
    image = cv2.resize(image, (new_w, new_long_edge))
    if debug:
        print('before', h, w)
        print('afterr', image.shape[:2])
    return image


def hsv_threshold_measure(image_, minValue=None, maxValue=None, show=False, threshSpotArea=200):
    if maxValue is None:
        maxValue = [360, 50, 255]
    if minValue is None:
        minValue = [0, 0, 245]
    ORANGE_MIN = np.array(minValue, np.uint8)
    ORANGE_MAX = np.array(maxValue, np.uint8)
    h_, w_, c_ = image_.shape
    crop_h = int(h_ * 0.1)
    crop_w = int(h_ * 0.1)
    y0 = crop_h
    y1 = h_ - crop_h
    x0 = crop_w
    x1 = w_ - crop_w
    crop_img = image_[y0:y1, x0:x1]
    hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
    contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    firstcheck = False
    max_cnt = 0
    for i in contours:
        cnt = cv2.contourArea(i)
        # print(cnt)
        if cnt > threshSpotArea:
            firstcheck = True
        if max_cnt < cnt:
            max_cnt = cnt

    if show:
        cv2.imshow("hsv_img frame_threshed", frame_threshed)

    ret = frame_threshed.var() if firstcheck else 0.0
    # print(ret)
    return ret


def balance_histogram(image):
    if len(image.shape) == 3:
        cols, rows, c = image.shape
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imshow('Equalized Image RGB', img_output)
    elif len(image.shape) == 2:
        cols, rows = image.shape
        img_output = cv2.equalizeHist(image)

        # cv2.imshow('Source image', gray_)
        cv2.imshow('Equalized Image', img_output)
    else:
        print('error image input dimension')
    return img_output


def change_brightness(image, visual=False, minimum_brightness=0.55):
    if len(image.shape) == 3:
        cols, rows, c = image.shape
        minimum_brightness *= c
    elif len(image.shape) == 2:
        cols, rows = image.shape
    else:
        print('error image input dimension')
    brightness = np.sum(image) / (255 * cols * rows)

    alpha = brightness / minimum_brightness
    bright_img = cv2.convertScaleAbs(image, alpha=1 / alpha, beta=0)
    if visual:
        print('brightness:', brightness)
        cv2.imshow("bright_img thresh", bright_img)
    return bright_img


def measure_brightness(gray_):
    cols, rows = gray_.shape
    brightness_ = np.sum(gray_) / (255 * cols * rows)
    return brightness_


def measure_demo(image=None, blurThreshold=150, spotThreshold=80, darkThreshold=.47, new_normalize_edge=1000,
                 visualize=False):
    if isinstance(image, str):
        image = cv2.imread(image)
    # image = normalize_scale(image, new_long_edge=new_normalize_edge)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('before', gray)
    # brightness = measure_brightness(gray)
    print('origin size: ', 'lap: ', cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    # table_gray = change_brightness(gray, minimum_brightness=0.55)
    # cv2.imshow('gray', gray)
    normalize_image = normalize_scale(image, new_long_edge=new_normalize_edge)
    normalize_gray = cv2.cvtColor(normalize_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(normalize_gray, cv2.CV_64F, ksize=3)

    print('normalized size: ', 'lap: ', lap.var())
    if visualize:
        cv2.imshow('after', normalize_image)
        cv2.waitKey()
    # hsv_bright_score = hsv_threshold_measure(image, show=visualize, threshSpotArea=70)
    #
    # blurry_ = False
    # spot_ = False
    # dark_ = False
    # blurry_Score = 0
    #
    # if brightness <= darkThreshold:
    #     dark_ = True
    # if brightness <= .5:
    #     # dark_ = True
    #     img_bh = balance_histogram(image)
    #
    #     lap_bright = cv2.Laplacian(img_bh, cv2.CV_64F)
    #     blurry_Score = lap_bright.var()
    #     if lap_bright.var() < blurThreshold:
    #         blurry_ = True
    #
    #     if visualize:
    #         print('lap_bright:', lap_bright.var())
    #         # cv2.imshow('Equalized Image RGB', img_bh)
    #         # cv2.imshow("lap_bright", lap_bright)
    # else:
    #
    #     blurry_Score = lap.var()
    #     if lap.var() < blurThreshold:
    #         blurry_ = True
    #     if visualize:
    #         print('lap_bright:', lap.var())
    #         # cv2.imshow('Equalized Image RGB', img_bh)
    #         # cv2.imshow("lap_bright", lap_bright)
    # if spotThreshold < hsv_bright_score:
    #     spot_ = True
    #
    # if visualize:
    #     # print('lap value:', blurry_Score)
    #     print('hsv_bright_score:', hsv_bright_score)
    #     print('brightness:', brightness)
    #     # cv2.imshow("lap", lap)
    #     text_blur = str(round(blurry_Score))
    #     if blurry_:
    #         text_blur += ' >>>> Blurry'
    #     cv2.putText(image, "lap value: {}  ".format(text_blur), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                 (0, 0, 255), 1)
    #     # if spot_:
    #     #     hsv_threshold_measure(image, show=False)
    #     #     cv2.putText(image, "{}: ".format('Bright Spot'), (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
    #     #                 1)
    #     # if dark_:
    #     #     cv2.putText(image, "{}: ".format('Too Dark'), (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
    #     #                 1)
    #     cv2.imshow("Image", image)
    #     print('spot_', spot_, 'dark_', dark_, 'blurry_', blurry_)
    #     cv2.waitKey(0)
    # print('Measure.Threshold: {}, Blurr score: {}'.format(blurThreshold, int(blurry_Score)))

    # return spot_, dark_, blurry_


def measure(image=None, blurThreshold=150, spotThreshold=80, darkThreshold=.47, new_normalize_edge=1000,
            visualize=False):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = normalize_scale(image, new_long_edge=new_normalize_edge)
    # hv = get_h_v(image, visual=visualize)
    # table_rgb = image.copy()
    # table_gray = cv2.cvtColor(table_rgb, cv2.COLOR_BGR2GRAY)
    # table_gray = normalize_scale(table_gray, new_long_edge=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = measure_brightness(gray)

    table_gray = change_brightness(gray, minimum_brightness=0.55)
    # cv2.imshow('gray', gray)
    if visualize:
        cv2.imshow('table_gray', table_gray)
    lap = cv2.Laplacian(table_gray, cv2.CV_64F, ksize=3)

    hsv_bright_score = hsv_threshold_measure(image, show=visualize, threshSpotArea=70)

    blurry_ = False
    spot_ = False
    dark_ = False
    blurry_Score = 0

    if brightness <= darkThreshold:
        dark_ = True
    if brightness <= .5:
        # dark_ = True
        img_bh = balance_histogram(image)

        lap_bright = cv2.Laplacian(img_bh, cv2.CV_64F)
        blurry_Score = lap_bright.var()
        if lap_bright.var() < blurThreshold:
            blurry_ = True

        if visualize:
            print('lap_bright:', lap_bright.var())
            # cv2.imshow('Equalized Image RGB', img_bh)
            # cv2.imshow("lap_bright", lap_bright)
    else:

        blurry_Score = lap.var()
        if lap.var() < blurThreshold:
            blurry_ = True
        if visualize:
            print('lap_bright:', lap.var())
            # cv2.imshow('Equalized Image RGB', img_bh)
            # cv2.imshow("lap_bright", lap_bright)
    if spotThreshold < hsv_bright_score:
        spot_ = True

    if visualize:
        # print('lap value:', blurry_Score)
        # print('hsv_bright_score:', hsv_bright_score)
        # print('brightness:', brightness)
        # cv2.imshow("lap", lap)
        text_blur = str(round(blurry_Score))
        if blurry_:
            text_blur += ' >>>> Blurry'
        cv2.putText(image, "lap value: {}  ".format(text_blur), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
        cv2.imshow("Image", image)
        print('spot_', spot_, 'dark_', dark_, 'blurry_', blurry_)
        cv2.waitKey(0)
    print('Measure.Threshold: {}, Blurr score: {}'.format(blurThreshold, int(blurry_Score)))

    return spot_, dark_, blurry_


def test_form3():
    img_path = '/data20.04/data/SEVT/SEVT_form3_1120_1/20201120_104840.jpg'
    img_path = ''
    if img_path == '':
        img_paths = get_img_paths(
            '/data20.04/data/SEVT/SEVT_form3_1125_errors',
            dropRoot=False)
    else:
        img_paths = [img_path]
    param_blur = 2000
    # READING ALL IMAGES
    for imagePath in img_paths:
        print('image path:', imagePath)
        spot_, dark_, blurry_ = measure(imagePath_=imagePath, visualize=True, blurThreshold=param_blur)
        print(spot_, dark_, blurry_)


if __name__ == '__main__':
    os.environ['DISPLAY'] = ':15.0'
    # val_fmlb()
    # val()
    test_form3()
