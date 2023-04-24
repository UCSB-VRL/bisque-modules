import os

import cv2

COLOR = (255, 1, 1)
THICKNESS = 2
LABEL_TO_COLOR = {0:(255, 1, 1), 1: (1, 255, 1), 2:(1, 1, 255) }

def draw_boxes(labels, image, out_image):
    image = cv2.imread(image)

    i = 0
    with open(labels, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = line.split()
            image = draw_one_box(box, image)

    cv2.imwrite(out_image, image)

def draw_one_box(box, img):
    start, end = (int(float(box[2])), int(float(box[3]))), (int(float(box[4])), int(float(box[5])))
    return cv2.rectangle(img, start, end, LABEL_TO_COLOR[int(box[0])], THICKNESS)


if __name__ == '__main__':
    labels = '/media/hdd5/connor/p_dawgs/out.txt'
    image = '/media/hdd5/connor/p_dawgs/test_no_gt.jpg'
    out_image = '/media/hdd5/connor/p_dawgs/out.jpg'

    draw_boxes(labels, image, out_image)