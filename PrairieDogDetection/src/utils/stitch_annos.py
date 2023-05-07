import os

import numpy as np
import cv2
from torchvision.ops import nms
import torch

OFFSET = 384
OUT_SIZE = (6000, 4000)


def gather_all_annos(im_name, im_dir, outfile, tiles_dir, offset, iou_thresh=0.5):
    """ aggregate all annotations from tiles into one file

    Args:
        im_name (_type_): _description_
        im_dir (_type_): _description_
        outfile (_type_): _description_
    """
    total_boxes = {}
    total_confs = {}
    after_nms = {}

    for file in os.listdir(im_dir):
        if not file.startswith(im_name):
            print(file)
            continue

        # get the size of each tile as it is not always 512x512
        
        im_h, im_w, _ = cv2.imread(os.path.join(tiles_dir, file.split('.')[0]+'.jpg')).shape
        
        j, i = file.split(".")[0].split("_")[-2:]
        j, i = int(j), int(i)
        with open(os.path.join(im_dir, file), "r") as f:
            lines = f.readlines()

        for line in lines:
            
            box = line.split()
            box = convert_box_coords(box, im_h, im_w)
            box = small_coords_to_big(box, i, j, offset)

            if box[0] not in total_boxes:
                total_boxes[box[0]] = []
                total_confs[box[0]] = []
                after_nms[box[0]] = []

            total_confs[box[0]].append(round(float(box[1]), 3))
            total_boxes[box[0]].append(box[2:])

    
    for cls in total_boxes:
        after_nms[cls] = set(nms(torch.tensor(total_boxes[cls]), torch.tensor(total_confs[cls]), iou_thresh).numpy().tolist())


    with open(outfile, "w") as f:
        for idx in total_boxes:
            for i, box in enumerate(total_boxes[idx]):
                if i in after_nms[idx]:
                    #breakpoint()
                    f.write('{} '.format(int(idx)))
                    f.write('{0:.2f} '.format(float(total_confs[idx][i])))
                    for j in range(4):
                        f.write('{} '.format(int(box[j])))
                    f.write("\n")
        


def convert_box_coords(box: list, im_h: int, im_w: int) -> list:
    """ convert box coordinates from (x_center, y_center, w, h) to (x1, y1, x2, y2)

    Args:
        box (): bounding box in format [class, conf, x_center, y_center, w, h]
        im_h (_type_): image height
        im_w (_type_): image width

    Returns:
        list: output bounding box converted to (class, conf, x1, y1, x2, y2)
    """

    for num in box[2:]:
        assert float(num) < 1

    if len(box) != 6:
        raise ValueError("box must be in format: [class, conf, x_center, y_center, w, h]")

    # convert boxes to (x1,y1) (x2,y2) for smaller image
    x_center, y_center = float(box[1]) * im_w, float(box[2]) * im_h
    w, h = float(box[3]) * im_w, float(box[4]) * im_h

    x1, y1 = round(x_center - w / 2), round(y_center - h / 2)
    x2, y2 = round(x_center + w / 2), round(y_center + h / 2)

    return [box[0], box[5], x1, y1, x2, y2]


def small_coords_to_big(box, i, j, offset):
    """ convert box coordinates from (x1, y1, x2, y2) for smaller image to (x1, y1, x2, y2) for big image

    Args:
        box (_type_): _description_
        i (_type_): _description_
        j (_type_): _description_
        offset (_type_): _description_

    Returns:
        _type_: _description_
    """

    if len(box) != 6:
        raise ValueError("box must be in format: [class, conf, x_center, y_center, w, h]")

    x1, y1 = float(box[2]) + i * offset, float(box[3]) + j * offset
    x2, y2 = float(box[4]) + i * offset, float(box[5]) + j * offset

    return [box[0], box[1], x1, y1, x2, y2]


if __name__ == "__main__":
    im_name = "test_no_gt"
    im_dir = "/home/connor/hdd5/p_dawgs/runs/detect/exp/labels"
    outfile = "/home/connor/hdd5/p_dawgs/out.txt"
    gather_all_annos(im_name, im_dir, outfile)
