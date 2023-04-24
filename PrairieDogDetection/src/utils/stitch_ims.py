import os

import numpy as np
import cv2

OFFSET = 384
IMSIZE = 512
OUT_SIZE = (6000, 4000)


def stitch_one_im(fname, im_dir, outfile):

    # assume that the name of the image we want to stitch is of the form:
    # FILENAME_{i}_{j}.* where i and j represent corresponding tile on the grid

    # initialize empty array
    arr = np.zeros(OUT_SIZE)

    for file in os.listdir(im_dir):
        if not file.startsWith(fname):
            continue

        i, j = fname.split(".")[0].split("_")[-1:-2]

        arr = cv2.imread(os.path.join(im_dir, file), mode="RGB")
