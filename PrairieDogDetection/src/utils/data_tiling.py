import os
import math
import shutil

import numpy as np
import cv2


class DataTiling:
    def __init__(self, size=512, offset=384):
        print("Init DataTiling class")
        self.tile_size = (size, size)
        self.offset = (offset, offset)

    def dataTiling(self, img, filename, dir_path, filetype=None):
        # create tiles directory
        # _dir_path = f'{dir_path}/{filename}_tiles'
        _dir_path = f"{dir_path}"
        # self.createDirectory(_dir_path)

        img_shape = img.shape
        for i in range(int(math.ceil(img_shape[0] / (self.offset[1] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1] / (self.offset[0] * 1.0)))):
                tile = img[
                    self.offset[1]
                    * i : min(self.offset[1] * i + self.tile_size[1], img_shape[0]),
                    self.offset[0]
                    * j : min(self.offset[0] * j + self.tile_size[0], img_shape[1]),
                ]

                if filetype == None:
                    tile_name = f"{filename}_{i}_{j}.npy"
                elif filetype == "png":
                    tile_name = f"{filename}_{i}_{j}.png"
                elif filetype == "JPG" or filetype == "jpg":
                    tile_name = f"{filename}_{i}_{j}.jpg"

                if tile_name not in _dir_path:
                    if filetype == None:
                        np.save(os.path.join(_dir_path, tile_name), tile)
                    else:
                        cv2.imwrite(os.path.join(_dir_path, tile_name), tile)

    def createDirectory(self, _dir_path):
        if not (os.path.isdir(_dir_path)):
            os.mkdir(_dir_path)
            print("\n Created", _dir_path)
        elif os.path.isdir(_dir_path):
            print("\n Already exist", _dir_path, "deleting it")
            shutil.rmtree(_dir_path)
            os.mkdir(_dir_path)
            print("\n Created", _dir_path)


def make_tiles(img_dir, img_tiles, size, offset):

    if not os.path.exists(img_tiles):
        os.makedirs(img_tiles)

    all_files = os.listdir(img_dir)
    all_file_paths = [os.path.join(img_dir, _name) for _name in all_files]

    # initializing DataTiling class
    DTObj = DataTiling(size, offset)

    def _callDataTiling(all_paths, dest_dir):
        for i in range(len(all_paths)):
            _img_path = all_paths[i]
            if _img_path.split(".")[-1] == "npy":
                img = np.load(_img_path)
            elif _img_path.split(".")[-1] == "JPG" or _img_path.split(".")[-1] == "jpg":
                img = cv2.imread(_img_path)
            else:
                continue
            filename = _img_path.split("/")[-1].split(".")[0]
            DTObj.dataTiling(img, filename, dest_dir, filetype="jpg")

    # create tiles
    _callDataTiling(all_file_paths, img_tiles)


if __name__ == "__main__":
    img_dir = "/module/src/data/images"
    img_tiles = "/module/src/data/image_tiles"
    make_tiles(img_dir, img_tiles)
