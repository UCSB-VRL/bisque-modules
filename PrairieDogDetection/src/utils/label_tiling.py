import os
from glob import glob
import shutil

label_folder_path = "/home/bowen/projects/prairieDog/data/labels/val/"
output_path = "/home/bowen/projects/prairieDog/data/labels/val_tiles/"
image_width = 6000
image_height = 4000
tile_size = 512
tile_size_x = 512
tile_size_y = 512
offset = int(tile_size * 0.75)
# offset = 128

label_files = glob(f"{label_folder_path}*")
# label_files  = ['/home/bowen/projects/prairieDog/all_labels/EnrNE_Day2_Run1__0101.txt']

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("output_path already exists, deleting it")
    shutil.rmtree(output_path)
    os.makedirs(output_path)


# load annotations to one dict
for one_label_file in label_files:
    one_label_file_basename = os.path.basename(one_label_file)
    with open(one_label_file) as f:
        # all annotations for one iamge
        lines = [line.rstrip() for line in f]
    i, j = 0, 0
    while j * offset < image_height:
        while i * offset < image_width:
            if image_width - i * offset < tile_size:
                tile_size_x = image_width - i * offset
            if image_height - j * offset < tile_size:
                tile_size_y = image_height - j * offset

            lines_tile = []
            left, down, right, up = (
                i * offset,
                j * offset + tile_size_y,
                i * offset + tile_size_x,
                j * offset,
            )
            # if one label in the patch
            for line in lines:
                line = line.split(" ")[1:]
                label, x, y, w, h = [float(x) for x in line]
                x_pixel = x * image_width
                y_pixel = y * image_height
                w_pixel = w * image_width
                h_pixel = h * image_height
                x_pixel_tile = x_pixel - left
                y_pixel_tile = y_pixel - up
                if (
                    x_pixel_tile < 0
                    or x_pixel_tile > tile_size
                    or y_pixel_tile < 0
                    or y_pixel_tile > tile_size
                ):
                    continue
                x_tile = x_pixel_tile / tile_size_x
                y_tile = y_pixel_tile / tile_size_y
                w_tile = w_pixel / tile_size_x
                h_tile = h_pixel / tile_size_y

                # option 1. include all labels
                lines_tile.append(f"{int(label)} {x_tile} {y_tile} {w_tile} {h_tile}\n")

                # option 2. remove boxes out of bound
                # if x_pixel_tile - w_pixel/2 > 0 and x_pixel_tile + w_pixel/2 < tile_size_x and y_pixel_tile - h_pixel/2 > 0 and y_pixel_tile + h_pixel/2 < tile_size_y:
                #     lines_tile.append(f'{int(label)} {x_tile} {y_tile} {w_tile} {h_tile}\n')
                # else:
                #     print('box out of bound')
                #     continue

            if lines_tile:
                tile_file_name = f"{one_label_file_basename[:-4]}_{j}_{i}.txt"
                tile_file_path = os.path.join(output_path, tile_file_name)
                with open(tile_file_path, "a") as f:
                    for one_line_tile in lines_tile:
                        f.write(one_line_tile)
            tile_size_x = 512
            tile_size_y = 512
            i += 1
        j += 1
        i = 0
