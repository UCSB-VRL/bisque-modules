import os
import shutil
import yaml

from utils.data_tiling import make_tiles
from utils.stitch_annos import gather_all_annos
from utils.draw_boxes import draw_boxes
from utils.gobjects import create_gobject_and_add_to_image

from detect import detect

from bqapi import BQSession



def run_module(bq: BQSession, input_path_dict:dict, output_folder_path:str, uri:str, params:dict):

    """ Runs the module

    Args:
        bq (BQSession): bisque user session
        input_path_dict (dict): dictionary of input paths
        output_folder_path (str): path to output folder
        uri (str): URI to bisque image ex: bisque.ece.ucsb.edu:/data_service/00-xxxxxxxxxxxx
        params (dict): dictionary of parameters

    Returns:
        dict: dictionary of output paths
    """

    # set up variables
    args = create_args()
    MODEL_CONFIG = args['MODEL_CONFIG']
    IM_DIR = args['IM_DIR']
    TILES_DIR = args['TILES_DIR']
    LABELS_DIR = args['LABELS_DIR']
    OUTFILE = args['OUTFILE']
    OUTIMAGE = args['OUTIMAGE']
    CLASSES = args['CLASSES']
    COLOR_MAP = args['COLOR_MAP']
    IOU_THRESH = args['IOU_THRESH']
    IM_SIZE = args['IM_SIZE']
    OFFSET = args['OFFSET']
    DETECT_DIR = args['DETECT_DIR']

    #clean up directories first
    cleanup(TILES_DIR, DETECT_DIR)
    
    if True: #params['Create New Annos'] == 'true':
        create_new_gobjs(bq, uri, input_path_dict, IM_DIR, TILES_DIR, IM_SIZE, OFFSET, MODEL_CONFIG, LABELS_DIR, OUTFILE, IOU_THRESH, OUTIMAGE, COLOR_MAP, CLASSES, params['new_gobject_name'])
    else:
        print('Skipping GObjects Creation')
        bq.update_mex('Skipping GObjects Creation')



    output_path = os.path.join(output_folder_path, OUTIMAGE)
    
    output_paths_dict = {}
    
    return output_paths_dict



def create_new_gobjs(bq, uri, input_path_dict, im_dir, tiles_dir, im_size, offset, model_config, labels_dir, outfile, iou_thresh, outimage, color_map, classes, gobj_name):
    
    input_fn = input_path_dict["Input Image"]
    fname = input_fn.split("/")[-1]
    shutil.copyfile(input_path_dict["Input Image"], os.path.join(im_dir, fname))

    print('Making Tiles')
    bq.update_mex('Tiling Large Image')
    make_tiles(im_dir, tiles_dir, im_size, offset)

    print('Running detections on tiles')
    bq.update_mex('Running Detections on Tiles')
    with open(model_config, "r") as f:
        model_args = yaml.load(f, Loader=yaml.FullLoader)
    detect(model_args)

    print('Gathering Tile Annotations')
    bq.update_mex('Gathering Tile Annotations')
    gather_all_annos(fname.split('.')[0], labels_dir, outfile, tiles_dir, offset, iou_thresh)

    print('Drawing Boxes')
    bq.update_mex('Drawing Boxes')
    draw_boxes(outfile, input_fn, outimage)

    BOXES = {}
    with open(outfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = line.split()
            if box[0] not in BOXES:
                BOXES[box[0]] = []
            BOXES[box[0]].append(box)

    print('Uploading GObjects to image')
    bq.update_mex('Trying to upload GObjects to image')
    try:
        create_gobject_and_add_to_image(bq, uri, BOXES, color_map, classes, gobj_name)
    except Exception as e:
        print(e)
        bq.update_mex('Failed to upload GObjects to image')
    
    return


def create_args(config = '/module/src/config.yaml'):
    with open(config, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args


def cleanup(*args):
    for path in args:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)




if __name__ == "__main__":
    sess = BQSession().init_local('admin', 'admin', bisque_root="localhost", create_mex=False)
    uri = '00-qmT73LFvHVc4JviNnAHJiE'
    run_module(sess, {"Input Image": "/module/src/test_no_gt.jpg"}, '/module/src', uri)
