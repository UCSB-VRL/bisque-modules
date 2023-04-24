import os
import yaml

import torch
import torchvision.io as io

from inference import main, Loader

CONFIG_PATH = '/module/src/example.yaml'

def run_module(input_path_dict, output_folder_path):

    output_paths_dict = {}
    
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.load(f, Loader=Loader)

    noisy = io.read_image(input_path_dict['Noisy Image']).unsqueeze(0)/255
    
    out_image = main(noisy, cfg, cfg['experiment_cfg']) * 255
    out_image = out_image.type(torch.uint8).squeeze(0)
    io.write_png(out_image, os.path.join(output_folder_path, 'output_image.png'))
    output_paths_dict['Output Image'] = os.path.join(output_folder_path, 'output_image.png')
    return output_paths_dict


if __name__ == '__main__':

    run_module({'Noisy Image': 'sample_images/noisy.png'}, 'sample_images')
