import os
import re
import shutil

from zipfile import ZipFile
import yaml

import numpy as np
import torch

from train import initialize_generator
from utils import generate_random_tensors, generate_binvox_file
from write_tiff import write_tiff

@torch.no_grad()
def get_mean_style(generator):
    mean_style = None

    mean_style = torch.randn(1, 512)

    return mean_style


@torch.no_grad()
def sample(generator, step, mean_style, args, n_sample=1):
    voxel = generator(
        torch.randn(n_sample, 512),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )

    voxel = voxel.view(n_sample, args['cube_len'], args['cube_len'], args['cube_len'])    
    return voxel

def generate():
    """
    Entrypoint for test
    Creates 6 binvox files using a specified generator
    """

    with open('/module/src/config.yaml') as f:
        args = yaml.load(f, Loader=yaml.Loader)
        
    test_no = 0
    test_path  = args['test_path']
    base_name = os.path.basename(test_path)
    model_name = os.path.splitext(base_name)[0]
    regex = re.compile(r'\d+')
    epoch_no = int(regex.findall(model_name)[0])
    network_name = os.path.split(test_path)[0]
    g_model = initialize_generator(args).eval()
    print()
    if not os.path.exists(f'{network_name}'):
        os.mkdir(f'{network_name}')

    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step = 4
     
    mean_style = get_mean_style(g_model) 
    print(network_name)
    voxels = sample(g_model, step, mean_style, args,  n_sample=8)
    #voxels = voxels.detach().cpu()
   
    for voxel in voxels:

        generate_binvox_file(voxel, f'{network_name}/{test_no}_{epoch_no}_epochs', args["cube_len"])
 
        test_no += 1
    
def run_module(input_path_dict, output_folder_path):
    generate()
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    count = 0
    with ZipFile(os.path.join(output_folder_path, 'output.zip'), 'w') as zip:
        for file in os.listdir('/module/src/saved_models_graindata'):
            if file.split('.')[-1] not in {'binvox'}:
                continue
            arr = np.load(os.path.join('/module/src/saved_models_graindata', file.split('.')[0] + '.npy'))
            write_tiff(arr, os.path.join('/module/src/saved_models_graindata', file.split('.')[0]+ '.ome.tiff'))
            zip.write(os.path.join('/module/src/saved_models_graindata', file), arcname=file)
            zip.write(os.path.join('/module/src/saved_models_graindata', file.split('.')[0]+'.ome.tiff'), arcname=file.split('.')[0]+'.ome.tiff')
            count += 1
        shutil.copy2('/module/src/saved_models_graindata/0_12_epochs.ome.tiff', os.path.join(output_folder_path, '0_12_epochs.ome.tiff'))

    output =  {'Output Zip': os.path.join(output_folder_path, 'output.zip'),
            'Output Tiff': os.path.join(output_folder_path, '0_12_epochs.ome.tiff')}
    return output

if __name__ == '__main__':
    run_module('', '/outputs')
