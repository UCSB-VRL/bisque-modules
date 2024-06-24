import os
import re
import shutil
import yaml

import h5py
import torch
import model
import numpy as np
import dream3d_import as d3
import json
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mat_sci_torch_quats.quats import fz_reduce, scalar_last2first, scalar_first2last
from mat_sci_torch_quats.symmetries import hcp_syms  


with open('/module/src/config.yaml') as f:
    args = yaml.load(f, Loader=yaml.Loader)

class Test:
    def __init__(self):
        self.scale = args['scale']        
        self.model = model.Model(args)
        self.save_path = args['save']
        self.dir = '/module/src'
        if not os.path.exists(f'{self.save_path}'):
            os.mkdir(f'{self.save_path}')
    
    def post_process(self, x):
        x = self.normalize(x)
        x = x.permute(0,2,3,1)

        # fz_reduction
        x = scalar_last2first(x)
        x = fz_reduce(x, hcp_syms)
        x = scalar_first2last(x)

        return x

    def normalize(self,x):
        x_norm = torch.norm(x, dim=1, keepdim=True)
                # make ||q|| = 1
        y_norm = torch.div(x, x_norm) 

        return y_norm
     
    def inference(self, lr):
        self.model.eval()
        with torch.no_grad():  
            sr = self.model(lr)
        sr = self.post_process(sr)
        sr = torch.squeeze(sr, dim=0) 
        sr = sr.cpu().numpy()
        np.save(f'{self.dir}/{self.save_path}/Ti64_SR.npy', sr)
    
        return sr

    # npy to dream3d
    def npy_2_dream3d(self, npy_arr, filetype):
        d3_sourceName = f'{self.dir}/reference.dream3d'
        d3_outputName = f'{self.dir}/{self.save_path}/output_{filetype}.dream3d'

        d3source = h5py.File(d3_sourceName, 'r')
        
        npy_arr = np.expand_dims(npy_arr, axis=0)

        xdim, ydim,zdim,channeldepth = np.shape(npy_arr)
        
        
        #Tupel = (xdim,ydim,zdim,channeldepth)  

        phases = np.int32(np.ones((xdim,ydim,zdim)))

        new_file = d3.create_dream3d_file(d3_sourceName, d3_outputName)

        in_path = 'DataContainers/ImageDataContainer' 
        out_path = 'DataContainers/ImageDataContainer'

        new_file = d3.copy_container(d3_sourceName, f'{in_path}/CellEnsembleData', d3_outputName, f'{out_path}/CellEnsembleData')

        new_file = d3.create_geometry_container_from_source(d3_sourceName, d3_outputName, dimensions=(xdim,ydim,zdim),
                                    source_internal_geometry_path=f'{in_path}/_SIMPL_GEOMETRY',
                                    output_internal_geometry_path=f'{out_path}/_SIMPL_GEOMETRY')

        new_file = d3.create_empty_container(d3_outputName, f'{out_path}/CellData', (xdim,ydim,zdim), 3)
        new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', npy_arr, 'Quats')
        new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', phases, 'Phases')

        # Close out source file to avoid weird memory errors.
        d3source.close()


    def change_var_in_json(self, Tupel, filetype):
        # change variable in JSON

        f = open(f'{self.dir}/pipeline.json')

        field_dict = json.load(f)

        field_dict["0"]["InputFile"] = f'{self.dir}/{self.save_path}/output_{filetype}.dream3d'
        field_dict["3"]["OutputFile"] = f'{self.dir}/{self.save_path}/output_{filetype}.dream3d'


        field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][0]["Tuple Dimensions"]= Tupel
        field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][1]["Tuple Dimensions"]= Tupel


        outfile = open(f'{self.dir}/pipeline.json', "w")
        json.dump(field_dict, outfile, indent=4)
        outfile.close()


    def pipelinerunner(self, filetype):

        os.system("/module/src/DREAM3D-6.5.141-Linux-x86_64/bin/PipelineRunner -p /module/src/pipeline.json")


    def dream3d_2_rgb(self, filetype):
        dream3d_file = h5py.File(f'{self.dir}/{self.save_path}/output_{filetype}.dream3d')
        img = dream3d_file['DataContainers']['ImageDataContainer']['CellData']['IPFColor']
        total_img = img.shape[0]

        for i in range(total_img):
            image = Image.fromarray(img[i,:,:,:], "RGB")
            image.save(f'{self.dir}/{self.save_path}/Ti64_{filetype}.png')


    def combine_ipf(self):
        img_sr = np.asarray(Image.open(f'{self.dir}/{self.save_path}/Ti64_SR.png'))
        img_lr = np.asarray(Image.open(f'{self.dir}/{self.save_path}/Ti64_LR.png'))
  
        fig, axes = plt.subplots(2,1, figsize=(12, 12), constrained_layout=True)
        axes[0].imshow(img_lr)
        axes[0].set_title('LR')        

        axes[1].imshow(img_sr)
        axes[1].set_title('SR')    
    
        plt.savefig(f'{self.dir}/{self.save_path}/Ti64_LR_SR.png')
        plt.close()
  

            
def quat_to_ipf(test, arr, ftype): 
    ydim, zdim, ch = arr.shape
    tupel = (ydim, zdim, 1)

    test.npy_2_dream3d(arr, ftype) 
    test.change_var_in_json(tupel, ftype)
    test.pipelinerunner(ftype)
    test.dream3d_2_rgb(ftype)  
 
 
def run_module(input_path_dict, output_folder_path):  

    if not os.path.exists(f'/outputs'):
        os.makedirs('/outputs')

    lr_data_path = input_path_dict['Input Image']
    ebsd_lr = np.load(f'{lr_data_path}')
    t = Test() 
    quat_to_ipf(t, ebsd_lr, 'LR')
 
    ebsd_lr_transpose = np.ascontiguousarray(ebsd_lr.transpose((2, 0, 1)))
    ebsd_lr = torch.from_numpy(ebsd_lr_transpose).float()
    ebsd_lr = torch.unsqueeze(ebsd_lr, dim=0)
                            
    ebsd_sr = t.inference(ebsd_lr)
 
    quat_to_ipf(t, ebsd_sr, 'SR')

    t.combine_ipf()
 
    #output_img_path_1 = f'{output_folder_path}/data/Ti64_SR.npy'
    #output_img_path_2 = f'{output_folder_path}/data/Ti64_LR.png'
    #output_img_path_3 = f'{output_folder_path}/data/Ti64_SR.png'

    output_img_path_1 = f'/module/src/data/Ti64_SR.npy'
    output_img_path_2 = f'/module/src/data/Ti64_LR_SR.png'
    output_img_path_3 = f'/module/src/data/Ti64_LR.png'
    output_img_path_4 = f'/module/src/data/Ti64_SR.png'



    #np.save(f'{output_img_path_1}', ebsd_sr) 
   
    shutil.copy2(f'{output_img_path_1}', os.path.join('/outputs', 'Ti64_SR.npy'))    
    shutil.copy2(f'{output_img_path_2}', os.path.join('/outputs', 'Ti64_LR_SR.png')) 
    shutil.copy2(f'{output_img_path_3}', os.path.join('/outputs', 'Ti64_LR.png')) 
    shutil.copy2(f'{output_img_path_4}', os.path.join('/outputs', 'Ti64_SR.png'))
 
    output_paths_dict = {}   
 
    output_paths_dict['Output Arr'] = os.path.join('/outputs', 'Ti64_SR.npy')
    output_paths_dict['Input Output IPF'] = os.path.join('/outputs', 'Ti64_LR_SR.png')
    output_paths_dict['Input IPF'] = os.path.join('/outputs', 'Ti64_LR.png') 
    output_paths_dict['Output IPF'] = os.path.join('/outputs', 'Ti64_SR.png') 

     
    return output_paths_dict 


if __name__ =='__main__':
    input_path_dict = {}
    curr_dir = os.getcwd()

    input_path_dict['Input Image'] = os.path.join(curr_dir, 'Ti64_LR.npy') 
    output_folder_path = curr_dir
   
    output_paths_dict = run_module(input_path_dict, output_folder_path)

    output_img_path_1 = output_paths_dict['Output Arr']
    output_img_path_2 = output_paths_dict['Input Output IPF']
    output_img_path_3 = output_paths_dict['Input IPF']
    output_img_path_4 = output_paths_dict['Output IPF']


