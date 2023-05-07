import h5py
import numpy as np
import dream3d_import as d3
import os
import json

from PIL import Image

filename = 'Ti64_LR'
fdir = f'/home/devendra/Desktop/BisQue_Modules/EBSDSR_Module/EBSDSR/src'
fpath = f'{fdir}/output_data/{filename}.npy'

# npy to dream3d
def npy_2_dream3d(npy_arr):
    d3_sourceName = f'{fdir}/IPF_mapping/Ti64_DIC_Homo_and_Cubochoric_FZ.dream3d'
    d3_outputName = f'{fdir}/IPF_mapping/output.dream3d'

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

def change_var_in_json(Tupel):
    # change variable in JSON

    f = open('pipeline.json')

    field_dict = json.load(f)

    field_dict["0"]["InputFile"] = f'{fdir}/IPF_mapping/output.dream3d'
    field_dict["3"]["OutputFile"] = f'{fdir}/IPF_mapping/output.dream3d'


    field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][0]["Tuple Dimensions"]= Tupel
    field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][1]["Tuple Dimensions"]= Tupel


    outfile = open("pipeline.json", "w")
    json.dump(field_dict, outfile, indent=4)
    outfile.close()
        
def pipelinerunner():

    os.system("/home/devendra/Desktop/BisQue_Modules/EBSDSR_Module/EBSDSR/src/IPF_mapping/DREAM3D-6.5.141-Linux-x86_64/bin/PipelineRunner -p /home/devendra/Desktop/BisQue_Modules/EBSDSR_Module/EBSDSR/src/IPF_mapping/pipeline.json")


def dream3d_2_rgb():
    import pdb; pdb.set_trace()

    dream3d_file = h5py.File(f'{fdir}/IPF_mapping/output.dream3d')
    img = dream3d_file['DataContainers']['ImageDataContainer']['CellData']['IPFColor']
    total_img = img.shape[0]

    for i in range(total_img):
        image = Image.fromarray(img[i,:,:,:], "RGB")
        image.save(f'{fdir}/output_data/{filename}.png')



if __name__ == "__main__":
    import pdb; pdb.set_trace()
    loaded_npy = np.load(f'{fpath}')
    ydim, zdim, ch = loaded_npy.shape
    tupel = (ydim, zdim, 1)   
 
    npy_2_dream3d(loaded_npy)

    change_var_in_json(tupel)

    pipelinerunner()
    
    dream3d_2_rgb()



