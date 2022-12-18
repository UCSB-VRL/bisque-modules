import pathlib
import logging as log
# import nibabel as nib

from torch import mode
import nphsegmentation as nsg

def run_module(input_path_dict, output_folder_path):
    output_paths_dict = dict()
    
    log.info(f"{input_path_dict['Input Image']=}")
    input_path = input_path_dict['Input Image']
    # n = nib.load(input_path)

    output_paths_dict["Segmented Image"] = nsg.main(pathlib.Path(input_path), 
                                                    pathlib.Path(output_folder_path), 
                                                    # modelPath = pathlib.Path.cwd() / 'src' / 'model_backup/epoch49_ResNet2D3Class_2Layer2x2_mixed2_300.pt',
                                                    modelPath = pathlib.Path.cwd() / 'src' / 'model_backup' / 'epoch50_2Dresnet_skullstrip5Class.pt',
                                                    rdir = pathlib.Path("/module/src"))
    log.info("Finished computing result!")

    return output_paths_dict
