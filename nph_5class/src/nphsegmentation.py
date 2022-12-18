import logging as log
import os
import argparse
import pathlib
import subprocess
import TestFunc as tf
import CSFseg as csf
from os.path import exists
import CTtools
from subprocess import call
import sys
from postSkullStrip import postSkullStrip
import nibabel as nib
import numpy as np


def skull_strip(inName, outName):
    outName =  outName.parent / (outName.name + "_Mask.nii.gz")
    print(outName)
    iname = "intermediate_" + outName.name
    ipath = outName.parent / iname
    print(f"{ipath=}")
    
    CTtools.bone_extracted(inName, ipath)

    stripped = postSkullStrip(inName, ipath)
    nii_image = nib.Nifti1Image(stripped.astype(np.float32), affine=np.eye(4))
    nib.save(nii_image, outName) # the corrected raw scans, should have a good number of slices bounded to just the brain + maybe some thin shape of the skull
    # call(['flirt', '-in', ct_scan_wodevice_bone, '-ref', MNI_152_bone, '-omat', nameOfAffineMatrix, '-bins', '256',
    #       '-searchrx', '-180', '180', '-searchry', '-180', '180', '-searchrz', '-180', '180', '-dof', '12',
    #       '-interp', 'trilinear'])
    

def main(input_path, output_path, rdir, betPath=pathlib.Path('/module/src/skull-strip/'), gtPath='gt', device='cuda', BS=200, modelPath=None):
    log.info("f{input_path.stem=}")
        
    device = tf.checkDevice(device)
    model  = tf.loadModel(modelPath, device)
    
    output_path_dict = dict()
    
    input_name = input_path.name.split('.')[0]
    
    # skull_strip(input_path, betPath / input_name, running_dir="/home/cirrus/projects/vision/Bisque_Module_NPH/Modules/NPHSegmentation/src/")
    # skull_strip(input_path, betPath / input_name, running_dir=rdir)
    skull_strip(input_path, betPath / input_name)
    
    resultName = tf.runTest(input_name, output_path, input_path, betPath, device, BS, model) # Filename
    # maxArea, maxPos, finalimg = csf.segVent(fileName[i], outputPath, resultName)
    # maxArea, maxPos, finalimg, outputName = csf.segVent(input_name, output_path, resultName) # outputName is filename

    # output_path_dict["Segmented Image"] = os.path.join(output_path, outputName) 
    # result = os.path.join(output_path, outputName) 
    result = os.path.join(output_path, resultName) 
    

    return result


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', default='model_backup/epoch50_2Dresnet_skullstrip5Class.pt')
    parser.add_argument('--outputPath', default='reconstructed')
    parser.add_argument('--dataPath', default='data-split/Scans')
    parser.add_argument('--betPath', default='data-split/skull-strip')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=200)
    parser.add_argument('--gtPath', default = 'data-split/gt')
    parser.add_argument('--strip_script_path', default = '/module/src')

    args = parser.parse_args()
    
    
    main(input_path  = pathlib.Path(args.dataPath),
         modelPath   = pathlib.Path(args.modelPath),
         output_path = pathlib.Path(args.outputPath),
         betPath     = pathlib.Path(args.betPath),
         gtPath      = args.gtPath,
         device      = args.device,
         BS          = args.batch_size,
         rdir        = args.strip_script_path)
