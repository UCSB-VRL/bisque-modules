import os
import csv
import numpy as np
import nibabel as nib
import logging
import tifffile
import xmltodict

logging.basicConfig(filename='PythonScript.log', filemode='a', level=logging.DEBUG)
log = logging.getLogger('bq.modules')

def compute_metric_tifffile(file):

    # Read input image into numpy array
    fieldnames = ['Name',
                 'Total Brain Volume', 
                 'Center Ventricle Volume', 
                 'Center Brain Volume', 
                 'Center Ventricle to Brain Volume Ratio']

    input_tiff = tifffile.imread(file)

    with tifffile.TiffFile(file) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value

    final_img_res = np.zeros((1, 
                    input_tiff.shape[1],
                    input_tiff.shape[2],
                    input_tiff.shape[3],
                    ))

    for x in range(input_tiff.shape[0]):
        #final_img_res += input_tiff[x,:,:,:]*(x+1)
        final_img_res += ((input_tiff[x,:,:,:]!=0)*1)*(x+1)

    # final_img = final_img_res
    final_img_res = np.swapaxes(final_img_res, 3, 0)
    final_img_res = np.swapaxes(final_img_res, 2, 1)
    final_img_res = np.squeeze(final_img_res, axis=3)
    final_img = final_img_res

    #Set metadata
    xml_dict = xmltodict.parse(tif_tags['ImageDescription'])
    pixdim_x = float(xml_dict['OME']['Image']['Pixels']["@PhysicalSizeX"])
    pixdim_y = float(xml_dict['OME']['Image']['Pixels']["@PhysicalSizeY"])
    pixdim_z = float(xml_dict['OME']['Image']['Pixels']["@PhysicalSizeZ"])

    log.info("***** file: %s" % file)
    mm = abs(pixdim_x * pixdim_y * pixdim_z)
    logging.info(f"mm={mm}")
    
    final_N = final_img
    logging.info(f"final_N_shape={final_N.shape}")

    final_N_counts = ((final_N == 1) | (final_N == 6)).sum(axis=(0,1))
    logging.info(f"final_N_counts={final_N_counts}")

    max_z = np.argmax(final_N_counts) # slice with highest count
    logging.info(f"max_z={max_z}")

    numSlice = int((35 // pixdim_z) // 2)
    logging.info(f"numSlice={numSlice}")

    total_brain_count = (final_N != 1).sum() # everything but 1
    logging.info(f"total_brain_count={total_brain_count}")

    focused_final_N = final_N[...,max_z - numSlice : max_z + numSlice + 1]
    logging.info(f"focused_final_N={focused_final_N}")

    ventricle_count_7 = ((focused_final_N == 1) | (focused_final_N == 6)).sum() # class 1,6
    logging.info(f"ventricle_count_7={ventricle_count_7}")
    logging.info(f"ventricle counts = {((focused_final_N == 1) | (focused_final_N == 6)).sum(axis=(0,1))}")

    csf_count_7 = ventricle_count_7 + (focused_final_N == 3).sum() # classes 1,3,6
    logging.info(f"csf_count_7={csf_count_7}")

    brain_count_7 = (focused_final_N != 0).sum()
    logging.info(f"brain_count_7={brain_count_7}")

    name = file.name.split('-')[-1].split('.')[0]

    results = (name, 
               int(total_brain_count*mm),  
               int(ventricle_count_7*mm), 
               int(brain_count_7*mm), 
               round(ventricle_count_7/brain_count_7, 3))

    return dict(zip(fieldnames, results))




def compute_metric(file):

    fieldnames = ['Name',
                 'Total Brain Volume', 
                 'Center Ventricle Volume', 
                 'Center Brain Volume', 
                 'Center Ventricle to Brain Volume Ratio']

    try:
        final_img = nib.load(file)
    except:
        return compute_metric_tifffile(file)
    log.info("***** file: %s" % file)
    print(file); print("fileckck")
    mm = abs(final_img.header['pixdim'][1] * final_img.header['pixdim'][2] * final_img.header['pixdim'][3])
    logging.info(f"mm={mm}")

    final_N = final_img.get_fdata()
    logging.info(f"final_N_shape={final_N.shape}")

    final_N_counts = ((final_N == 1) | (final_N == 6)).sum(axis=(0,1))
    logging.info(f"final_N_counts={final_N_counts}")

    max_z = np.argmax(final_N_counts) # slice with highest count
    logging.info(f"max_z={max_z}")

    numSlice = int((35 // final_img.header['pixdim'][3]) // 2)
    logging.info(f"numSlice={numSlice}")

    total_brain_count = (final_N != 1).sum() # everything but 1
    logging.info(f"total_brain_count={total_brain_count}")

    focused_final_N = final_N[...,max_z - numSlice : max_z + numSlice + 1]
    logging.info(f"focused_final_N={focused_final_N}")

    ventricle_count_7 = ((focused_final_N == 1) | (focused_final_N == 6)).sum() # class 1,6
    logging.info(f"ventricle_count_7={ventricle_count_7}")
    logging.info(f"ventricle counts = {((focused_final_N == 1) | (focused_final_N == 6)).sum(axis=(0,1))}")

    csf_count_7 = ventricle_count_7 + (focused_final_N == 3).sum() # classes 1,3,6
    logging.info(f"csf_count_7={csf_count_7}")

    brain_count_7 = (focused_final_N != 0).sum()
    logging.info(f"brain_count_7={brain_count_7}")

    name = file.name.split('-')[-1].split('.')[0]

    results = (name, 
               int(total_brain_count*mm),  
               int(ventricle_count_7*mm), 
               int(brain_count_7*mm), 
               round(ventricle_count_7/brain_count_7, 3))

    return dict(zip(fieldnames, results))
