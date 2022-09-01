import imghdr
import time
import os
import sys
import psutil
import pickle
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
import skvideo.io as skvid
from PIL import Image

# constants
NUM_CLASSES = 5
MODEL_NAME = 'resnet18'
FEATURE_EXTRACT = 0
THRESH = 0.5
TMP_PATH = './tmp'
FRAMES_HOME = os.path.join(TMP_PATH, 'frames')
FRAMES_LIST_HOME = os.path.join(TMP_PATH, 'frames_list.txt')
IDX_TO_SUBSTRATE = {0:'Boulder', 1:'Cobble', 2:'Mud', 3:'Rock', 4:'Sand'}
        

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def initialize_transforms(input_size):
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms


def prepare_single_image(arr, transforms):
    '''function to return output of single image'''

    h,w,c = arr.shape
    # crop out top and sides
    arr_7575 = arr[int(0.25*h):, int(0.125*w):int(0.875*w),:]
    img = Image.fromarray(arr_7575)
    tens = transforms(img)

    return tens


def predict(vid, k, model_weight, batch_size, num_workers, outfile):
    tick = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Cuda is available: {}'.format(torch.cuda.is_available()))
    # now create the model and transforms
    model, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device(device)))
    model.eval()
    transforms = initialize_transforms(input_size)
    
    #get video data and transform one frame to see how much memory to allocate
    metadata = skvid.ffprobe(vid)
    _h, _w = int(metadata['video']['@height']), int(metadata['video']['@width'])
    #import pdb; pdb.set_trace()
    _tens = prepare_single_image(np.ones((_h,_w,3), dtype=np.uint8), transforms)
    h, w = _tens.shape[1], _tens.shape[2]

    videogen = skvid.vreader(vid)
    counter = 0
    fnums = []
    global_counter = 0
    batch = torch.zeros([batch_size, 3, h, w], device='cpu')
    ids = {}
    for frame in videogen:
        if global_counter % 100 == 0:
            print('Running inference on {}th frame'.format(global_counter))
        if counter < batch_size:
            if global_counter % k == 0:
                batch[counter] = prepare_single_image(frame, transforms)
                counter += 1
                ids['frame_{}'.format(global_counter)] = []
                fnums.append(global_counter)
            global_counter += 1

        else:
            counter = 0
            batch = batch.to(device)
            outputs = model(batch)
            preds = torch.sigmoid(outputs)
            over_thresh = np.where(np.array(preds.detach().cpu()) > THRESH)
            for idx, fnum in enumerate(over_thresh[0]):
                ids['frame_{}'.format(fnums[len(fnums)-batch_size+fnum])].append(IDX_TO_SUBSTRATE[over_thresh[1][idx]])
    
    # check if images still in arr when done
    if counter > 0:
        last_batch = batch[:counter-1]
        last_batch = last_batch.to(device)
        outputs = model(last_batch)
        preds = torch.sigmoid(outputs)
        over_thresh = np.where(np.array(preds.detach().cpu()) > THRESH)
        for idx, fnum in enumerate(over_thresh[0]):
            ids['frame_{}'.format(fnums[len(fnums)-batch_size+fnum])].append(IDX_TO_SUBSTRATE[over_thresh[1][idx]])

    df = pd.DataFrame.from_dict(ids, orient='index')
    df.to_csv(outfile, header=False)
    tock = time.time()
    print('Time taken to run inference:{}s'.format(tock-tick))
    return

    
if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int)
    p.add_argument('--vid', type=str)
    p.add_argument('--k', type=int, help='run inference on every "kth" frame')
    p.add_argument('--num_workers', type=int)
    p.add_argument('--model_weight', type=str)
    p.add_argument('--outfile', type=str)
    args = p.parse_args()

    batch_size = args.batch_size
    vid = args.vid
    k = args.k
    num_workers = args.num_workers
    model_weight = args.model_weight
    outfile = args.outfile

    predict(vid, k, model_weight, batch_size, num_workers, outfile)
