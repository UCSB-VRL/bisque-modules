import argparse, os,sys, shutil
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt
from sequential_dataloader import SequentialDataset

import torch
from tqdm import tqdm
from source.model import DQLR


def saveImgs(img1,img2,savePath):
	
	utils.save_image(
                torch.cat([img1, img2], 0),
                savePath,
                nrow=2,
                normalize=True,
                range=(-1, 1),
            )

def saveImg(img,savePath):
	
	utils.save_image(img,
                savePath,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
def make_out_path_structure(output_path,jump=1, force=False):
    if not os.path.exists(output_path):
        print('Creating output folder structure...')
        os.makedirs(os.path.join(output_path,'predicted','{}'.format(int(1))))
        os.makedirs(os.path.join(output_path,'predicted','{}'.format(int(jump+1))))
        os.makedirs(os.path.join(output_path,'predicted','{}'.format(int(2*jump+1))))
        os.makedirs(os.path.join(output_path,'comparison','1_{}'.format(int(1))))
        os.makedirs(os.path.join(output_path,'comparison','1_{}'.format(int(jump+1))))
        os.makedirs(os.path.join(output_path,'comparison','1_{}'.format(int(2*jump+1))))
        print('Created output folder structure.')
    elif force==True:
        print('Deleting output directory forcibly.')
        shutil.rmtree(output_path)
        make_out_path_structure(output_path,jump, force)
    else:
        sys.exit('Output folder structure already exists. Please provide new out_path.')


def main(data_path, out_path, jump=1, pretrained_path='./checkpoints/model.pt', force=False):
    make_out_path_structure(out_path, jump, force)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    model = DQLR()   

    try:
        model.load_state_dict(torch.load(pretrained_path))
        device = torch.device("cuda:0")
        model = model.to(device)
        
        print('Loaded model on GPU.')
    except:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        device = torch.device("cpu")
        print('Loaded model on CPU.')

    print('Preparing custom data...')    
    data = SequentialDataset(data_path, clip_len=3, clip_jump=jump, preprocess=False, transform=transform)

    # loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
    print('Custom data prepared.')

    with torch.no_grad():
        model.eval()
        for t_idx in tqdm(range(len(data))):
            img, imgs = data.__getitem__(t_idx)        
            img, imgs = img.to(device), imgs.to(device)       
        
            out, _ = model(img.unsqueeze_(0))

        # Two Images side by side - url to the saved path.

            saveImg(out[0],'{0}/predicted/1/1_{1}.png'.format(out_path,str(t_idx).zfill(3)))
            saveImg(out[1],'{0}/predicted/{1}/{1}_{2}.png'.format(out_path,str(jump + 1),str(t_idx).zfill(3)))
            saveImg(out[2],'{0}/predicted/{1}/{1}_{2}.png'.format(out_path,str(2*jump + 1),str(t_idx).zfill(3)))
       
            saveImgs(imgs[0,:,:,:].unsqueeze_(0), out[0],'{0}/comparison/1_1/1_{1}_{2}.png'.format(out_path,'1', str(t_idx).zfill(3)))
            saveImgs(imgs[1,:,:,:].unsqueeze_(0), out[1],'{0}/comparison/1_{1}/1_{1}_{2}.png'.format(out_path,str(jump + 1),str(t_idx).zfill(3)))
            saveImgs(imgs[2,:,:,:].unsqueeze_(0), out[2],'{0}/comparison/1_{1}/1_{1}_{2}.png'.format(out_path,str(2*jump + 1),str(t_idx).zfill(3)))



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default='./data')
#     parser.add_argument('--output', type=str, default='./outputs/default')
#     parser.add_argument('--jump', type=int, default = 1)
#     parser.add_argument('--force', action='store_true')
#     parser.add_argument('--pretrained', type=str, default='./checkpoints/model.pt')
#     args = parser.parse_args()
#
#     print(args)
#     main(args.data, args.output, args.jump, args.pretrained, args.force)

    
       
    
