import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

filename = 'Ti64_SR'
fdir = f'/home/devendra/Desktop/BisQue_Modules/EBSDSR_Module/EBSDSR/src'
fpath = f'{fdir}/output_data/{filename}.npy'


def plot_quat(arr):
    channels = ['q1', 'q2', 'q3', 'q0'] 
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout= True)
    kwargs_imshow = {'vmin': -1, 'vmax': 1}

    for idx, ax, ch in zip(range(4), axes.reshape(-1), channels):
        quat_ch = arr[:,:, idx] 
        im = ax.imshow(quat_ch, **kwargs_imshow, cmap='jet')
        ax.set_title(ch, fontweight="bold")
        
    #cbar =fig.colorbar(im, ax = axes.ravel().tolist(), shrink=0.95)
    #cbar.set_ticklabels(np.arange(0,1,0.2))
    #cbar.set_ticklabels([-1 , 0, 1])

    plt.savefig(f'{fdir}/output_data/{filename}_quat.png')

    plt.close()      

if __name__ == "__main__":
    
    np_arr = np.load(f'{fpath}') 
    plot_quat(np_arr)
