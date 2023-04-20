import numpy as np
import tifffile
import scipy.ndimage as ndi

def write_tiff(arr, ofnm):

    arr_trim = arr
    for i in range(3):
        inds = np.nonzero(arr_trim.sum(tuple({0,1,2}- {i})))[0]
        slices = [slice(0,None) for _ in range(3)]
        left = min(inds)
        right = max(inds)
        right = left + max(5, right-left)
        slices[i] = slice(left, right+1)
        arr_trim = arr_trim[tuple(slices)]

    arr_trim = ndi.zoom(arr_trim[:3,:3,:3], 32, order=0)

    #print(arr_trim.shape)

    arr_trim = np.pad(arr_trim, 16)
    print(arr_trim.shape)


    '''tifffile.imwrite(ofnm, 255*.astype('uint8'), bigtiff=True,
        metadata={'axes': 'ZYX', 'SignificantBits': 8})'''

    tifffile.imwrite(ofnm, 255*arr_trim.astype('uint8'),
        bigtiff=True, metadata={'axes': 'ZYX', 'SignificantBits': 8})


if __name__ == '__main__':
    arr = np.load('0_12_epochs.npy')
    write_tiff(arr, '/outputs/sample.ome.tiff')