import h5py
import numpy as np
from PIL import Image

#加载HDF5文件
def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["images"][()]

#写入HDF5文件
def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("images", data=arr, dtype=arr.dtype)


# 
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    
    all_stripe = []
    
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)

    return totimg


#图像可视化 
def visualize(data, save_path=None):
    assert (len(data.shape) == 3)
    
    img = None
    #改变图片的形状(height, width, 3)或(height, width)
    if data.shape[2] == 1:
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    # 像素(0, 255)
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))
    else:
        img = Image.fromarray((data*255).astype(np.uint8))
    
    if save_path != None:
        img.save(save_path)
    return img