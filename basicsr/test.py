import numpy as np
import os
import argparse
from tqdm import tqdm
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch
import torch.nn.functional as F
from basicsr.models.archs.SCNet_arch import SCNet, SCNetLocal
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import math
import cv2
import os
from metrics.psnr_ssim import calculate_ssim

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


yaml_file = ""
weights = ""
lq_dir = ""
gt_dir = ""
result_dir = ''
sigmas = []

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def load_img(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def save_img(filepath, img):
    cv2.imwrite(filepath,img)

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

checkpoint = torch.load(weights, map_location='cuda', weights_only=True)
model_restoration = SCNetLocal(**x['network_g'])
model_restoration.load_state_dict(checkpoint['params'])


print(f"Experiment: { yaml_file.split('/')[-1]}")
for sigma_test in sigmas:
    print("Compute results for noise level",sigma_test)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    PSNR_List = []
    SSIM_List = []

    files_lq = natsorted(glob(os.path.join(lq_dir, '*.png')))
    files_gt = natsorted(glob(os.path.join(gt_dir, '*.png')))

    with torch.no_grad():
        for file_lq, file_gt in tqdm(zip(files_lq, files_gt), total=len(files_lq)):
            torch.cuda.empty_cache()
            img = np.float32(load_img(file_lq))/255.
            img_copy = np.float32(load_img(file_gt))
            img_copy = np.expand_dims(img_copy, axis=2)

            np.random.seed(seed='')  # for reproducibility
            img += np.random.normal(0, sigma_test/255., img.shape)

            img = torch.from_numpy(img).unsqueeze(0)
            input_ = img.unsqueeze(0).cuda()

            h,w = input_.shape[2], input_.shape[3]
            restored = model_restoration(input_)

            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            noisy_save = torch.clamp(input_,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr = calculate_psnr(restored*255.0, img_copy)
            ssim = calculate_ssim(restored*255.0, img_copy, crop_border=0)

            PSNR_List.append(psnr)
            SSIM_List.append(ssim)

            save_file = os.path.join(result_dir, os.path.split(file_lq)[-1])
            save_img(save_file, img_as_ubyte(restored))

    print(f'PSNR = {np.mean(PSNR_List)} SSIM = {np.mean(SSIM_List)}')