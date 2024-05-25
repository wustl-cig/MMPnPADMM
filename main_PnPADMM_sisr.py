import json
import os.path
import glob
import cv2
import logging
import time
import random

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import OrderedDict
import hdf5storage
from tqdm import tqdm
import torch
import scipy.io as sio
from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util
from scipy.optimize import fminbound


"""
How to run:
step 1: set the priors from model_zoo, | mismatched | updated 
step 2: set the test set | MetFaces1_a | MetFaces1_b
step 3: 'python main_dpir_sisr.py'
"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    test_path = "./testsets/MetFaces1_a"                                                    # set the test set | MetFaces1_a | MetFaces1_b | MetFaces10

    model_path = "./model_zoo/BreCahad_Metfaces_models/mismatched/metfaces.pth"       # mimstaches ====>  metfaces.pth | brecahad.pth | afhq.pth | celeba.pth | rxrx1.pth
                                                                                      # updated    ====> img4.pth | img16.pth| img32.pth|  img64.pth

    noise_level_img = 0/255.0                               # set AWGN noise level for LR image, default: 0,
    noise_level_model = noise_level_img                     # set noise level of model, default 0
    test_sf = [4]                                           # set scale factor, default: [2, 4], [2], [4]
    iter_num = 15                                           # set number of iterations, default: 15 for SISR
    show_img = False                                        # default: False
    save_L = True                                           # save LR image
    save_E = True                                           # save estimated image
    single_result_save = False                              # fixed
    n_channels = 3                                          # fixed
    results = 'results'                                     # fixed
    mode = 'metfaces'                                       # set the test dataset ====> metfaces | rx
    retrieve_image = True                                   # set True if recon image, set False if recon table

    model_type = model_path.split("/")[-1].split(".")[0]

    test_size = 256                     # dimension of test image
    sigmarange = [0,600]
    result_folder = datetime.now().strftime("%Y%m%d-%H%M%S")#model_type + '_prior_' + str(test_sf[0]) + '_sf_' + mode + '_testtype'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    L_path = os.path.join(test_path) # L_path, for Low-quality images
    E_path = os.path.join(results, result_folder)   # E_path, for Estimated images
    util.mkdir(E_path)
    logger_name = result_folder
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)


    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_type, noise_level_img, noise_level_model))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']

    test_results_ave = OrderedDict()


    test_results_ave['psnr_sf_k'] , test_results_ave['ssim_sf_k']= [], []
    for sf in test_sf:
        border = sf
        k_num = 1 if retrieve_image  else 8
        test_results_ave['psnr_k'], test_results_ave['ssim_k'] = [], []

        for k_index in range(k_num):
            if retrieve_image:
                k_index = 3

            logger.info('--------- sf:{:>1d} --k:{:>2d} ---------'.format(sf, k_index))
            k = kernels[0, k_index].astype(np.float64)
            test_results_ave['psnr_imgs'], test_results_ave['ssim_imgs'] = [], []

            for idx, img in enumerate(L_paths):

                img_H = util.imread_uint(img, n_channels=n_channels)
                img_H = cv2.resize(img_H, (test_size, test_size))
                img_H = util.modcrop(img_H, sf)  # modcrop
                img_L = sr.classical_degradation(img_H, k, sf)
                util.imshow(img_L) if show_img else None
                img_L = util.uint2single(img_L)
                np.random.seed(seed=0)  # for reproducibility
                img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN
                img_name, ext = os.path.splitext(os.path.basename(img))


                alg = lambda modelSigma1: sr.fn(kernel= k, model = model , iter_num = iter_num, sf = sf, img_L = img_L, noise_level_model = noise_level_model, modelSigma1= modelSigma1, device = device)
                modelSigma1 = sr.optimizeTau(img_H, alg, sigmarange)



                img_E = sr.fn(kernel= k, model = model , iter_num = iter_num, sf = sf, img_L = img_L, noise_level_model = noise_level_model, modelSigma1= modelSigma1, device = device)


                psnr = util.calculate_psnr(img_E, img_H, border=border)
                ssim = util.calculate_ssim(img_E, img_H, border=border)
                test_results_ave['psnr_imgs'].append(psnr)
                test_results_ave['ssim_imgs'].append(ssim)

                if save_L:
                    util.imsave(util.single2uint(img_L).squeeze(), os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LR.png'))

                if single_result_save:
                    logger.info(
                        '{:->4d}--> {:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.2f}dB'.format(idx + 1,img_name + ext,sf, k_index,psnr, ))
                if save_E:
                    util.imsave(img_E, os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(
                        k_index) +'_PSNR_'+"{:.2f}".format(psnr)+'_SSIM_ '+"{:.4f}".format(ssim)+'.png'))

            ave_psnr = sum(test_results_ave['psnr_imgs']) / len(test_results_ave['psnr_imgs'])
            ave_ssim = sum(test_results_ave['ssim_imgs']) / len(test_results_ave['ssim_imgs'])
            logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) PSNR {:.2f} dB  SSIM:   {:.4f} '.format(mode, sf, k_index, ave_psnr, ave_ssim))

            test_results_ave['psnr_k'].append(ave_psnr)
            test_results_ave['ssim_k'].append(ave_ssim)

        ave_psnr_k = sum(test_results_ave['psnr_k']) / len(test_results_ave['psnr_k'])
        ave_ssim_k = sum(test_results_ave['ssim_k']) / len(test_results_ave['ssim_k'])
        test_results_ave['psnr_sf_k'].append(ave_psnr_k)
        test_results_ave['ssim_sf_k'].append(ave_ssim_k)

    ave_psnr_sf_k = sum(test_results_ave['psnr_sf_k']) / len(test_results_ave['psnr_sf_k'])
    ave_ssim_sf_k = sum(test_results_ave['ssim_sf_k']) / len(test_results_ave['ssim_sf_k'])
    logger.info('------> Average PSNR of ({}) {:.2f} dB, Average SSIM {:.4f}'.format(mode,ave_psnr_sf_k,ave_ssim_sf_k))
    with open(E_path+'/result.json', 'w') as f:
        f.write(json.dumps(test_results_ave))

if __name__ == '__main__':

    main()
