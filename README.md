# Prior Mismatch and Adaptation in PnP-ADMM with a Nonconvex Convergence Analysis 



Abstract
----------
Plug-and-Play (PnP) priors is a widely-used family of methods for solving imaging inverse problems by integrating physical measurement models with image priors specified using image denoisers.  PnP methods have been shown to achieve state-of-the-art performance when the prior is obtained using powerful deep denoisers. Despite extensive work on PnP, the topic of *distribution mismatch* between the training and testing data has often been overlooked in the PnP literature. This paper presents a set of new theoretical and numerical results on the topic of prior distribution mismatch and domain adaptation for the *{alternating direction method of multipliers (ADMM)* variant of PnP. Our theoretical result provides an explicit error bound for PnP-ADMM due to the mismatch between the desired denoiser and the one used for inference. Our analysis contributes to the work in the area by considering the mismatch under *nonconvex* data-fidelity terms and *expansive* denoisers. Our first set of numerical results quantifies the impact of the prior distribution mismatch on the performance of PnP-ADMM on the problem of image super-resolution. Our second set of numerical results considers a simple and effective domain adaption strategy that closes the performance gap due to the use of mismatched denoisers. Our results suggest the relative robustness of PnP-ADMM to prior distribution mismatch, while also showing that the performance gap can be significantly reduced with only a few training samples from the desired distribution.

----------
## How to Run the Code
----------
Download the pretrained models 

pip install gdown 

gdown --folder https://drive.google.com/drive/folders/1kHUPl6vFMmHZwSgEwi9Swz7QOHRF-mqz

-------- 
Select the model_path from the following options: 
 1. "./model_zoo/BreCahad_Metfaces_models/mismatch/*.pth"
 2. "./model_zoo/BreCahad_Metfaces_models/updated/*.pth"
 3. "./model_zoo/CelebA_RxRx1_models/mismatch/*.pth"
 4. "./model_zoo/CelebA_RxRx1_models/updated/*.pth"

--------
Set the reqiured variables (retrieve image or table, scaling factor, sample name). 

--------
python main_PnPADMM_sisr.py 

-------
The results will be saved in the "results" folder. 
