# <img src="website/tesla.gif" width="40" height="40" style="vertical-align: bottom"/> <b>TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation</b>

This repository contains official [PyTorch](https://pytorch.org/) implementation for [CVPR 2023](https://cvpr2023.thecvf.com/) paper **TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation** by Devavrat Tomar, Guillaume Vray, Behzad Bozorgtabar, and Jean-Philippe Thiran.

#### [[arxiv]](http://arxiv.org/abs/2303.09870) [[Project]](https://behzadbozorgtabar.com/TeSLA.html)

### *Abstract*
Most recent test-time adaptation methods focus on only classification tasks, use specialized network architectures, destroy model calibration or rely on lightweight information from the source domain. To tackle these issues, this paper proposes a novel **Te**st-time **S**elf-**L**earning method with automatic **A**dversarial augmentation dubbed **TeSLA** for adapting a pre-trained source model to the unlabeled streaming test data. In contrast to conventional self-learning methods based on cross-entropy, we introduce a new test-time loss function through an implicitly tight connection with the mutual information and online knowledge distillation. Furthermore, we propose a learnable efficient adversarial augmentation module that further enhances online knowledge distillation by simulating high entropy augmented images. Our method achieves state-of-the-art classification and segmentation results on several benchmarks and types of domain shifts, particularly on challenging measurement shifts of medical images. TeSLA also benefits from several desirable properties compared to competing methods in terms of calibration, uncertainty metrics, insensitivity to model architectures, and source training strategies, all supported by extensive ablations.

### *Overview of TeSLA Framework*
<img src="website/tesla_overview.svg">


(a) The student model <img src="website/svgs/deb18c89b908abf80bef809cbdcbae2d.svg#gh-light-mode-only" align=middle width=14.252356799999989pt height=22.831056599999986pt/><img src="website/svgs_dark/deb18c89b908abf80bef809cbdcbae2d.svg#gh-dark-mode-only" align=middle width=14.252356799999989pt height=22.831056599999986pt/> is adapted on the test images by minimizing the proposed test-time objective <img src="website/svgs/a8c95121d37068acdbc35e9975f50c86.svg#gh-light-mode-only" align=middle width=22.31974139999999pt height=22.465723500000017pt/><img src="website/svgs_dark/a8c95121d37068acdbc35e9975f50c86.svg#gh-dark-mode-only" align=middle width=22.31974139999999pt height=22.465723500000017pt/>  . The high-quality soft-pseudo labels required by <img src="website/svgs/a8c95121d37068acdbc35e9975f50c86.svg#gh-light-mode-only" align=middle width=22.31974139999999pt height=22.465723500000017pt/><img src="website/svgs_dark/a8c95121d37068acdbc35e9975f50c86.svg#gh-dark-mode-only" align=middle width=22.31974139999999pt height=22.465723500000017pt/> are obtained from the exponentially weighted averaged teacher model <img src="website/svgs/5c7704963fa9ece758ae7def4b308098.svg#gh-light-mode-only" align=middle width=13.01377934999999pt height=22.831056599999986pt/><img src="website/svgs_dark/5c7704963fa9ece758ae7def4b308098.svg#gh-dark-mode-only" align=middle width=13.01377934999999pt height=22.831056599999986pt/> and refined using the proposed Soft-Pseudo Label Refinement (PLR) on the corresponding test images. The soft-pseudo labels are further utilized for teacher-student knowledge distillation via <img src="website/svgs/9ca5d7ed36b5da46a0cde6b76ae0a92a.svg#gh-light-mode-only" align=middle width=25.50469679999999pt height=22.465723500000017pt/><img src="website/svgs_dark/9ca5d7ed36b5da46a0cde6b76ae0a92a.svg#gh-dark-mode-only" align=middle width=25.50469679999999pt height=22.465723500000017pt/> on the adversarially augmented views of the test images. (b) The adversarial augmentations are obtained by applying learned sub-policies sampled i.i.d from <img src="website/svgs/865a2c771b7419b8742c1a4a04cc5584.svg#gh-light-mode-only" align=middle width=10.045686749999991pt height=22.648391699999998pt/> <img src="website/svgs_dark/865a2c771b7419b8742c1a4a04cc5584.svg#gh-dark-mode-only" align=middle width=10.045686749999991pt height=22.648391699999998pt/> using the probability distribution <img src="website/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg#gh-light-mode-only" align=middle width=12.83677559999999pt height=22.465723500000017pt/><img src="website/svgs_dark/df5a289587a2f0247a5b97c1e8ac58ca.svg#gh-dark-mode-only" align=middle width=12.83677559999999pt height=22.465723500000017pt/>  with their corresponding magnitudes selected from <img src="website/svgs/fb97d38bcc19230b0acd442e17db879c.svg#gh-light-mode-only" align=middle width=17.73973739999999pt height=22.465723500000017pt/><img src="website/svgs_dark/fb97d38bcc19230b0acd442e17db879c.svg#gh-dark-mode-only" align=middle width=17.73973739999999pt height=22.465723500000017pt/>. The parameters <img src="website/svgs/fb97d38bcc19230b0acd442e17db879c.svg#gh-light-mode-only" align=middle width=17.73973739999999pt height=22.465723500000017pt/><img src="website/svgs_dark/fb97d38bcc19230b0acd442e17db879c.svg#gh-dark-mode-only" align=middle width=17.73973739999999pt height=22.465723500000017pt/> and <img src="website/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg#gh-light-mode-only" align=middle width=12.83677559999999pt height=22.465723500000017pt/><img src="website/svgs_dark/df5a289587a2f0247a5b97c1e8ac58ca.svg#gh-dark-mode-only" align=middle width=12.83677559999999pt height=22.465723500000017pt/> of the augmentation module are updated by the *unbiased gradient estimator* of the loss <img src="website/svgs/10b6ebc26c060d3fcbcc764955f8476f.svg#gh-light-mode-only" align=middle width=35.03099654999999pt height=22.465723500000017pt/><img src="website/svgs_dark/10b6ebc26c060d3fcbcc764955f8476f.svg#gh-dark-mode-only" align=middle width=35.03099654999999pt height=22.465723500000017pt/> computed on the augmented test images.


## **Requirements**

Fist install Anaconda (Python >= 3.8) using this [link](https://docs.anaconda.com/anaconda/install/index.html). Create the following CONDA environment by running the following command:
```
conda env create -f environment.yml
```
Activate the TeSLA environment as:
```
conda activate TeSLA
```

## **Datasets Download Links**
| Dataset Name      	| Download Link                                                                                      	| Extract to Relative Path               	|
|-------------------	|----------------------------------------------------------------------------------------------------	|----------------------------------------	|
| CIFAR-10C         	| [click here](https://zenodo.org/record/2535967 )                                                   	| ../Datasets/cifar_dataset/CIFAR-10-C/  	|
| CIFAR-100C        	| [click here](https://zenodo.org/record/3555552)                                                    	| ../Datasets/cifar_dataset/CIFAR-100-C/ 	|
| ImageNet-C        	| [click here](https://zenodo.org/record/2536630)                                                    	| ../Datasets/imagenet_dataset/          	|
| VisDA-C           	| [click here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) 	| ../Datasets/visda_dataset              	|
| Kather            	|                                                                                                    	|                                        	|
| VisDA-S           	| [click here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/segmentation)   	| ../Datasets/visda_dataset              	|
| (MRI) Spinal Cord 	| [click here](http://niftyweb.cs.ucl.ac.uk/program.php?p=CHALLENGE)                                 	| ../Datasets/MRI/SpinalCord             	|
| (MRI) Prostate    	| [click here](https://liuquande.github.io/SAML/)                                                    	| ../Datasets/MRI/Prostate               	|

## **Pre-trained Source Models Links**
### **Classification Task**
| Dataset Name 	| Download Link                                                                                         	| Extract to Relative Path       	|
|--------------	|-------------------------------------------------------------------------------------------------------	|--------------------------------	|
| CIFAR-10     	| [click here](https://drive.google.com/drive/folders/1bwf3qnaquRcfnoTfxKDwikVd_LnCitAm?usp=sharing)    	| ../Source_classifiers/cifar10  	|
| CIFAR-100    	| [click here](https://drive.google.com/drive/folders/1bnnkYORAwrjWI0jNhfVm_w0MvZH_DwJC?usp=share_link) 	| ../Source_classifiers/cifar100 	|
| ImageNet     	| PyTorch Default                                                               	|                                	|
| VisDA-C      	| [click here](https://drive.google.com/drive/folders/18PFWydp5nIA2lZ_zZ5FxskwlfHS_1eHV?usp=share_link) 	| ../Source_classifier/VisDA     	|
| Kather       	| [click here](https://drive.google.com/drive/folders/1uCDSqv-fgBsWNDZUUtshe-JG_0bs3wEQ?usp=share_link) 	| ../Source_classifier/Kather    	|

### **Segmentation Task**

| Dataset Name                   	| Download Link                                                                                         	| Extract to Relative Path                 	|
|--------------------------------	|-------------------------------------------------------------------------------------------------------	|------------------------------------------	|
| VisDA-S                        	| [click here](https://drive.google.com/drive/folders/1kxRHDKxB90PwqTcYUpMHNR1IG5DZBy8K?usp=share_link) 	| ../Source_classifier/Segmentation/VisDA/ 	|
| MRI (Spinal Cord and Prostate) 	| [click here](https://drive.google.com/drive/folders/1cV5Y2TRKUSJiUZqzFZCRqpQzxF__H1NF?usp=share_link) 	| ../Source_classifier/Segmentation/MRI/   	|

## **Code for training source models from scratch**

The above pre-trained source models can be obtained using the code available at: https://github.com/devavratTomar/tesla_appendix

## **Examples of adapting source models using TeSLA**
### Classification task on CIFAR, ImageNet, VisDA, and Kather datasets for online and offline adaptation:
**(1) Common Image Corruptions: CIFAR-10C**
```
bash scripts_classification/online/cifar10.sh
bash scripts_classification/offline/cifar10.sh
```

**(2) Common Image Corruptions: CIFAR-100C**
```
bash scripts_classification/online/cifar100.sh
bash scripts_classification/offline/cifar100.sh
```
**(3) Common Image Corruptions: ImageNet-C**
```
bash scripts_classification/online/imagenet.sh
bash scripts_classification/offline/imagenet.sh
```
**(4) Synthetic to Real Adaptation: VisDA-C**
```
bash scripts_classification/online/visdac.sh
bash scripts_classification/offline/visdac.sh
```
**(5) Medical Measurement Shifts: Kather**
```
bash scripts_classification/online/kather.sh
bash scripts_classification/offline/kather.sh
```
### Segmentation task on VisDA-S and MRI datasets for online and offline adaptation:
**(1) GTA5 to CityScapes**
```
bash scripts_segmentation/online/cityscapes.sh
bash scripts_segmentation/offline/cityscapes.sh
```
**(2) Domain shifts of MRI**
```
bash scripts_segmentation/online/spinalcord.sh
bash scripts_segmentation/offline/prostate.sh
```
## **Licence**
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.