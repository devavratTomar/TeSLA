# <img src="website/tesla.gif" width="40" height="40" style="vertical-align: bottom"/> <b>TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation</b>

This repository contains official [PyTorch](https://pytorch.org/) implementation for [CVPR 2023](https://cvpr2023.thecvf.com/) paper **TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation** by Devavrat Tomar, Guillaume Vray, Behzad Bozorgtabar, and Jean-Philippe Thiran.

### *Abstract*
Most recent test-time adaptation methods focus on only classification tasks, use specialized network architectures, destroy model calibration or rely on lightweight information from the source domain. To tackle these issues, this paper proposes a novel **Te**st-time **S**elf-**L**earning method with automatic **A**dversarial augmentation dubbed **TeSLA** for adapting a pre-trained source model to the unlabeled streaming test data. In contrast to conventional self-learning methods based on cross-entropy, we introduce a new test-time loss function through an implicitly tight connection with the mutual information and online knowledge distillation. Furthermore, we propose a learnable efficient adversarial augmentation module that further enhances online knowledge distillation by simulating high entropy augmented images. Our method achieves state-of-the-art classification and segmentation results on several benchmarks and types of domain shifts, particularly on challenging measurement shifts of medical images. TeSLA also benefits from several desirable properties compared to competing methods in terms of calibration, uncertainty metrics, insensitivity to model architectures, and source training strategies, all supported by extensive ablations.

### *Overview of TeSLA Framework*
<img src="website/tesla_overview.svg" style="background-color:white; border: solid white;">

(a) The student model <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/deb18c89b908abf80bef809cbdcbae2d.svg?invert_in_darkmode" align=middle width=14.252356799999989pt height=22.831056599999986pt/>  is adapted on the test images by minimizing the proposed test-time objective <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/a8c95121d37068acdbc35e9975f50c86.svg?invert_in_darkmode" align=middle width=22.31974139999999pt height=22.465723500000017pt/> . The high-quality soft-pseudo labels required by <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/a8c95121d37068acdbc35e9975f50c86.svg?invert_in_darkmode" align=middle width=22.31974139999999pt height=22.465723500000017pt/> are obtained from the exponentially weighted averaged teacher model <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/5c7704963fa9ece758ae7def4b308098.svg?invert_in_darkmode" align=middle width=13.01377934999999pt height=22.831056599999986pt/> and refined using the proposed Soft-Pseudo Label Refinement (PLR) on the corresponding test images. The soft-pseudo labels are further utilized for teacher-student knowledge distillation via <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/9ca5d7ed36b5da46a0cde6b76ae0a92a.svg?invert_in_darkmode" align=middle width=25.50469679999999pt height=22.465723500000017pt/> on the adversarially augmented views of the test images. (b) The adversarial augmentations are obtained by applying learned sub-policies sampled i.i.d from <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/865a2c771b7419b8742c1a4a04cc5584.svg?invert_in_darkmode" align=middle width=10.045686749999991pt height=22.648391699999998pt/> using the probability distribution <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> with their corresponding magnitudes selected from <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode" align=middle width=17.73973739999999pt height=22.465723500000017pt/>. The parameters <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode" align=middle width=17.73973739999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.83677559999999pt height=22.465723500000017pt/> of the augmentation module are updated by the *unbiased gradient estimator* of the loss <img src="https://rawgit.com/in	git@github.com:devavratTomar/TeSLA/main/svgs/10b6ebc26c060d3fcbcc764955f8476f.svg?invert_in_darkmode" align=middle width=35.03099654999999pt height=22.465723500000017pt/> computed on the augmented test images.

#### [[arxiv]](https://arxiv.org/abs/xxxxxx) [[Project]](https://behzadbozorgtabar.com/TeSLA.html)

## **Requirements**

## **Dataset Download Links**
### (1) CIFAR-10C, CIFAR-100C
### (2) ImageNet-C
### (3) VisDA-C
### (4) Kather
### (5) VisDA-S
### (6) MRI


## **Pre-trained Source Models**

## **Code for training Source Models from scratch**

## **Examples for adapting source models using TeSLA**

## Classification on CIFAR, ImageNet, VisDA, and Kather datasets

### (1) Common Image Corruptions: CIFAR-10C
### (2) Common Image Corruptions: CIFAR-100C
### (3) Common Image Corruptions: ImageNet-C
### (4) Synthetic to Real Adaptation: VisDA-C
### (5) Medical Measurement Shifts: Kather


## Segmentation  on VisDA-S and MRI datasets

### (1) GTA5 to CityScapes
### (2) Domain shifts of MRI

## **Licence**