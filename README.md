# SAVSR (AAAI 2024)

This repository is the official PyTorch implementation of the following paper: 

> **SAVSR: Arbitrary-Scale Video Super-Resolution via A Learned Scale-Adaptive Architecture**
>
> Zekun Li<sup>1</sup>, Hongying Liu<sup>2</sup>, Fanhua Shang<sup>2</sup>, Yuanyuan Liu<sup>1</sup>, Liang Wan<sup>2</sup>, Wei Feng<sup>2</sup>
>
> <sup>1</sup> Xidian University
>
> <sup>2</sup> Tianjin University



## Brief Introduction of SAVSR

> **Abstract**: Deep learning-based video super-resolution (VSR) networks have gained significant performance improvements in recent years. However, existing VSR networks can only support a fixed integer scale super-resolution task, and when we want to perform VSR at multiple scales, we need to train several models. This implementation certainly increases the consumption of computational and storage resources, which limits the application scenarios of VSR techniques. In this paper, we propose a novel **Scale-adaptive Arbitrary-scale Video Super-Resolution network (SAVSR)**, which is the first work focusing on spatial VSR at arbitrary scales including both non-integer and asymmetric scales. We also present an omni-dimensional scale-attention convolution, which dynamically adapts according to the scale of the input to extract inter-frame features with stronger representational power. Moreover, the proposed spatio-temporal adaptive arbitrary-scale upsampling performs VSR tasks using both temporal features and scale information. And we design an iterative bi-directional architecture for implicit feature alignment. Experiments at various scales on the benchmark datasets show that the proposed SAVSR outperforms state-of-the-art (SOTA) methods at non-integer and asymmetric scales.



## Installation & Dataset

> - Python >= 3.8
> - PyTorch >= 1.10.1



```bash
cd SAVSR

# install mmcv
pip install -U openmim
mim install 'mmengine'
mim install mmcv

# install lbasicsr
pip install -r requirements.txt
python setup.py develop
```



### Dataset

- Training set: [Vimeo90K](http://toflow.csail.mit.edu/)
- Testing set: Vid4 + UDM10

Please refer to the video dataset preparation of [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Video-Super-Resolution). We encourage to use the lmdb datasets to speed up training.

**Note that** storing LR-HR video pairs for all scales requires a large amount of storage space, so we directly utilize GT for arbitrary-scale downsampling during training and testing.



## Training & Testing

1. Please download the dataset corresponding to the task and place them in the folder specified by the training option in folder `/options/train/SAVSR/`
2. Follow the instructions below to train or test our SAVSR.

**Please note:** "2" in the following instructions means two GPUs. Please modify it according to your configuration. You are also encouraged to modify the YAML file in  "options/train/SAVSR/" to set more training settings.

```bash
# train SAVSR
bash dist_train 2 options/train/SAVSR/train_ASVSR_Vimeo90K_asBI.yml

# test SAVSR
# Vid4 dataset
python -u lbasicsr/test.py -opt options/test/SAVSR/test_SAVSR_Vid4_asBI.yml
# DUM10 dataset
python -u lbasicsr/test.py -opt options/test/SAVSR/test_SAVSR_UDM10_asBI.yml
```



## Results

- Vid4 dataset

|       | x1.1 | x1.2 | x1.3 | x1.4 | x1.5 | x1.6 | x1.7 | x1.8 | x1.9 | x2   |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| SAVSR |      |      |      |      |      |      |      |      |      |      |
|       | x2.1 | x2.2 | x2.3 | x2.4 | x2.5 | x2.6 | x2.7 | x2.8 | x2.9 | x3   |
| SAVSR |      |      |      |      |      |      |      |      |      |      |
|       | x3.1 | x3.2 | x3.3 | x3.4 | x3.5 | x3.6 | x3.7 | x3.8 | x3.9 | x4   |
| SAVSR |      |      |      |      |      |      |      |      |      |      |



- UDM10 dataset

|       | x1.1 | x1.2 | x1.3 | x1.4 | x1.5 | x1.6 | x1.7 | x1.8 | x1.9 | x2   |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| SAVSR |      |      |      |      |      |      |      |      |      |      |
|       | x2.1 | x2.2 | x2.3 | x2.4 | x2.5 | x2.6 | x2.7 | x2.8 | x2.9 | x3   |
| SAVSR |      |      |      |      |      |      |      |      |      |      |
|       | x3.1 | x3.2 | x3.3 | x3.4 | x3.5 | x3.6 | x3.7 | x3.8 | x3.9 | x4   |
| SAVSR |      |      |      |      |      |      |      |      |      |      |



## Citations

You may want to cite:
```
……
```



## Acknowledgement

The codes are based on  [BasicSR](https://github.com/XPixelGroup/BasicSR), [OVSR](https://github.com/psychopa4/OVSR), [ArbSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR) and [ODConv](https://github.com/OSVAI/ODConv). Please also follow their licenses. Thanks for their awesome works.
