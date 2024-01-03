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

We retrained the model after cleaning the code, and the results on some scales may be higher than those in the paper.

### Vid4 dataset

- PSNR (dB) / SSIM at symmetric scale

|       |      x1.1      |      x1.2      |      x1.3      |      x1.4      |      x1.5      |      x1.6      |      x1.7      |      x1.8      |      x1.9      |       x2       |
| :---: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| SAVSR | 43.55 / 0.9952 | 41.31 / 0.9920 | 40.00 / 0.9890 | 39.12 / 0.9862 | 38.23 / 0.9829 | 37.32 / 0.9786 | 36.44 / 0.9734 | 35.63 / 0.9680 | 34.97 / 0.9627 | 34.67 / 0.9599 |
|       |    **x2.1**    |    **x2.2**    |    **x2.3**    |    **x2.4**    |    **x2.5**    |    **x2.6**    |    **x2.7**    |    **x2.8**    |    **x2.9**    |     **x3**     |
| SAVSR | 34.03 / 0.9553 | 33.57 / 0.9513 | 33.13 / 0.9470 | 32.71 / 0.9421 | 32.31 / 0.9369 | 31.89 / 0.9307 | 31.46 / 0.9241 | 31.07 / 0.9177 | 30.69 / 0.9097 | 30.33 / 0.9035 |
|       |    **x3.1**    |    **x3.2**    |    **x3.3**    |    **x3.4**    |    **x3.5**    |    **x3.6**    |    **x3.7**    |    **x3.8**    |    **x3.9**    |     **x4**     |
| SAVSR | 29.83 / 0.8945 | 29.32 / 0.8858 | 28.96 / 0.8765 | 28.77 / 0.8687 | 28.44 / 0.8607 | 28.09 / 0.8518 | 27.69 / 0.8422 | 27.51 / 0.8334 | 27.26 / 0.8235 | 27.17 / 0.8184 |

- PSNR (dB) / SSIM at asymmetr scale

|       |     x1.5/x4     |     x2/x4      |    x2/x3.75    |   x1.5/x3.5    |   x1.6/x3.05   | **x1.7/x3.75** |
| :---: | :-------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| SAVSR | 30.45 / 0.9027  | 29.99 / 0.8901 | 30.35 / 0.8982 | 31.61 / 0.9234 | 32.58 / 0.9376 | 30.65 / 0.9056 |
|       | **x2.95/x3.75** |  **x3.9/x2**   | **x3.5/x1.5**  |  **x3.5/x2**   | **x3.5/x1.75** |  **x4/x1.4**   |
| SAVSR | 29.14 / 0.8716  | 29.23 / 0.8901 | 30.50 / 0.9173 | 30.26 / 0.9101 | 30.37 / 0.9140 | 29.44 / 0.8959 |

### UDM10 dataset

- PSNR (dB) / SSIM at symmetric scale

|       |      x1.1      |      x1.2      |      x1.3      |      x1.4      |      x1.5      |      x1.6      |      x1.7      |      x1.8      |      x1.9      |       x2       |
| :---: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| SAVSR | 55.40 / 0.9992 | 53.55 / 0.9987 | 52.01 / 0.9982 | 50.81 / 0.9975 | 49.84 / 0.9969 | 48.90 / 0.9961 | 48.09 / 0.9953 | 47.37 / 0.9945 | 46.74 / 0.9937 | 46.23 / 0.9927 |
|       |    **x2.1**    |    **x2.2**    |    **x2.3**    |    **x2.4**    |    **x2.5**    |    **x2.6**    |    **x2.7**    |    **x2.8**    |    **x2.9**    |     **x3**     |
| SAVSR | 45.69 / 0.9917 | 45.17 / 0.9907 | 44.81 / 0.9896 | 44.42 / 0.9884 | 44.04 / 0.9872 | 43.56 / 0.9858 | 43.24 / 0.9844 | 42.83 / 0.9831 | 42.48 / 0.9816 | 42.15 / 0.9801 |
|       |    **x3.1**    |    **x3.2**    |    **x3.3**    |    **x3.4**    |    **x3.5**    |    **x3.6**    |    **x3.7**    |    **x3.8**    |    **x3.9**    |     **x4**     |
| SAVSR | 41.80 / 0.9785 | 41.38 / 0.9768 | 41.06 / 0.9750 | 40.76 / 0.9734 | 40.44 / 0.9715 | 40.10 / 0.9697 | 39.77 / 0.9677 | 39.41 / 0.9658 | 39.12 / 0.9638 | 38.88 / 0.9619 |

- PSNR (dB) / SSIM at asymmetr scale

|       |     x1.5/x4     |     x2/x4      |    x2/x3.75    |   x1.5/x3.5    |   x1.6/x3.05   | **x1.7/x3.75** |
| :---: | :-------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| SAVSR | 41.49 / 0.9821  | 41.34 / 0.9801 | 41.92 / 0.9823 | 42.94 / 0.9863 | 44.29 / 0.9892 | 42.09 / 0.9837 |
|       | **x2.95/x3.75** |  **x3.9/x2**   | **x3.5/x1.5**  |  **x3.5/x2**   | **x3.5/x1.75** |  **x4/x1.4**   |
| SAVSR | 40.76 / 0.9752  | 41.29 / 0.9750 | 42.54 / 0.9809 | 42.23 / 0.9798 | 42.41 / 0.9806 | 41.29 / 0.9748 |



## Citations

You may want to cite:
```
……
```



## Acknowledgement

The codes are based on  [BasicSR](https://github.com/XPixelGroup/BasicSR), [OVSR](https://github.com/psychopa4/OVSR), [ArbSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR) and [ODConv](https://github.com/OSVAI/ODConv). Please also follow their licenses. Thanks for their awesome works.
