import argparse
import glob
import os

from math import floor
from typing import Union, Tuple

import cv2
import numpy as np
import torch
# import core
from os import path as osp

import torchvision
from numpy import mean
from torch import Tensor
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from lbasicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from lbasicsr.data.transforms import mod_crop, as_mod_crop
from lbasicsr.utils import img2tensor, scandir, tensor2img, imwrite
from lbasicsr.data.core import imresize


def read_img_seq(path, require_mod_crop=False, require_as_mod_crop=False, scale: Tuple = (4, 4), return_imgname=False):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        require_as_mod_crop (bool): Require arbitrary scale mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    if require_as_mod_crop:
        imgs = [as_mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder and {gt_key} folder should both in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_meta_info_file(folder, meta_info_file):
    """Generate paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        meta_info_file (str): Path to the meta information file.

    Returns:
        list[str]: Returned path list.
    """
    gt_folder = folder

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    # paths = []
    # for gt_name in gt_names:
    #     gt_path = osp.join(gt_folder, gt_name)
    #     paths.append(gt_path)
    paths = [osp.join(gt_folder, gt_name) for gt_name in gt_names]
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    # from scipy.ndimage import filters as filters
    from scipy.ndimage import gaussian_filter
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    # return filters.gaussian_filter(kernel, sigma)
    return gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x


def arbitrary_scale_downsample(x: Tensor, scale: Union[tuple, float], mode='torch', degradation='BI'):
    """Downsamping with arbitrary scale (use bicubic).

    :param x: (Tensor) Frames to be downsampled, with shape (b, t, c, h, w).
    :param scale: (float) Downsampling factor. Supported arbitrary scale.
    :param mode: bicubic implementation, core or torch
    :return:
        (Tensor) Arbitrary scale downsampled frames.
    """
    # if torch.cuda.is_available() and (not x.is_cuda):
    #     x = x.cuda()
    # print(x.device)
    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)

    # determine the type of scale
    if isinstance(scale, tuple):
        scale_h = scale[0]
        scale_w = scale[1]
    else:
        scale_h = scale
        scale_w = scale

    # step_h = cal_step(scale_h)
    # step_w = cal_step(scale_w)

    # crop borders
    b, t, c, h, w = x.size()
    # h = round(floor(h / step_h / scale_h) * step_h * scale_h)
    # w = round(floor(w / step_w / scale_w) * step_w * scale_w)
    # x = x[..., :h, :w]

    # bicubic downsampling
    if degradation == 'BI':
        x = x.view(-1, c, h, w)
        if mode == 'torch':
            x = T.Resize(size=(round(h / scale_h), round(w / scale_w)), interpolation=InterpolationMode.BICUBIC,
                         antialias=True)(x)
        elif mode == 'core':
            x = imresize(x, sizes=(round(h / scale_h), round(w / scale_w)))
        x = x.view(b, t, c, x.size(-2), x.size(-1))
    elif degradation == 'BD':
        x = duf_downsample(x, kernel_size=13, scale=scale_h)

    if squeeze_flag:
        x = x.squeeze(0)

    return x


def downsample_img():
    # ------ init -------------------------------------------------------------------------------------------
    is_bi_sr = True  # if generate bicubic sr results at the same time

    gt_path = '/data/lzk/workspace/LbasicSR/datasets/DIV2K/DIV2K_valid_HR'
    save_path_ = '/data/lzk/workspace/LbasicSR/datasets/DIV2K/DIV2K_valid_arbitrary_scale_BI'
    imgs_name_gt = sorted(glob.glob(osp.join(gt_path, '*')))

    # scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
    #           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]

    # scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    # scales = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
    scales = [6, 12, 18, 24, 30]
    # -------------------------------------------------------------------------------------------------------

    for cur_scale in scales:
        print("\ncurrent scale: {}".format(cur_scale))
        if isinstance(cur_scale, tuple):
            scale = cur_scale
            save_path = osp.join(save_path_, "x{}_x{}".format(scale[0], scale[1]))
        else:
            scale = (cur_scale, cur_scale)
            save_path = osp.join(save_path_, "x{}".format(scale[0]))

        for img_name_gt in imgs_name_gt:
            img_name = osp.basename(img_name_gt)
            print(img_name)
            img_gt = cv2.imread(img_name_gt).astype(np.float32) / 255.

            print("origin GT shape: {}".format(img_gt.shape))
            orig_H, orig_W = img_gt.shape[0], img_gt.shape[1]
            img_gt = as_mod_crop(img_gt, scale)
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)

            # # save crop gt -----------------------------------------------------------------------
            # save_crop_gt_path = osp.join(save_path, f'x{scale[0]}_x{scale[1]}_crop_gt', img_name)
            # result_img_crop_gt = tensor2img(img_gt)
            # imwrite(result_img_crop_gt, save_crop_gt_path)

            img_gt = img_gt.unsqueeze(0)
            if orig_H != img_gt.shape[-2] or orig_W != img_gt.shape[-1]:
                print("after as_mod_crop, GT shape: {}".format(img_gt.shape))

            b, c, h, w = img_gt.size()
            img_lr = arbitrary_scale_downsample(img_gt, scale, mode='torch')
            print("after downsample, LR shape: {}".format(img_lr.shape))

            # ------ get bicubic -----------------------------------------------------------------
            if is_bi_sr:
                img_sr = T.Resize(size=(h, w), interpolation=InterpolationMode.BICUBIC,
                                  antialias=True)(img_lr)
                print("bicubic SR shape: {}".format(img_sr.shape))

                save_bi_img_path = osp.join(save_path + "_bicubic_sr", img_name)
                result_bi_img_lr = tensor2img(img_sr)
                print(save_bi_img_path)
                imwrite(result_bi_img_lr, save_bi_img_path)
            # ------------------------------------------------------------------------------------

            save_img_path = osp.join(save_path, img_name)
            result_img_lr = tensor2img(img_lr)
            print(save_img_path)
            imwrite(result_img_lr, save_img_path)

            print('-' * 100)


def downsample_video(data_root: str, gt_dir: str = 'GT', degradation: str = 'BI'):
    # init
    # gt_root = '/data/lzk/workspace/LbasicSR/datasets/Vid4/GT'
    gt_root = osp.join(data_root, gt_dir)
    if degradation == 'BI':
        # save_path = '/data/lzk/workspace/LbasicSR/datasets/UDM10/arbitrary_scale_BI'
        save_path_ = osp.join(data_root, 'arbitrary_scale_BI')
    else:
        # save_path = '/data/lzk/workspace/LbasicSR/datasets/UDM10/arbitrary_scale_BD'
        save_path_ = osp.join(data_root, 'arbitrary_scale_BD')

    subfolers_gt = sorted(glob.glob(osp.join(gt_root, '*')))

    # scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
    #           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    scales = [(1.5, 4), (2, 4), (2, 3.75), (1.5, 3.5), (1.6, 3.05), (1.7, 3.75), 
              (2.95, 3.75), (3.9, 2), (3.5, 1.5), (3.5, 2), (3.5, 1.75), (4, 1.4)]

    if degradation == 'BD':
        scales = [2, 3, 4]

    for scale in scales:
        if isinstance(scale, tuple):
            scale = scale
            save_path = osp.join(save_path_, "x{}_x{}".format(scale[0], scale[1]))
        else:
            scale = (scale, scale)
            save_path = osp.join(save_path_, "x{}".format(scale[0]))

        for subfoler_gt in subfolers_gt:
            subfoler_name = osp.basename(subfoler_gt)
            print(subfoler_name)
            img_paths_gt = sorted(list(scandir(subfoler_gt, full_path=True)))

            max_id = len(img_paths_gt)
            print(max_id)
            # print(img_paths_gt)
            imgs_gt = read_img_seq(img_paths_gt, require_as_mod_crop=True,
                                   scale=scale)  # Tensor [41, 3, 576, 720], [0, 1] range
            print(imgs_gt.size())
            imgs_gt = imgs_gt.unsqueeze(0)
            print(imgs_gt.size())

            imgs_lr = arbitrary_scale_downsample(imgs_gt, scale, mode='torch', degradation=degradation)
            print(imgs_lr.size())

            i = 0
            for img_path_gt in img_paths_gt:
                img_name = osp.splitext(osp.basename(img_path_gt))[0]
                save_img_path = osp.join(save_path, subfoler_name, f'{img_name}.png')
                img_lr = imgs_lr[:, i, ...]  # 按帧取
                result_img_lr = tensor2img(img_lr)
                print('{}/{}: {}'.format(i + 1, len(img_paths_gt), save_img_path))
                imwrite(result_img_lr, save_img_path)
                i = i + 1
            print('=' * 100)


def quick_test():
    """根据超分后的结果快速验证 PSNR SSIM

    :return:
    """
    # # init
    # print(os.getcwd())imim
    # print(os.path.abspath('.'))

    # save the results of resize asymmetric sr
    save_resize_img = False
    videoinr = False

    # set sr and gt root path ---------------------------------------------------------------------------
    sr_root_ = '/data/lzk/workspace/lte/work_dir/swinir-liif/Vid4'
    gt_root = '/data/lzk/workspace/LbasicSR/datasets/Vid4/GT'
    # ---------------------------------------------------------------------------------------------------

    # set scale ------------
    scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
              2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
              3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    # scales = [(1.5, 4), (2, 4), (1.5, 3.5), (1.6, 3.05), (3.5, 2), (3.5, 1.75), (4, 1.4)]
    # scales = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    # scales = [(3.5, 2), (3.5, 1.75), (4, 1.4)]
    # scales = [(2, 3.75), (1.7, 3.75), (2.95, 3.75), (3.9, 2), (3.5, 1.5)]
    # scales = [(3.5, 2), (3.5, 1.75), (4, 1.4)]
    # ----------------------

    for scale in scales:
        if isinstance(scale, tuple):
            scale = scale
            sr_root = osp.join(sr_root_, 'x{}_x{}'.format(scale[0], scale[1]))
            save_resize_img = True
        else:
            scale = (scale, scale)
            sr_root = osp.join(sr_root_, 'x{}'.format(scale[0]))
        print('sr_root:', sr_root)
        print('gt_root:', gt_root)

        subfolers_gt = sorted(glob.glob(osp.join(gt_root, '*')))
        # subfolers_sr = sorted(glob.glob(osp.join(sr_root, '*')))

        psnr_all = []
        ssim_all = []

        for subfoler_gt in subfolers_gt:
            subfoler_name = osp.basename(subfoler_gt)
            print(subfoler_name)
            subfoler_sr = osp.join(sr_root, subfoler_name)

            img_paths_gt = sorted(list(scandir(subfoler_gt, full_path=True)))
            img_paths_sr = sorted(list(scandir(subfoler_sr, full_path=True)))

            max_id = len(img_paths_gt)
            print(max_id)

            # imgs_gt = read_img_seq(img_paths_gt, require_as_mod_crop=True, scale=scale)
            # imgs_sr = read_img_seq(img_paths_sr, require_as_mod_crop=True, scale=scale)

            # -------- origin ------------------------------------------
            # imgs_gt = [cv2.imread(v) for v in img_paths_gt]
            # print("gt size: {}".format(imgs_gt[0].shape))
            # imgs_sr = [cv2.imread(v) for v in img_paths_sr]
            # print("sr size: {}".format(imgs_sr[0].shape))

            # -------- videoinr ------------------------------------------------------
            # imgs_gt = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths_gt]
            # print("gt size: {}".format(imgs_gt[0].shape))
            # imgs_sr = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths_sr]
            # print("sr size: {}".format(imgs_sr[0].shape))

            imgs_gt = [cv2.imread(v, cv2.IMREAD_UNCHANGED) for v in img_paths_gt]
            print("gt size: {}".format(imgs_gt[0].shape))
            imgs_sr = [cv2.imread(v, cv2.IMREAD_UNCHANGED) for v in img_paths_sr]
            print("sr size: {}".format(imgs_sr[0].shape))

            # only for VideoINR sr city
            # if imgs_sr[0].shape[1] != imgs_gt[0].shape[1]:  # W !=
            #     print("VideoINR crop to adjust W ......")
            #     imgs_sr = [img[:, 0:imgs_gt[0].shape[1], :] for img in imgs_sr]
            # if imgs_sr[0].shape[0] != imgs_gt[0].shape[0]:    # H !=
            #     print("VideoINR crop to adjust H ......")
            #     imgs_sr = [img[0:imgs_gt[0].shape[0], ...] for img in imgs_sr]
            if videoinr:
                print("gt crop ......")
                imgs_gt = [as_mod_crop(img, scale) for img in imgs_gt]
                print("gt size: {}".format(imgs_gt[0].shape))
                if imgs_sr[0].shape != imgs_gt[0].shape:
                    print('VideoINR need to BI adjust')
                    # # torch BI -----------------------------------------------------------------------------------------
                    # imgs_sr = img2tensor(imgs_sr)
                    # imgs_sr = [T.Resize(size=(imgs_gt[0].shape[0], imgs_gt[0].shape[1]),
                    #                    interpolation=InterpolationMode.BICUBIC, antialias=True)(img) for img in imgs_sr]
                    # imgs_sr = [tensor2img(img) for img in imgs_sr]

                    # -------- cv2 BI -------------------------------------------------------------------------------------
                    imgs_sr = [cv2.resize(img, (imgs_gt[0].shape[1], imgs_gt[0].shape[0]), interpolation=cv2.INTER_AREA)
                               for img in imgs_sr]

                    # imgs_sr = [(img * 255.0).round().astype(np.uint8) for img in imgs_sr]

            # as_mod_crop ---------------------------------------------------------------------------
            if save_resize_img:
                print('resize asymmetric sr, only crop GT')
                print("after crop ......")
                imgs_gt = [as_mod_crop(img, scale) for img in imgs_gt]
                print("gt size: {}".format(imgs_gt[0].shape))
            else:
                if imgs_gt[0].shape == imgs_sr[0].shape:
                    print("no crop ......")
                else:
                    print("after crop ......")
                    imgs_gt = [as_mod_crop(img, scale) for img in imgs_gt]
                    print("gt size: {}".format(imgs_gt[0].shape))
                    if imgs_gt[0].shape == imgs_sr[0].shape:
                        pass
                    elif abs(imgs_gt[0].shape[0] - imgs_sr[0].shape[0]) < 5 or abs(
                            imgs_gt[0].shape[1] - imgs_sr[0].shape[1]) < 5:
                        print("The difference between sr and gt is not much, after crop ......")
                        if imgs_gt[0].shape[0] - imgs_sr[0].shape[0] > 0:
                            imgs_gt = [img_gt[0:imgs_sr[0].shape[0], ...] for img_gt in imgs_gt]
                            print("gt size: {}".format(imgs_gt[0].shape))
                        else:
                            imgs_sr = [img_sr[0:imgs_gt[0].shape[0], ...] for img_sr in imgs_sr]
                            print("sr size: {}".format(imgs_sr[0].shape))
                    else:
                        imgs_sr = [as_mod_crop(img, scale) for img in imgs_sr]  # list[ndarray(C,H,W)]
                    print("sr size: {}".format(imgs_sr[0].shape))

            # asymmetric scale BI resize
            if imgs_gt[0].shape != imgs_sr[0].shape and scale[0] != scale[1]:
                # use opencv to resize
                imgs_sr = [
                    cv2.resize(imgs_sr[i], (imgs_gt[0].shape[1], imgs_gt[0].shape[0]), interpolation=cv2.INTER_CUBIC)
                    for i in range(max_id)]
                print('after resize ......')
                print("sr size: {}".format(imgs_sr[0].shape))
                if save_resize_img:
                    i = 0
                    for img_path_sr in img_paths_sr:
                        img_name = osp.splitext(osp.basename(img_path_sr))[0]
                        save_img_path = osp.join(sr_root + '_resize', subfoler_name, f'{img_name}.png')
                        img_sr = imgs_sr[i]
                        print(save_img_path)
                        imwrite(img_sr, save_img_path)
                        i += 1

            imgs_psnr = []
            imgs_ssim = []

            for img_id, img_gt in enumerate(imgs_gt):
                img_psnr = calculate_psnr(imgs_sr[img_id], img_gt, crop_border=0, test_y_channel=True)
                imgs_psnr.append(img_psnr)
                # print(img_psnr)
                img_ssim = calculate_ssim(imgs_sr[img_id], img_gt, crop_border=0, test_y_channel=True)
                imgs_ssim.append(img_ssim)
                # print(img_ssim)

            psnr_all.append(mean(imgs_psnr))
            print("PSNR {}: {}".format(subfoler_name, mean(imgs_psnr)))
            ssim_all.append(mean(imgs_ssim))
            print("SSIM {}: {}".format(subfoler_name, mean(imgs_ssim)))
            print("-" * 50)

        print("avg PSNR: {}".format(mean(psnr_all)))
        print("avg SSIM: {}".format(mean(ssim_all)))

        print("current scale: {}".format(scale))
        print("avg PSNR: {:.2f}".format(np.float64(mean(psnr_all))))
        print("avg SSIM: {:.4f}".format(np.float64(mean(ssim_all))))
        print('{:.2f}/{:.4f}'.format(np.float64(mean(psnr_all)), np.float64(mean(ssim_all))))
        print("=" * 100 + "\n")


def quick_test_isr(args):
    """根据ISR后的结果快速验证 PSNR SSIM

    :return:
    """
    # # init
    # print(os.getcwd())
    # print(os.path.abspath('.'))

    # save the results of resize asymmetric sr
    save_resize_img = False
    is_y_channel = args.y_channel
    is_bicubic_test = args.is_bicubic_test

    # set sr and gt root path ---------------------------------------------------------------------------
    # sr_root_ = '/data/lzk/workspace/liif/work_dir/rdn-liif/Set5'
    # gt_root = '/data/lzk/workspace/LbasicSR/datasets/Set5/HR'
    sr_root_ = args.sr_root
    gt_root = args.gt_root
    if 'div2k' in sr_root_.lower():
        is_y_channel = False
    # ---------------------------------------------------------------------------------------------------

    # set scale ------------
    # scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
    #           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    # scales = [(1.5, 4), (2, 4), (1.5, 3.5), (1.6, 3.05), (3.5, 2), (3.5, 1.75), (4, 1.4)]
    # scales = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    # problem scales
    # scales = [1.4, 2.3, 2.8]
    # common scales
    # scales = [2, 3, 4]
    # scales = [3.5, 4]
    scales = [2, 3, 4, 1.4, 2.5, 3.7]
    # ----------------------

    for scale in scales:
        if isinstance(scale, tuple):
            scale = scale
            sr_root = osp.join(sr_root_, 'x{}_x{}'.format(scale[0], scale[1]))
            save_resize_img = True
        else:
            scale = (scale, scale)
            sr_root = osp.join(sr_root_, 'x{}'.format(scale[0]))
        if is_bicubic_test:
            sr_root = sr_root + "_bicubic_sr"
        print('sr_root:', sr_root)
        print('gt_root:', gt_root)

        sr_imgs_path = sorted(glob.glob(osp.join(sr_root, '*.png')))

        psnr_all = []
        ssim_all = []

        for sr_img_path in sr_imgs_path:
            sr_img_name = sr_img_path.split("/")[-1]  # 0801.png
            print(sr_img_name)
            gt_img_path = osp.join(gt_root, sr_img_name)

            img_gt = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
            print("gt shape: {}".format(img_gt.shape))
            img_sr = cv2.imread(sr_img_path, cv2.IMREAD_UNCHANGED)
            print("sr shape: {}".format(img_sr.shape))

            # as_mod_crop ---------------------------------------------------------------------------
            if save_resize_img:
                print('resize asymmetric sr, only crop GT')
                print("after crop ......")
                img_gt = as_mod_crop(img_gt, scale)
                print("gt shape: {}".format(img_gt.shape))
            else:
                if img_gt.shape == img_sr.shape:
                    print("no crop ......")
                else:
                    print("gt after crop ......")
                    img_gt = as_mod_crop(img_gt, scale)
                    print("gt shape: {}".format(img_gt.shape))
                    if img_gt.shape == img_sr.shape:
                        pass
                    elif abs(img_gt.shape[0] - img_sr.shape[0]) < 5 or abs(img_gt.shape[1] - img_sr.shape[1]) < 5:
                        print("The difference between sr and gt is not much, after crop ......")
                        if img_gt.shape[0] - img_sr.shape[0] > 0:  # H: GT > SR
                            img_gt = img_gt[0:img_sr.shape[0], ...]
                            print("gt shape: {}".format(img_gt.shape))
                        else:
                            img_sr = img_sr[0:img_gt.shape[0], ...]
                            # print("sr shape: {}".format(img_sr.shape))
                        if img_gt.shape[1] - img_sr.shape[1] > 0:  # W: GT > SR
                            img_gt = img_gt[:, 0:img_sr.shape[1], ...]
                            print("gt shape: {}".format(img_gt.shape))
                        else:
                            img_sr = img_sr[:, 0:img_gt.shape[1], ...]
                            # print("sr shape: {}".format(img_sr.shape))
                    else:
                        img_sr = as_mod_crop(img_sr, scale)  # list[ndarray(C,H,W)]
                    print("sr shape: {}".format(img_sr.shape))

            # asymmetric scale BI resize
            if img_gt.shape != img_sr.shape and scale[0] != scale[1]:
                # use opencv to resize
                img_sr = cv2.resize(img_sr, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                print('after resize ......')
                print("sr shape: {}".format(img_sr.shape))
                if save_resize_img:
                    save_img_path = osp.join(sr_root + '_resize', sr_img_name)
                    print(save_img_path)
                    imwrite(img_sr, save_img_path)

            img_psnr = calculate_psnr(img_sr, img_gt, crop_border=0, test_y_channel=is_y_channel)
            psnr_all.append(img_psnr)
            print("- PSNR: {}".format(img_psnr))
            img_ssim = calculate_ssim(img_sr, img_gt, crop_border=0, test_y_channel=is_y_channel)
            ssim_all.append(img_ssim)
            print("- SSIM: {}".format(img_ssim))
            print("-" * 100)

        print("avg PSNR: {}".format(mean(psnr_all)))
        print("avg SSIM: {}".format(mean(ssim_all)))

        print("current scale: {}".format(scale))
        if is_y_channel:
            print("Y channel")
        else:
            print("RGB channel")
        print("avg PSNR: {:.2f}".format(np.float64(mean(psnr_all))))
        print("avg SSIM: {:.4f}".format(np.float64(mean(ssim_all))))
        print('{:.2f}/{:.4f}'.format(np.float64(mean(psnr_all)), np.float64(mean(ssim_all))))
        print("=" * 100 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr_root', default='/data/lzk/workspace/LbasicSR/datasets/Urban100/arbitrary_scale_BI')
    parser.add_argument('--gt_root', default='/data/lzk/workspace/LbasicSR/datasets/Urban100/HR')
    parser.add_argument('--y_channel', default=True)
    parser.add_argument('--is_bicubic_test', default=True)

    args = parser.parse_args()

    # hr_img = torch.randn([1, 7, 3, 720, 576])
    # print(hr_img.size())
    # lr_img = arbitrary_scale_downsample(hr_img, 4.0)
    # print(lr_img.size())

    # ======================
    # downsample_img()
    # downsample_video(data_root='/data/lzk/workspace/LbasicSR/datasets/UDM10', gt_dir='GT', degradation='BI')
    quick_test()
    # quick_test_isr(args)
    # ======================

    # ==================================
    # scales = np.linspace(11, 40, 30)
    # print(scales)
    # scales = scales / 10
    # print(scales)
    # print(type(scales))
    # for scale in scales:
    #     print("scale =", scale)
    #     step = cal_step(scale)
    #     print(step)
    #     print("="*50)
    # ==================================
