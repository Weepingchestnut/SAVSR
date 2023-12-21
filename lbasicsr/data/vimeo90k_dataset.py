import random
from pathlib import Path

import torch
from torch.utils import data as data

from lbasicsr.data.data_util import arbitrary_scale_downsample, generate_frame_indices
from lbasicsr.data.transforms import augment, paired_random_crop, single_random_crop, single_random_spcrop, mod_crop
from lbasicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from lbasicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Vimeo90KDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains the following items, separated by a white space.

    1. clip name;
    2. frame number;
    3. image shape

    Examples:

    ::
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    - Key examples: "00001/0001"
    - GT (gt): Ground-Truth;
    - LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:

    ::

        num_frame | frame list
                1 | 4
                3 | 3,4,5
                5 | 2,3,4,5,6
                7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        
        # indices of input images
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]
        
        # for more frame input ------------------------------------------------------------------------------
        if opt['num_frame'] > 7:
            self.neighbor_list = [i + (9 - 7) // 2 for i in range(7)]               # [1, 2, 3, 4, 5, 6, 7]
            # head_list = generate_frame_indices(1, 7, 13, padding='reflection')      # [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 5]
            # tail_list = generate_frame_indices(7, 7, 13, padding='reflection')      # [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, -1]
            pad_len = (opt['num_frame'] - 7) // 2
            head_list = self.neighbor_list[1:1+pad_len]
            tail_list = self.neighbor_list[6-pad_len:-1]
            self.neighbor_list = head_list[::-1] + self.neighbor_list + tail_list[::-1]         # [4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4]
        # ---------------------------------------------------------------------------------------------------

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        # for x3 scale -------------------------------------------------------
        if scale == 3:
            img_gt = mod_crop(img_gt, scale)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class ASVimeo90KDataset(Vimeo90KDataset):
    def __init__(self, opt):
        super(ASVimeo90KDataset, self).__init__(opt)

        self.epoch = 0
        self.init_int_scale = opt.get('init_int_scale', False)
        self.single_scale_ft = opt.get('single_scale_ft', False)
        self.CL_train_set = opt.get('CL_train_set', None)
        self.only_sy_scale = opt.get('only_sy_scale', False)
        
        self.lq_size = opt.get('lq_size', 60)
        self.max_scale = opt.get('max_scale', 4)

        if self.only_sy_scale:
            self.scale_h_list = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
            ]

            self.scale_w_list = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
            ]
        else:
            self.scale_h_list = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                1.5, 1.5, 1.5, 1.5, 1.5,
                2.0, 2.0, 2.0, 2.0, 2.0,
                2.5, 2.5, 2.5, 2.5, 2.5,
                3.0, 3.0, 3.0, 3.0, 3.0,
                3.5, 3.5, 3.5, 3.5, 3.5,
                4.0, 4.0, 4.0, 4.0, 4.0,
                # 6.0, 7.0, 7.5, 8.0
            ]

            self.scale_w_list = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                2.0, 2.5, 3.0, 3.5, 4.0,
                1.5, 2.5, 3.0, 3.5, 4.0,
                1.5, 2.0, 3.0, 3.5, 4.0,
                1.5, 2.0, 2.5, 3.5, 4.0,
                1.5, 2.0, 2.5, 3.0, 4.0,
                1.5, 2.0, 2.5, 3.0, 3.5,
                # 6.0, 7.0, 7.5, 8.0
            ]
        
        if opt.__contains__('scale_h_list') and opt.__contains__('scale_w_list'):
            self.scale_h_list = opt['scale_h_list']
            self.scale_w_list = opt['scale_w_list']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring GT frames
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'

            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)  # ndarry (256, 448, 3) [0, 1]

            img_gts.append(img_gt)

        # augmentation - flip, rotate
        img_gts = single_random_crop(img_gts, 
                                     gt_patch_size=(self.lq_size * self.max_scale, self.lq_size * self.max_scale))
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        img_gts = img2tensor(img_gts)  # list
        img_gts = torch.stack(img_gts, dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'gt': img_gts, 'key': key}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def cl_train_stg(self):
        if self.epoch >= self.CL_train_set[0]:
            idx_scale = random.randrange(0, len(self.scale_h_list))
            scale_h = self.scale_h_list[idx_scale]
            scale_w = self.scale_w_list[idx_scale]
            return scale_h, scale_w
        if self.epoch % 10 <= self.CL_train_set[1]:
            scale_h, scale_w = 4, 4
        elif self.CL_train_set[1] < self.epoch % 10 <= self.CL_train_set[2]:
            scale_h = random.randint(2, 4)
            scale_w = scale_h
        elif self.epoch % 10 > self.CL_train_set[2]:
            idx_scale = random.randrange(0, len(self.scale_h_list))
            scale_h = self.scale_h_list[idx_scale]
            scale_w = self.scale_w_list[idx_scale]

        return scale_h, scale_w

    def as_collate_fn(self, batch):
        out_batch = {}
        elem = batch[0]
        for key in elem.keys():
            if key == 'gt':
                gts_list = [d[key] for d in batch]
                elem_cur = gts_list[0]
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in gts_list])
                    storage = elem_cur.storage()._new_shared(numel)
                    out = elem_cur.new(storage)
                out_batch[key] = torch.stack(gts_list, 0, out=out)  # GT: B x T x C x H x W
            elif key == 'key':
                key_list = [d[key] for d in batch]
                out_batch[key] = key_list

        # get arbitrary scale ------------------------------------------------
        if self.CL_train_set is not None:
            scale_h, scale_w = self.cl_train_stg()
        elif self.single_scale_ft:
            scale_h = self.opt['scale'][0]
            scale_w = self.opt['scale'][1]
        elif self.epoch == 0 and self.init_int_scale:
            scale_h = random.randint(2, 4)
            scale_w = scale_h
        else:
            idx_scale = random.randrange(0, len(self.scale_h_list))
            scale_h = self.scale_h_list[idx_scale]
            scale_w = self.scale_w_list[idx_scale]
        lq_size = self.opt['lq_size']
        gt_size = (round(lq_size * scale_h), round(lq_size * scale_w))

        b, t, c, h, w = out_batch['gt'].size()
        out_batch['gt'] = single_random_crop(out_batch['gt'].view(-1, c, h, w), gt_size)
        out_batch['lq'] = arbitrary_scale_downsample(out_batch['gt'], (scale_h, scale_w), self.opt['downsample_mode'])
        out_batch['gt'] = out_batch['gt'].view(b, t, c, gt_size[0], gt_size[1])
        out_batch['gt'] = out_batch['gt'][:, t//2]
        out_batch['lq'] = out_batch['lq'].view(b, t, c, lq_size, lq_size)
        out_batch['scale'] = (scale_h, scale_w)

        return out_batch


@DATASET_REGISTRY.register()
class ASVimeo90KRecurrentDataset(ASVimeo90KDataset):
    def __init__(self, opt):
        super(ASVimeo90KRecurrentDataset, self).__init__(opt)

        self.flip_sequence = opt.get('flip_sequence', False)
        self.neighbor_list = [1, 2, 3, 4, 5, 6, 7]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring GT frames
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'

            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)  # ndarry (256, 448, 3) [0, 1]

            img_gts.append(img_gt)

        # make the img size always 256x448x3 --------------------
        # if img_gts[0].shape != (256, 448, 3):
        #     img_gts = single_random_spcrop(img_gts, (256, 448))
        # -------------------------------------------------------
        
        # randomly crop
        img_gts = single_random_crop(img_gts,
                                     gt_patch_size=(self.lq_size * self.max_scale, self.lq_size * self.max_scale))

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        img_gts = img2tensor(img_gts)  # list
        img_gts = torch.stack(img_gts, dim=0)

        if self.flip_sequence:  # flip the sequence: 7 frames to 14 frames ([1, 2, 3, 4, 5, 6, 7] --> [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1])
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)

        return {'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)

    def as_collate_fn(self, batch):
        out_batch = {}
        elem = batch[0]
        for key in elem.keys():
            if key == 'gt':
                gts_list = [d[key] for d in batch]
                elem_cur = gts_list[0]
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in gts_list])
                    storage = elem_cur.storage()._new_shared(numel)
                    out = elem_cur.new(storage)
                out_batch[key] = torch.stack(gts_list, 0, out=out)  # GT: B x T x C x H x W
            elif key == 'key':
                key_list = [d[key] for d in batch]
                out_batch[key] = key_list

        # get arbitrary scale --------------------------------------------------
        if self.CL_train_set is not None:
            scale_h, scale_w = self.cl_train_stg()
        elif self.single_scale_ft:
            scale_h = self.opt['scale'][0]
            scale_w = self.opt['scale'][1]
        elif self.epoch == 0 and self.init_int_scale:
            scale_h = random.randint(2, 4)
            scale_w = scale_h
        else:
            idx_scale = random.randrange(0, len(self.scale_h_list))
            scale_h = self.scale_h_list[idx_scale]
            scale_w = self.scale_w_list[idx_scale]
        lq_size = self.opt['lq_size']
        gt_size = (round(lq_size * scale_h), round(lq_size * scale_w))
        # ----------------------------------------------------------------------
        b, t, c, h, w = out_batch['gt'].size()
        out_batch['gt'] = single_random_crop(out_batch['gt'].view(-1, c, h, w), gt_size)
        out_batch['lq'] = arbitrary_scale_downsample(out_batch['gt'], (scale_h, scale_w), self.opt['downsample_mode'])
        out_batch['gt'] = out_batch['gt'].view(b, t, c, gt_size[0], gt_size[1])
        out_batch['lq'] = out_batch['lq'].view(b, t, c, lq_size, lq_size)
        # out_batch['scale'] = torch.tensor(scale)
        out_batch['scale'] = (scale_h, scale_w)

        return out_batch


@DATASET_REGISTRY.register()
class Vimeo90KRecurrentDataset(Vimeo90KDataset):

    def __init__(self, opt):
        super(Vimeo90KRecurrentDataset, self).__init__(opt)

        self.flip_sequence = opt['flip_sequence']
        self.neighbor_list = [1, 2, 3, 4, 5, 6, 7]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)  # ndarray (85, 149, 3)
            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)  # (256, 448, 3)
            # for x3 scale -------------------------------------------------------
            if scale == 3:
                img_gt = mod_crop(img_gt, scale)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        img_gts = torch.stack(img_results[7:], dim=0)

        if self.flip_sequence:  # flip the sequence: 7 frames to 14 frames
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
