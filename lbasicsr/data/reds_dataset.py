import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from lbasicsr.data.data_util import arbitrary_scale_downsample
from lbasicsr.data.transforms import augment, paired_random_crop, single_random_crop
from lbasicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from lbasicsr.utils.flow_util import dequantize_flow
from lbasicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class REDSDataset(data.Dataset):
    """REDS dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        # self.flow_root = Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        self.flow_root = None
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'X4/{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.num_frame:]

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.num_half_frames, torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_flows: (t, 2, h, w)
        # img_gt: (c, h, w)
        # key: str
        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class ASREDSDataset(REDSDataset):

    def __init__(self, opt):
        super(ASREDSDataset, self).__init__(opt)

        self.epoch = 0
        self.init_int_scale = opt.get('init_int_scale', False)
        
        self.single_scale_ft = opt.get('single_scale_ft', False)
        if self.single_scale_ft:
            if isinstance(self.opt['scale'], tuple):
                self.opt['scale'] = self.opt['scale']
            else:
                self.opt['scale'] = (self.opt['scale'], self.opt['scale'])
        
        self.CL_train_set = opt.get('CL_train_set', None)
        self.only_sy_scale = opt.get('only_sy_scale', False)

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
            ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, f'Wrong length of neighbor list: {len(neighbor_list)}'

        # get the neighboring GT frames -----------------------------------------------
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
        # -----------------------------------------------------------------------------

        img_gts = img2tensor(img_gts)  # list
        img_gts = torch.stack(img_gts, dim=0)

        return {'gt': img_gts, 'key': key}

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def cl_train_stg(self):
        """Continuous learning training strategies.

        Returns:
            int: scale_h
            int: scale_w
        """
        if self.epoch >= self.CL_train_set[0]:
            idx_scale = random.randrange(0, len(self.scale_h_list))
            scale_h = self.scale_h_list[idx_scale]
            scale_w = self.scale_w_list[idx_scale]
            return scale_h, scale_w
        if self.epoch % 10 <= self.CL_train_set[1]:     # only scale = 4
            scale_h, scale_w = 4, 4
        elif self.CL_train_set[1] < self.epoch % 10 <= self.CL_train_set[2]:    # scale = 2, 3, 4
            scale_h = random.randint(2, 4)
            scale_w = scale_h
        elif self.epoch % 10 > self.CL_train_set[2]:    # full scales
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
        out_batch['gt'] = out_batch['gt'][:, t // 2]
        out_batch['lq'] = out_batch['lq'].view(b, t, c, lq_size, lq_size)
        out_batch['scale'] = (scale_h, scale_w)

        return out_batch


@DATASET_REGISTRY.register()
class REDSRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'X4/{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class ASREDSRecurrentDataset(REDSRecurrentDataset):

    def __init__(self, opt):
        super(ASREDSRecurrentDataset, self).__init__(opt)
        # arbitrary-scale setting
        self.epoch = 0
        self.init_int_scale = opt.get('init_int_scale', False)
        self.single_scale_ft = opt.get('single_scale_ft', False)
        self.CL_train_set = opt.get('CL_train_set', None)
        self.only_sy_scale = opt.get('only_sy_scale', False)
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

        # scale = self.opt['scale']
        # gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        # img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                # img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                # img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            # img_bytes = self.file_client.get(img_lq_path, 'lq')
            # img_lq = imfrombytes(img_bytes, float32=True)
            # img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        # img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        img_gts = single_random_crop(img_gts, 
                                     gt_patch_size=(self.opt['lq_size'] * self.max_scale, self.opt['lq_size'] * self.max_scale))

        # augmentation - flip, rotate
        # img_lqs.extend(img_gts)
        # img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        # img_results = img2tensor(img_results)
        # img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        # img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
        img_gts = img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
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
