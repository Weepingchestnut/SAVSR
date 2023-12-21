import logging
import torch
from os import path as osp

from lbasicsr.data import build_dataloader, build_dataset
from lbasicsr.models import build_model
from lbasicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from lbasicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers  新建logger并初始化，打印基础信息
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='lbasicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader    创建测试集和 dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        # test_scale = opt['scale']
        if 'downsampling_scale' in test_loader.dataset.opt.keys():
            test_scale = test_loader.dataset.opt['downsampling_scale']
        elif 'val_scale' in test_loader.dataset.opt.keys():
            test_scale = test_loader.dataset.opt['val_scale']
        else:
            test_scale = opt['scale']
        logger.info(f'Testing {test_set_name} x{test_scale}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
