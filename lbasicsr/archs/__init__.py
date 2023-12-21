import importlib
from copy import deepcopy
from os import path as osp

from lbasicsr.utils import get_root_logger, scandir
from lbasicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'lbasicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    # ==================================================
    # 根据配置文件yml中的 arch 类型，创建相应的 network 实例
    net = ARCH_REGISTRY.get(network_type)(**opt)
    # ==================================================
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
