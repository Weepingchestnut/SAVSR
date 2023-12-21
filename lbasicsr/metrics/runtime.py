from typing import Tuple, Union
import torch
from torch.backends import cudnn
import torchvision
from torchvision.transforms import InterpolationMode
import tqdm
import numpy as np


def VSR_runtime_test(model: torch.nn.Module, input: torch.Tensor, scale: Union[Tuple, float, int], post_BI: bool = False,
                     repetitions: int = 300):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # repetitions = 300
    
    cudnn.benchmark = True
    
    if isinstance(scale, Tuple):
        scale = scale
    else:
        scale = (scale, scale)
    
    b, t, c, h, w = input.shape
    H, W = round(h * scale[0]), round(w * scale[1])
    
    if model.training:
        model.eval()
    
    with torch.no_grad():
        sr = model(input)
        if sr.shape[-2] == H and sr.shape[-1] == W:
            post_BI = False
        else:
            post_BI = True
            print('use BI post-processing ...')
    
    print('Warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(input)
    
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    
    print('Testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            if not post_BI:
                starter.record()
                _ = model(input)
                ender.record()
            else:
                starter.record()
                over_scale_sr = model(input)
                _ = torchvision.transforms.Resize(
                    size=(H, W), interpolation=InterpolationMode.BICUBIC, antialias=True)(
                        over_scale_sr.view(-1, c, over_scale_sr.shape[-2], over_scale_sr.shape[-1]))
                ender.record()
            
            torch.cuda.synchronize()    # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)     # 从 starter 到 ender 之间用时，单位为毫秒ms
            timings[rep] = curr_time
    
    avg = timings.sum()/repetitions
    print('\nAverage Runtime: {:.3f} ms\n'.format(avg))
    



