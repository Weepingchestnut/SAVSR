from collections import OrderedDict
import math
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .video_recurrent_model import ASVideoRecurrentModel, VideoRecurrentModel
from .video_base_model import VideoBaseModel
from ..utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ASVSRModel(VideoBaseModel):
    """ASVSR Model

    """

    def __init__(self, opt):
        super(ASVSRModel, self).__init__(opt)

    def optimize_parameters(self, current_iter):
        if hasattr(self, 'scale'):
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                self.net_g.module.set_scale(self.scale)
            else:
                self.net_g.set_scale(self.scale)
            print('current iteration scale: {}'.format(self.scale))

        super(ASVSRModel, self).optimize_parameters(current_iter)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            # if self.opt['is_train']:
            #     if self.opt['datasets']['val'].__contains__('downsampling_scale'):
            #         self.net_g_ema.set_scale(self.opt['datasets']['val']['downsampling_scale'])
            # else:
            #     self.net_g_ema.set_scale(self.opt['scale'])
            self.net_g_ema.set_scale(self.opt['scale'])
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            # if self.opt['is_train']:
            #     if self.opt['datasets']['val'].__contains__('downsampling_scale'):
            #         if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
            #             self.net_g.module.set_scale(self.opt['datasets']['val']['downsampling_scale'])
            #         else:
            #             self.net_g.set_scale(self.opt['datasets']['val']['downsampling_scale'])
            # else:
            #     if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
            #         self.net_g.module.set_scale(self.opt['scale'])
            #     else:
            #         self.net_g.set_scale(self.opt['scale'])
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                self.net_g.module.set_scale(self.opt['scale'])
            else:
                self.net_g.set_scale(self.opt['scale'])
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)   # network influence
            self.net_g.train()


@MODEL_REGISTRY.register()
class ASVSRRecurrentModel(ASVideoRecurrentModel):
    """ASVSR Recurrent Model

    """

    def __init__(self, opt):
        super(ASVSRRecurrentModel, self).__init__(opt)
        
        if 'train' in self.opt:
            self.loss_frame_seq = list(range(self.opt['train']['sub_frame'], 
                                             self.opt['datasets']['train']['num_frame'] - self.opt['train']['sub_frame']))
            self.alpha = self.opt['train']['alpha']
            
            # Mixed Precision Training
            self.scaler = GradScaler()

    def optimize_parameters(self, current_iter):
        # set current scale
        if hasattr(self, 'scale'):
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                self.net_g.module.set_scale(self.scale)
            else:
                self.net_g.set_scale(self.scale)
            if current_iter % self.opt['logger']['print_freq'] == 0:
                print('current iteration scale: {}'.format(self.scale))
        
        self.optimizer_g.zero_grad()
        
        with autocast():
            self.output = self.net_g(self.lq, self.opt['train']['sub_frame'])

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                # ------ loss ----------------------------------------------------------------------------
                # l_pix = self.cri_pix(self.output[0], self.gt[:, self.loss_frame_seq, ...]) + \
                #         self.alpha * self.cri_pix(self.output[1], self.gt[:, self.loss_frame_seq, ...])
                # --> pre_sr is bilinear upscale
                l_pix = self.cri_pix(self.output, self.gt[:, self.loss_frame_seq, ...])
                # ----------------------------------------------------------------------------------------
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

                l_total_v = l_total.detach()
                if l_total_v > 5 or l_total_v < 0 or math.isnan(l_total_v):
                    raise RuntimeWarning(f'loss error {l_total_v}')

            # -----------------------------------
            # Mix Precision Training
            # l_total.backward()
            # self.optimizer_g.step()
            # -->
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            # -----------------------------------
            
            self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def test(self):
        n = self.lq.size(1)     # 当前视频序列帧数
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                self.net_g.module.set_scale(self.scale)
            else:
                self.net_g.set_scale(self.scale)
            # --------------------------------------
            # self.output = self.net_g(self.lq)[0]
            # --> pre_sr is bilinear upscale
            self.output = self.net_g(self.lq)
            # --------------------------------------

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
    


