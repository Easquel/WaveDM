import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet, DiffusionUNet_Global
import torch.distributed as dist
from torchvision.utils import make_grid

from einops import rearrange


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):

        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        elif isinstance(module, nn.parallel.DistributedDataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.parallel.DistributedDataParallel(module_copy)

        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b, total, use_global=False,use_FFT=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    if not use_FFT:
        x_cond = x0[:, :3, :, :]
    else:
        x_fft = torch.fft.fft2(x0[:, :3, :, :], dim=(-2, -1))
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        x_cond = torch.cat([x0[:, :3, :, :],x_amp,x_phase],dim=1)
    if use_global == False:
        output = model(torch.cat([x_cond, x], dim=1), t.float())
    else:
        output = model(torch.cat([x_cond, x], dim=1), t.float(), total)
    x0_pred = (x - output * (1 - a).sqrt()) / a.sqrt()
    simple_loss = (e - output).square().sum(dim=(1, 2, 3))
    mse_loss = (x0[:, 3:, :, :] - x0_pred).square().sum(dim=(1, 2, 3))
    return simple_loss.mean(dim=0), output, x0_pred, mse_loss.mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        if self.config.data.lap == True:
            from models.Lap import Lap_Pyramid_Conv,Lap_high_trans
            num_high = 2
            self.lap = Lap_Pyramid_Conv(num_high=num_high,channels=config.model.in_channels * 2 if config.data.conditional else config.model.in_channels,device=self.device)
            self.lap.to(self.device)
            self.lap_high_trans = Lap_high_trans(res_num=3,res_dim=32,num_high=num_high).to(self.device)
            self.lap_opt = torch.optim.Adam(self.lap_high_trans.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
            self.lap_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.lap_opt,self.config.training.n_epochs)

        if config.data.global_attn == True:
            self.model = DiffusionUNet_Global(config)
        else:
            self.model = DiffusionUNet(config)
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        print("my local rank",self.args.local_rank)
        if os.path.isfile(self.args.resume):
            if dist.get_rank() >= 0:
                self.load_ddm_ckpt(self.args.resume)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)#,find_unused_parameters=True)
        if self.config.data.lap == True:
            self.lap_high_trans = torch.nn.parallel.DistributedDataParallel(self.lap_high_trans,device_ids=[self.args.local_rank],output_device=self.args.local_rank)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, self.device)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if self.config.data.lap == True:
            miss,unexp = self.lap_high_trans.load_state_dict(checkpoint['lap_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train_with_lap_dec(self,x):
        lap_pyr = self.lap.pyramid_decom(x)
        x = lap_pyr[-1]
        self.lap_high_trans.train()
        lap_pyr_inp = []
        for level in range(self.lap.num_high+1):
            lap_pyr_inp.append(lap_pyr[level][:, :3, :, :])
        pyr_inp_trans = self.lap_high_trans(lap_pyr_inp)
        return x, lap_pyr, pyr_inp_trans

    def train_the_lap_loss(self,pyr_inp_trans,lap_pyr):
        loss_mse = nn.MSELoss()
        loss_lap_trans = 0
        for level in range(self.lap.num_high):
            loss_lap_trans += loss_mse(pyr_inp_trans[level], lap_pyr[level][:, 3:, :, :])

        self.lap_opt.zero_grad()
        loss_lap_trans.backward(retain_graph=True)
        self.lap_opt.step()
        return loss_lap_trans

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, _ = DATASET.get_loaders()
        num_of_pixel = self.config.model.pred_channels * self.config.data.image_size ** 2

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            e_pred_diff_list = []
            for i, (x, y, total) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)

                if self.config.data.global_attn == True:
                    total = total.flatten(start_dim=0, end_dim=1) if total.ndim == 5 else total
                    total = total.to(self.device)
                    total = data_transform(total)

                if self.config.data.lap:
                    x, lap_pyr, pyr_inp_trans = self.train_with_lap_dec(x)
                    loss_trans = self.train_the_lap_loss(pyr_inp_trans, lap_pyr)

                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss, e_pred, x0_pred, mse_loss = noise_estimation_loss(self.model, x, t, e, b, total, self.config.data.global_attn, self.config.data.use_FFT)
                e_pred_diff_list.append(e_pred.mean().item()-e.mean().item())

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, loss mean: {loss.item()/num_of_pixel}, mse loss mean: {mse_loss.item()/num_of_pixel},data time: {data_time / (i+1)} \n")
                    if self.config.data.lap:
                        print(f"loss trans: {loss_trans.item()/2}\n")

                self.optimizer.zero_grad()
                if self.config.training.use_mse:
                    mse_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                self.ema_helper.update(self.model)
                data_start = time.time()

                if (dist.get_world_size()>1 and self.step % self.config.training.validation_freq == 0) or (dist.get_world_size()==1 and self.step % 30 == 0):
                    if dist.get_rank() == 0:
                        self.model.eval()
                        _, val_loader = DATASET.get_loaders(parse_patches=False, validation=self.args.test_set)
                        # self.sample_validation_patches(val_loader, self.step)
                        self.restore(val_loader, validation=self.args.test_set, r=self.args.grid_r, epoch=epoch)

                if self.step % 10 ==0 :#self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    if dist.get_rank() == 0:
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model.module.state_dict(),
                            #'lap_state_dict': self.lap_high_trans.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_epoch' + str(epoch+1) + '_ddpm'))
                        print('checkpoint saved')

            if self.config.data.lap:
                self.lap_scheduler.step()

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None,total=None,use_global=False):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size,
                                                              total=total,use_global=use_global,use_FFT=self.config.data.use_FFT)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                import pdb

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)

            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))

    def restore_lap_dec(self,x_cond):
        lap_pyr = self.lap.pyramid_decom(x_cond)
        x_cond = lap_pyr[-1]
        x_gt_lowf = x_cond[:, 3:, :, :]  # low frequency

        self.lap_high_trans.eval()
        lap_pyr_inp = []
        for level in range(self.lap.num_high+1):
            lap_pyr_inp.append(lap_pyr[level][:, :3, :, :])
        pyr_inp_trans = self.lap_high_trans(lap_pyr_inp)

        return x_cond,x_gt_lowf,lap_pyr,pyr_inp_trans

    def restore_lap_rec(self,lap_pyr,x_output,x_gt_lowf,pyr_inp_trans):
        lap_pyr_output = []
        for level in range(self.lap.num_high):
            lap_pyr_output.append(lap_pyr[level])
        lap_pyr_output.append(torch.cat([x_output.to(self.device), x_gt_lowf], dim=1))

        # for check
        lap_pyr_check = []
        for level in range(self.lap.num_high):
            lap_pyr_check.append(lap_pyr[level])
        lap_pyr_check.append(torch.cat([x_gt_lowf, x_output.to(self.device)], dim=1))
        x_check_not_trans = self.lap.pyramid_recons(lap_pyr_check)
        x_check3 = x_check_not_trans[:, :3, :, :]
        x_check4 = x_check_not_trans[:, 3:, :, :]

        # for check trans
        lap_pyr_check_trans = []
        for level in range(self.lap.num_high-1):
            lap_pyr_check_trans.append(torch.cat([pyr_inp_trans[level], pyr_inp_trans[level]], dim=1))
        lap_pyr_check_trans.append(torch.cat([x_gt_lowf, x_output.to(self.device)], dim=1))
        x_check = self.lap.pyramid_recons(lap_pyr_check_trans)
        x_check1 = x_check[:, :3, :, :]
        x_check2 = x_check[:, 3:, :, :]

        x_output = self.lap.pyramid_recons(lap_pyr_output)
        x_cond = self.lap.pyramid_recons(lap_pyr)  # total frequency
        x_cond = x_cond[:, :3, :, :]
        x_gt_rec = x_output[:,3:,:,:]
        x_output = x_output[:, :3, :, :]
        return x_output,x_cond,x_check1,x_check2,x_gt_rec,x_check3,x_check4

    def standard_output(self,x_output,x_cond):
        x_output += x_cond.mean().item() - x_output.mean().item()
        x_output = (x_output - x_output.min()) / (x_output.max() - x_output.min())
        x_output = x_output * 2 - 1
        x_output += x_cond.mean().item() - x_output.mean().item()
        return x_output

    def restore(self, val_loader, validation='snow', r=None, epoch=0):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            all_samples = []
            for i, (x, y, total) in enumerate(val_loader):

                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x.to(self.device)
                x_cond = data_transform(x_cond)

                if self.config.data.global_attn == True:
                    total = total.to(self.device)
                    total = data_transform(total)

                if self.config.data.lap:
                    x_cond, x_gt_lowf, lap_pyr,pyr_inp_trans = self.restore_lap_dec(x_cond)

                x_cond = x_cond[:,:3,:,:]

                x_output = self.diffusive_restoration(x_cond, r=r, total=total,use_global=self.config.data.global_attn)

                #x_output = self.standard_output(x_output,x_cond) #不一定要有

                if self.config.data.lap:
                    x_output, x_cond, x_check1, x_check2, x_gt_rec, x_check3, x_check4 = self.restore_lap_rec(lap_pyr,x_output,x_gt_lowf,pyr_inp_trans)

                x_output = inverse_data_transform(x_output)
                x_cond = inverse_data_transform(x_cond)

                x_gt = x[:, 3:, :, :]
                x_gt = data_transform(x_gt)
                x_gt = inverse_data_transform(x_gt)

                #for check
                if self.config.data.lap:
                    x_check1 = inverse_data_transform(x_check1)
                    x_check2 = inverse_data_transform(x_check2)
                    x_check3 = inverse_data_transform(x_check3)
                    x_check4 = inverse_data_transform(x_check4)
                    x_gt_rec = inverse_data_transform(x_gt_rec)

                all_samples.append(x_cond.cpu())
                all_samples.append(x_output.cpu())
                all_samples.append(x_gt)
                # all_samples.append(x_check1.cpu())
                # all_samples.append(x_check2.cpu())
                if i==1:
                    break
            
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=3)
            utils.logging.save_image(grid, os.path.join(image_folder, f"{y}_output" + "_epoch" + str(epoch) + ".png"))
            # utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))


    def diffusive_restoration(self, x_cond, r=None, total=None,use_global=False):
        p_size = self.config.data.patch_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.device)
        x_output = self.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size,total=total,use_global=use_global)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        if h_list[-1]+output_size<h:
            h_list.append(h-output_size)
        if w_list[-1]+output_size<w:
            w_list.append(w-output_size)
        return h_list, w_list
