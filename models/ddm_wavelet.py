import os
import time
import glob
import pdb
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
from torchvision.transforms.functional import crop
from utils.sampling import compute_alpha
from models.wavelet import WaveletTransform
from models.arch import HFRM
# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm
# https://github.com/IGITUGraz/WeatherDiffusion


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


def noise_estimation_loss(model, x0, t, e, b, total=None, use_global=False, inp_channels=3,pred_channels=3,use_other_channels=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_inp = x0[:, :inp_channels, :, :] # bx48xhxw
    x_tar = x0[:, inp_channels:inp_channels+pred_channels, :, :]
    xt = x_tar * a.sqrt() + e * (1.0 - a).sqrt()
    if use_other_channels:
        x_other = x0[:, inp_channels+pred_channels:, :, :]
        x = torch.cat([xt,x_other],dim=1)
    if use_global == False:
        output = model(torch.cat([x_inp, x], dim=1), t.float())
    else:
        output = model(torch.cat([x_inp, x], dim=1), t.float(), total)

    x0_pred = (xt - output * (1 - a).sqrt()) / a.sqrt()
    simple_loss = (e - output).square().sum(dim=(1, 2, 3))
    mse_loss = (x_tar - x0_pred).square().sum(dim=(1, 2, 3))
    return simple_loss.mean(dim=0), output, x0_pred, mse_loss.mean(dim=0)


class DenoisingDiffusion_Wavelet(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.wavelet_dec = WaveletTransform(scale=2, dec=True).to(self.device)
        self.wavelet_rec = WaveletTransform(scale=2, dec=False).to(self.device)

        img_channel = 3
        dim = 32
        enc_blks = [2, 2, 2, 4]
        middle_blk_num = 6
        dec_blks = [2, 2, 2, 2]
        self.generator = HFRM(in_channel=img_channel, dim=dim, mid_blk_num=middle_blk_num,enc_blk_nums=enc_blks,dec_blk_nums=dec_blks).to(self.device).eval()
        self.generator.load_state_dict(torch.load("saved_models/raindrop/lastest.pth", map_location=self.device),strict=True)



        self.generator.requires_grad_(False)

        if config.data.global_attn == True:
            self.model = DiffusionUNet_Global(config)
        else:
            self.model = DiffusionUNet(config)
        self.model.to(self.device)
        print("Total_params_model_real: {}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
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

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        # pdb.set_trace()
        checkpoint = utils.logging.load_checkpoint(load_path, self.device)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def all_wavlet_dec(self,x):
        x_inp = x[:,:3,:,:]
        x_tar = x[:,3:,:,:]
        x_inp = self.wavelet_dec(x_inp)
        x_tar = self.wavelet_dec(x_tar)

        return torch.cat([x_inp,x_tar],dim=1)

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
                x_all = data_transform(x)

                

                if self.config.data.global_attn == True:
                    total = total.flatten(start_dim=0, end_dim=1) if total.ndim == 5 else total
                    total = total.to(self.device)
                    total = data_transform(total)

                if not self.config.data.wavelet_in_unet:
                    x_all = self.all_wavlet_dec(x_all) #bx96xhxw
                    if self.config.model.use_other_channels and not self.config.model.use_gt_in_train:
                        # x_cond_wdnet_wav = self.wavelet_dec(x[:, :3, :, :])
                        # x_res_wdnet_wav = self.generator(x_cond_wdnet_wav)
                        # x_output_wdnet = self.wavelet_rec(x_res_wdnet_wav) + x[:, :3, :, :]
                        with torch.no_grad():
                            x_output_wdnet = self.generator(x[:, :3, :, :].to(self.device))
                        x_output_wdnet_norm = data_transform(x_output_wdnet)
                        x_output_wdnet_wav = self.wavelet_dec(x_output_wdnet_norm)

                if self.config.model.use_other_channels:
                    if self.config.model.use_gt_in_train:
                        #x_for_pred = x_all
                        x_gt = x_all[:, x_all.shape[1]//2:]
                        x_for_pred = torch.cat([x_all[:, :x_all.shape[1] // 2 + self.config.model.pred_channels],
                                   x_gt[:, self.config.model.other_channels_begin:]], dim=1)
                    else:
                        x_for_pred = torch.cat([x_all[:, :x_all.shape[1]//2 + self.config.model.pred_channels],
                             x_output_wdnet_wav[:, self.config.model.other_channels_begin:]], dim=1)
                else:
                    x_for_pred = x_all[:,:x_all.shape[1]//2 + self.config.model.pred_channels,:,:]

                e = torch.randn_like(x_for_pred[:, x_all.shape[1]//2:x_all.shape[1]//2 + self.config.model.pred_channels, :, :])
                b = self.betas
                # pdb.set_trace()

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss, e_pred, x0_pred, mse_loss = noise_estimation_loss(self.model, x_for_pred, t, e, b, total,
                                                                        self.config.data.global_attn,inp_channels=x_all.shape[1]//2,
                                                                        pred_channels=self.config.model.pred_channels,use_other_channels=self.config.model.use_other_channels)
                e_pred_diff_list.append(e_pred.mean().item()-e.mean().item())

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, loss mean: {loss.item()/num_of_pixel}, mse loss mean: {mse_loss.item()/num_of_pixel}, data time: {data_time / (i+1)} \n")

                self.optimizer.zero_grad()
                if self.config.training.use_mse:
                    mse_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                self.ema_helper.update(self.model)
                data_start = time.time()

                if (dist.get_world_size()>1 and self.step % self.config.training.validation_freq == 0) or (dist.get_world_size()==1 and self.step % 10 == 0):
                    if dist.get_rank() == 0:
                        self.model.eval()
                        _, val_loader = DATASET.get_loaders(parse_patches=False, validation=self.args.test_set)
                        # self.sample_validation_patches(val_loader, self.step)
                        self.restore(val_loader, validation=self.args.test_set, r=self.args.grid_r, epoch=epoch)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    if dist.get_rank() == 0:
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_epoch' + str(epoch+1) + '_ddpm'))


    def sample_image(self, x_cond, x, x_other = None, last=True, patch_locs=None, patch_size=None,total=None,use_global=False,use_other=False):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            # xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
            #                                         corners=patch_locs, p_size=patch_size, total=total,
            #                                         use_global=use_global)
            xs = self.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size,
                                                    total=total,use_global=use_global,x_other=x_other,use_other=use_other)
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
                x_all = x.to(self.device)
                x_all = data_transform(x_all)

                if self.config.data.global_attn == True:
                    total = total.to(self.device)
                    total = data_transform(total)

                x_cond = x_all[:, :3, :, :]
                x_gt = x_all[:, 3:, :, :]

                if not self.config.data.wavelet_in_unet:
                    x_cond = self.wavelet_dec(x_cond)
                    x_gt = self.wavelet_dec(x_gt)
                    if self.config.model.use_other_channels:
                        # x_cond_wdnet_wav = self.wavelet_dec(x[:, :3, :, :].to(self.device))
                        # x_res_wdnet_wav = self.generator(x_cond_wdnet_wav)
                        # x_output_wdnet = self.wavelet_rec(x_res_wdnet_wav) + x[:, :3, :, :].to(self.device)
                        with torch.no_grad():
                            x_output_wdnet = self.generator(x[:, :3, :, :].to(self.device))
                        x_output_wdnet_norm = data_transform(x_output_wdnet)
                        x_output_wdnet_wav = self.wavelet_dec(x_output_wdnet_norm)

                if self.config.data.wavelet and not self.config.data.wavelet_in_unet and self.config.model.use_other_channels:
                        x_other = x_output_wdnet_wav[:,self.config.model.other_channels_begin:,:,:]
                else:
                    x_other = None

                x_output_list = self.diffusive_restoration(x_cond, x_other=x_other, r=r, total=total,last=False, use_global=self.config.data.global_attn,use_other=self.config.model.use_other_channels)
                #x_output = self.standard_output(x_output,x_cond)

                x_output = x_output_list[1][-5]
                if not self.config.data.wavelet_in_unet and self.config.model.pred_channels < self.config.model.in_channels:
                    x_output_hrgt_cat = torch.cat([x_output.to(self.device)[:, :self.config.model.pred_channels],
                                          x_gt[:, self.config.model.pred_channels:]], dim=1)
                    x_output = torch.cat([x_output.to(self.device)[:, :self.config.model.pred_channels],
                         x_output_wdnet_wav[:, self.config.model.pred_channels:]], dim=1)

                if not self.config.data.wavelet_in_unet:
                    x_output = self.wavelet_rec(x_output.to(self.device))
                    x_cond = self.wavelet_rec(x_cond.to(self.device))
                    x_output_hrgt_cat = self.wavelet_rec(x_output_hrgt_cat.to(self.device))
                    x_output_hrgt_cat = inverse_data_transform(x_output_hrgt_cat)

                x_output = inverse_data_transform(x_output)
                x_cond = inverse_data_transform(x_cond)

                gt = x[:, 3:, :, :]
                psnr = utils.torchPSNR(gt.to(self.device), x_output)
                print("psnr", psnr)

                all_samples.append(x_cond.cpu())
                all_samples.append(x_output_hrgt_cat.cpu())
                all_samples.append(x_output.cpu())
                all_samples.append(gt)
                if i==1:
                    break

            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=4)
            utils.logging.save_image(grid, os.path.join(image_folder, f"{y}_output" + "_epoch" + str(epoch) + ".png"))
            # utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))


    def diffusive_restoration(self, x_cond, x_other=None,r=None, last=True, total=None,use_global=False,use_other=False):
        if self.config.data.wavelet_in_unet:
            p_size = self.config.data.patch_size
        else:
            p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]

        x = torch.randn((x_cond.shape[0],self.config.model.pred_channels,x_cond.shape[2],x_cond.shape[3]), device=self.device)
        x_output = self.sample_image(x_cond, x, x_other=x_other, patch_locs=corners, last=last,patch_size=p_size,
                                     total=total,use_global=use_global,use_other=use_other)
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

    def generalized_steps_overlapping(self, x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True,
                                      total=None, x_other=None, use_global=False,use_other=False):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []

            #not use noise to start, not work
            if not self.config.data.begin_from_noise:
                a = (1 - b).cumprod(dim=0).index_select(0, torch.tensor(self.num_timesteps-1).to(self.device)).view(-1, 1, 1, 1)
                x = x_cond[:, :, :, :] * a.sqrt() + x * (1.0 - a).sqrt()

            xs = [x]

            x_grid_mask = torch.zeros_like(x, device=x.device)
            for (hi, wi) in corners:
                x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

            print("patch num :", len(corners))
            for i_t, j_t in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i_t).to(x.device)
                next_t = (torch.ones(n) * j_t).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to(x_cond.device)
                et_output = torch.zeros_like(x, device=x.device)
                counts = torch.zeros_like(x, device=x.device)

                if manual_batching:
                    manual_batching_size = 8
                    xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                    # x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                    x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                    if use_other:
                        x_other_patch = torch.cat([crop(x_other, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                    for i in range(0, len(corners), manual_batching_size):
                        if use_global == False:
                            x_cond_and_t_patch = torch.cat(
                                [x_cond_patch[i:i + manual_batching_size], xt_patch[i:i + manual_batching_size]], dim=1)
                            if use_other:
                                x_cond_and_t_patch = torch.cat(
                                    [x_cond_and_t_patch, x_other_patch[i:i + manual_batching_size]],dim=1)
                            outputs = model(x_cond_and_t_patch, t)
                        else:
                            x_cond_and_t_patch = torch.cat(
                                [x_cond_patch[i:i + manual_batching_size], xt_patch[i:i + manual_batching_size]], dim=1)
                            total_batch = total.repeat(x_cond_and_t_patch.shape[0], 1, 1, 1)
                            outputs = model(x_cond_and_t_patch, t, total_batch)
                        for idx, (hi, wi) in enumerate(corners[i:i + manual_batching_size]):
                            et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
                            counts[0, :, hi:hi + p_size, wi:wi + p_size] += 1
                else:
                    for (hi, wi) in corners:
                        xt_patch = crop(xt, hi, wi, p_size, p_size)
                        x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                        # x_cond_patch = data_transform(x_cond_patch)
                        et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(
                            torch.cat([x_cond_patch, xt_patch], dim=1), t)

                et = torch.div(et_output, x_grid_mask)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))

                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))
                print(f"t:{i_t} e pred:{et.mean().item()} e pred std:{et.std().item()} x0 pred:{x0_t.mean().item()} x next:{xt_next.mean().item()}")

        return xs, x0_preds
