import random
import numpy as np
import os
import cv2
import shutil
import wandb 
import math

import gdown

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR

import model.network.net as net
from model.network.glow import Glow
from model.utils.utils import IterLRScheduler, remove_prefix
from model.layers.activation_norm import calc_mean_std
from model.losses.tv_loss import TVLoss

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename + '.pth.tar')

def get_smooth(I, direction):
    weights = torch.tensor([[0., 0.], [-1., 1.]]).cuda()
    w_x = weights.view(1, 1, 2, 2)
    w_y = weights.t().view(1, 1, 2, 2)
    w = w_x if direction == 'x' else w_y
    return torch.abs(torch.nn.functional.conv2d(I, w, padding=1))

def avg_pool(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    grad_x = get_smooth(I_gray, 'x')
    grad_y = get_smooth(I_gray, 'y')
    return torch.mean(
        grad_x * torch.exp(-10 * avg_pool(R_gray, 'x')) +
        grad_y * torch.exp(-10 * avg_pool(R_gray, 'y'))
    )

def load_vgg_weights(cfg, vgg):
    vgg_drive_id = cfg.get('vgg_drive_id', '15CeUI3FMgSFGSUPAldAzGAWAhMNL5iIO')
    vgg_url = cfg.get('vgg_url')
    vgg_local_path = cfg.get('vgg')

    if not vgg_local_path:
        raise ValueError(
            "Missing VGG weights path. Set `vgg: ./vgg_normalised.pth` "
            "or another local path in the config."
        )

    if not os.path.exists(vgg_local_path):
        print(f"VGG model weights not found at {vgg_local_path}. Downloading...")
        try:
            if vgg_url:
                gdown.download(url=vgg_url, output=vgg_local_path, quiet=False, fuzzy=True)
            else:
                gdown.download(id=vgg_drive_id, output=vgg_local_path, quiet=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not download VGG weights automatically.\n"
                f"Expected local file: {vgg_local_path}\n"
                f"Google Drive id: {vgg_drive_id}\n"
                "Manual fix: download vgg_normalised.pth in a browser and place it at "
                f"{vgg_local_path}, or set `vgg` to the downloaded file path in the config. "
                "If you have a direct mirror, set `vgg_url` in the config."
            ) from exc

    print(f"Loading VGG weights from {vgg_local_path}")

    # Load weights into the existing model instance
    vgg.load_state_dict(torch.load(vgg_local_path))
    return vgg


class merge_model(nn.Module):
    def __init__(self, cfg):
        super(merge_model, self).__init__()
        self.glow = Glow(
            in_channel=3,
            n_flow=cfg['n_flow'],
            n_block=cfg['n_block'],
            affine=cfg['affine'],
            conv_lu=not cfg['no_lu'],
            alpha=not cfg['attention']==None
        )
        # self.refiner = Refiner(in_channels=3)

    def forward(self, content_images,style_code):
        z_c = self.glow(content_images, forward=True)
        stylized = self.glow(z_c, forward=False, style=style_code)

        # stylized = self.refiner(stylized)
        return stylized

    def freeze_flow_train_modulation(self):
        print("\n🔒 Freezing all parameters...")
        for param in self.glow.parameters():
            param.requires_grad = False

        print("🔓 Unfreezing modulation parameters...\n")
        for block_idx, block in enumerate(self.glow.blocks):
            for name, module in block.named_children():
                if name in ['modulate', 'channel_attn', 'spatial_attn']:
                    for param_name, param in module.named_parameters(recurse=True):
                        param.requires_grad = True
                        print(f"✅ Unfroze: glow.blocks[{block_idx}].{name}.{param_name}")

        print("\n🧾 Parameter Summary (after freeze/unfreeze):")
        for name, param in self.named_parameters():
            status = "✅ Trainable" if param.requires_grad else "❌ Frozen"
            print(f"{status:<12} | {name}")
        print()


class Trainer:
    def __init__(self, cfg, seed=0):
        set_random_seed(seed)
        self.cfg = cfg
        self.init = True

        # Merge model and optimizer
        self.model = merge_model(cfg).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.lr_scheduler_mode = cfg.get('lr_scheduler', 'iter_multistep')
        if self.lr_scheduler_mode == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get('total_epoch', 1),
                eta_min=cfg.get('lr_min', cfg['lr'] * 0.1),
            )
        elif self.lr_scheduler_mode == 'cosine_warmup':
            total_epoch = max(1, cfg.get('total_epoch', 1))
            warmup_epoch = max(0, cfg.get('warmup_epoch', 0))
            min_lr_ratio = cfg.get('lr_min', cfg['lr'] * 0.1) / cfg['lr']

            def lr_lambda(epoch):
                if warmup_epoch > 0 and epoch < warmup_epoch:
                    return float(epoch + 1) / float(warmup_epoch)
                progress = (epoch - warmup_epoch) / max(1, total_epoch - warmup_epoch)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.lr_scheduler_mode == 'epoch_multistep':
            lr_mults = cfg.get('lr_mults', 0.5)
            gamma = lr_mults[0] if isinstance(lr_mults, list) else lr_mults
            self.lr_scheduler = MultiStepLR(
                self.optimizer,
                milestones=cfg.get('lr_epoch_steps', cfg.get('lr_steps', [])),
                gamma=gamma,
            )
        else:
            self.lr_scheduler = IterLRScheduler(
                self.optimizer,
                cfg.get('lr_steps', []),
                cfg.get('lr_mults', 0.5),
                last_iter=cfg.get('last_iter', 0)
            )

        # Content encoder (VGG) & Style encoder
        # Load pretrained VGG from torchvision
        vgg = net.vgg
        vgg = load_vgg_weights(cfg, vgg)

        self.encoder = net.Net(vgg).cuda()

        self.tv_loss = TVLoss().cuda()

        # Logging paths
        self.log_path = os.path.join(
            cfg['output'],
            f"{cfg['job_name']}_{int(cfg.get('keep_ratio',1)*100)}_{cfg['n_flow']}_{cfg['n_block']}"
        )
        self.model_log_path = os.path.join(self.log_path, 'model_save')
        self.img_log_path = os.path.join(self.log_path, 'img_save')
        self.img_test_path = os.path.join(self.log_path, 'test_save')
        self.img_att_path = os.path.join(self.log_path, 'att_save')
        for path in [self.model_log_path, self.img_log_path, self.img_test_path, self.img_att_path]:
            os.makedirs(path, exist_ok=True)

        shutil.copy(cfg['cfg_path'],os.path.join(self.log_path, os.path.basename(cfg['cfg_path'])))

    def load_model(self, checkpoint_path, flow_only=False):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        full_state = remove_prefix(ckpt['state_dict'], 'module.')

        if flow_only:
            print(f"\nLoading FLOW-only weights from: {checkpoint_path}")
            # Filter out only flow-relevant keys
            filtered_state = {
                k: v for k, v in full_state.items()
                if any([
                    '.flows.' in k,
                    '.actnorm' in k,
                    '.invconv' in k,
                    '.coupling' in k
                ])
            }
            # Load into model with missing keys allowed
            missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)

            print("Loaded FLOW weights only.")
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}\n")
        else:
            # Load entire model
            self.model.load_state_dict(full_state)
            print(f"\nLoaded full model weights from: {checkpoint_path}\n")

        # Load optimizer only if full model is restored
        if not flow_only and 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            print("Optimizer state loaded\n")

    def save_model(self, filename, epoch=None, metrics=None):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if epoch is not None:
            state['epoch'] = epoch
        if metrics is not None:
            state['metrics'] = metrics
        if hasattr(self.lr_scheduler, 'state_dict'):
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        save_checkpoint(state, os.path.join(self.model_log_path, filename))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step_lr_scheduler(self):
        if self.lr_scheduler_mode != 'iter_multistep':
            self.lr_scheduler.step()


    def train(self, batch_id, content_imgs, style_imgs, epoch):
        content = content_imgs.cuda()
        style = style_imgs.cuda()

        style_code = self.encoder.cat_tensor(style)
        stylized = self.model(content,style_code)
        stylized = torch.clamp(stylized, 0, 1)

        # Smoothness loss
        loss_smooth = self.tv_loss(stylized)

        loss_c, loss_s, loss_r, loss_p = self.encoder(content, style, stylized, keep_ratio=self.cfg.get('keep_ratio', 0.5))
        
        # stage 1 loss
        loss_scm = torch.zeros_like(loss_r)

        loss_c = loss_c.mean() * self.cfg.get('content_weight', 1.0)
        loss_s = loss_s.mean() * self.cfg.get('style_weight', 1e-4)
        loss_r = loss_r.mean() * self.cfg.get('recon_weight', 1.0)
        loss_p = loss_p.mean() * self.cfg.get('p_weight', 1.0)
        loss_smooth = loss_smooth * self.cfg.get('smooth_weight', 1.0)
        loss_scm = loss_scm * self.cfg.get('scm_weight', 0.05)

        total_loss = loss_c + loss_s + loss_r + loss_p + loss_smooth + loss_scm
        # total_loss = loss_p

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler_mode == 'iter_multistep':
            self.lr_scheduler.step()

        # Logging
        if batch_id % self.cfg.get('log_freq', 100) == 0:
            fname = f"{epoch}_{batch_id}.jpg"
            out = torch.cat([
                content[-1:], stylized[-1:], style[-1:]
            ], dim=3)
            save_image(out.cpu(), os.path.join(self.img_log_path, fname))

            if self.cfg['wandb'] is not None:
                wandb.log({
                    "train/image_log": wandb.Image(out.cpu(), caption=f"Epoch {epoch}, Batch {batch_id}"),
                },step=epoch)

            print('saved at ',os.path.join(self.img_log_path, fname))

        return [
            total_loss.item(),
            loss_c.item(),
            loss_s.item(),
            loss_r.item(),
            loss_p.item(),
            loss_smooth.item(),
            loss_scm.item()
        ]

    @torch.no_grad()
    def test(self, epoch, batch_id, content_imgs, style_imgs, gt_imgs, len_data=1000, suffix=''):
        content = content_imgs.cuda()
        style = style_imgs.cuda()
        gt = gt_imgs.cuda()

        style_code = self.encoder.cat_tensor(style)
        stylized = self.model(content,style_code)
        stylized = torch.clamp(stylized, 0, 1)

        total_psnr = 0
        total_ssim = 0
        count = 0

        # Save attention overlays (only if attention enabled)
        if self.cfg['attention'] != None:
          for block_idx, block in enumerate(self.model.glow.blocks):
              if hasattr(block, 'last_attention'):
                  attn_maps = block.last_attention  # [B, 1, H, W]

                  images_to_log = []

                  for img_idx in range(min(attn_maps.shape[0],5)):
                      attn_overlay =logger.overlay_attention_on_image(
                          attn_maps[img_idx], content[img_idx]
                      )
                      save_name = f"epoch{epoch}_batch{batch_id}_idx{img_idx}_attn_block{block_idx}.jpg"
                      save_path = os.path.join(self.img_att_path, save_name)

                      cv2.imwrite(save_path, cv2.cvtColor(attn_overlay, cv2.COLOR_RGB2BGR))

                      if self.cfg['wandb'] is not None :
                            images_to_log.append(wandb.Image(
                                attn_overlay,
                                caption=f"Epoch {epoch}, Batch {batch_id}, Img {img_idx}, Block {block_idx}"
                                ))

                  if self.cfg['wandb'] is not None:
                      wandb.log({
                          f"attention/block_{block_idx}": images_to_log,
                      },step=epoch)

        images_to_log = []
        for i in range(min(content.size(0),5)):
            output_img = stylized[i].detach().cpu().permute(1, 2, 0).numpy()
            gt_img = gt[i].detach().cpu().permute(1, 2, 0).numpy()

            # Convert to float32
            output_img = output_img.astype(np.float32)
            gt_img = gt_img.astype(np.float32)

            # Compute PSNR & SSIM
            psnr = compare_psnr(gt_img, output_img, data_range=1.0)
            ssim = compare_ssim(gt_img, output_img, multichannel=True, data_range=1.0, channel_axis=-1)

            total_psnr += psnr
            total_ssim += ssim
            count += 1
        
            out = torch.cat([
                content[i:i+1], style[i:i+1], stylized[i:i+1], gt[i:i+1]
            ], dim=3)
            name = f"{suffix}_epoch{epoch}_batch{batch_id}_idx{i}.jpg"
            save_image(out.cpu(), os.path.join(self.img_test_path, name))

            if self.cfg.get('wandb', None) is not None:
                images_to_log.append(wandb.Image(
                    out.cpu(),
                    caption=f"{suffix} | Epoch {epoch}, Batch {batch_id}, Index {i}, PSNR: {psnr:.2f}, SSIM: {ssim:.3f}"
                ))

        avg_psnr = total_psnr / count if count > 0 else 0
        avg_ssim = total_ssim / count if count > 0 else 0
        print(f"[Test] Epoch {epoch} — PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        if self.cfg.get('wandb', None) is not None:
            wandb.log({
                "test/images": images_to_log,
            },step=epoch)

        return avg_psnr,avg_ssim
