import os
import cv2
import numpy as np
import wandb

from tqdm import tqdm

from torch.utils.data import DataLoader

from model.utils.utils import get_config
from model.trainers.trainer import Trainer
from dataset.dataset_ImagePair import ImagePairDataset
from logger import TrainingLogger

code_name = 'Dataset_OHAZE'
root_dir = f'./dataset/Dataset_OHAZE'
cfg_path = f'./config/Dataset_OHAZE_1_6.yaml'

wandb_project = 'TESIS_CLEAN'

args = get_config(cfg_path)
args['cfg_path'] = cfg_path

resume = args['resume']
img_size = (args['img_h'],args['img_w']) #OpenCV
batch_size = args['batch_size']
wandb_key = args['wandb']

if wandb_key is not None:
    print(f'\n\nSetting up wandb\n\n')
    wandb.login(key=wandb_key)

    wandb_run = wandb.init(
        project=wandb_project,
        entity="alfin-nurhalim",
        name=code_name,
        config=args,
    )

train_dataset_mixed = ImagePairDataset(root_dir=root_dir, 
                                        phase='train',
                                        augment=args['aug'],
                                        stage=args['stage'],
                                        img_size=img_size)

test_dataset_mixed = ImagePairDataset(root_dir=root_dir, 
                                        phase='test',
                                        augment=False, 
                                        stage=args['stage'],
                                        img_size=img_size)

print(f'\n\nTrain Dataset : {len(train_dataset_mixed)}')
print(f'Test Dataset : {len(test_dataset_mixed)}\n\n')
train_loader_mixed = DataLoader(train_dataset_mixed, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader_mixed = DataLoader(test_dataset_mixed, batch_size=batch_size, shuffle=False, num_workers=2)

print('attention:',args['attention'])
print('FLOW :',args['n_flow'], 'BLOCK: ',args['n_block'])
print('epoch: ',args['total_epoch'])
print('\n\n')

trainer = Trainer(args)

if resume is not None:
  trainer.load_model(resume,flow_only=False)

# if args['stage']==2:
#   trainer.model.freeze_flow_train_modulation()

training_logger = TrainingLogger(trainer.log_path)

for epoch in range(args['total_epoch']):
    progress_bar = tqdm(enumerate(train_loader_mixed),
                        total=len(train_loader_mixed),
                        desc=f"Epoch {epoch}")

    last_loss = None
    for batch_id, (source_image, gt_image) in progress_bar:
        loss_list = trainer.train(batch_id, source_image, gt_image, epoch)
        
        loss,loss_c,loss_s,loss_r,loss_p,loss_smooth,loss_scm = loss_list if loss_list is not None else [0,0,0,0,0]

        last_loss = loss  # Update last_loss continuously

        if loss_list is not None and wandb_key is not None:
            wandb.log({
                'train/loss_total': loss,
                'train/loss_content': loss_c,
                'train/loss_style': loss_s,
                'train/loss_restoration': loss_r,
                'train/loss_pixel': loss_p,
                'train/loss_smooth': loss_smooth,
                'train/loss_scm': loss_scm,
            },step=epoch)

        progress_bar.set_postfix({'Loss': f'{loss:.4f}',
                                  'Loss_c': f'{loss_c:.4f}',
                                  'Loss_s': f'{loss_s:.4f}',
                                  'Loss_r': f'{loss_r:.4f}',
                                  'loss_p': f'{loss_p:.4f}',
                                  'Loss_smooth': f'{loss_smooth:.4f}',
                                  'loss_scm': f'{loss_scm:.4f}'})

    if last_loss is not None:
        training_logger.log_epoch_loss(epoch, last_loss)

    print(f"\nEpoch: {epoch} - Last Batch Loss: {last_loss:.4f}")

    if epoch%10 == 0:
      if 'train' in args['eval']:
        print('\n\nTesting on train set ......')
        avg_psnr = []
        avg_ssim = []
        test_bar = tqdm(enumerate(train_loader_mixed),
                          total=len(train_loader_mixed),
                          desc=f"Epoch {epoch}")
        for batch_id, (source_image, gt_image) in test_bar:
            psnr,ssim = trainer.test(epoch, batch_id, source_image, gt_image)
            avg_psnr.append(psnr)
            avg_ssim.append(ssim)

        avg_psnr = sum(avg_psnr)/len(avg_psnr)
        avg_ssim = sum(avg_ssim)/len(avg_ssim)

        print(f"\nEpoch: {epoch} - TRAINING Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f} ")
        if wandb_key is not None:
            wandb.log({
                'train/psnr': avg_psnr,
                'train/ssim': avg_ssim
            },step=epoch)

      if 'test' in args['eval']:
          print('\n\nTesting on test set ......')
          avg_psnr = []
          avg_ssim = []
          test_bar = tqdm(enumerate(test_loader_mixed),
                            total=len(test_loader_mixed),
                            desc=f"Epoch {epoch}")
          for batch_id, (source_image, gt_image) in test_bar:
              psnr,ssim = trainer.test(epoch, batch_id, source_image, gt_image, len_data=len(test_loader_mixed))
              avg_psnr.append(psnr)
              avg_ssim.append(ssim)

          avg_psnr = sum(avg_psnr)/len(avg_psnr)
          avg_ssim = sum(avg_ssim)/len(avg_ssim)

          print(f"\nEpoch: {epoch} - TEST Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f} ")
          if wandb_key is not None:
              wandb.log({
                  'test/psnr': avg_psnr,
                  'test/ssim': avg_ssim
              },step=epoch)

    # using wandb
    # training_logger.update_graph()

    print('\n\n')

if wandb_key is not None:
    wandb_run.finish()
