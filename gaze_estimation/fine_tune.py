import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataset
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import gaze_estimation_net
import click
from collections import OrderedDict
import tqdm
import itertools
import os
# tensorboard stuff
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------
# python fine_tune.py --batch 120 --base_checkpoint /root/data/gaze_estimation/official.pth.tar --steps 100000 --checkpoint_output_dir . --h5_dataset_path /root/data/dataset_prep/faze_preprocess/outputs_sted/GazeCapture.h5
@click.command()

# Required parameters
@click.option('--batch', help='Batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--base_checkpoint', help='Base checkpoint path', metavar='DIR', type=str, required=True)
@click.option('--steps', help='Number of steps', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--checkpoint_output_dir', help='Checkpoint output directory', metavar='DIR', type=str, required=True)
@click.option('--h5_dataset_path', help='HDF5 dataset path', metavar='DIR', type=str, required=True)

# Optional parameters
@click.option('--output_steps', help="Number of steps between each checkpoint", metavar='INT', type=click.IntRange(min=1), default=10000, show_default=True)
@click.option('--finetune_checkpoint', help='Finetune checkpoint path', metavar='DIR', type=str, default=None, show_default=True)
@click.option('--freeze_vgg', help='Freeze VGG', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--validate', help='Validate', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)

def main(**kwargs):

    conifg_dict = OrderedDict(kwargs)

    hdf_file_path = conifg_dict['h5_dataset_path']
    batch_size = conifg_dict['batch']
    base_checkpoint_path = conifg_dict['base_checkpoint']
    checkpoint_output_dir = conifg_dict['checkpoint_output_dir']
    assert os.path.isdir(checkpoint_output_dir), "checkpoint_output_dir must be a directory"
    steps = conifg_dict['steps']
    output_steps = conifg_dict['output_steps']
    finetune_checkpoint_path = conifg_dict['finetune_checkpoint']
    freeze_vgg = conifg_dict['freeze_vgg']
    validate = conifg_dict['validate']
    seed = conifg_dict['seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter()
    gazeCaptureDataset = dataset.HDFDataset(
        hdf_file_path=hdf_file_path,
    )

    # check finetune checkpoint
    if finetune_checkpoint_path is not None:
        assert os.path.isfile(finetune_checkpoint_path), "finetune_checkpoint_path must be a file"
        ckpt_name = os.path.basename(finetune_checkpoint_path)
        # at_step_:07d_vggFreezeBool.pth.tar
        resume_step = int(ckpt_name.split('_')[2])
        assert resume_step < steps, "resume_step must be less than steps"
        del ckpt_name
    else:
        resume_step = 0

    if not validate:
        gazeCaptureDataLoader = DataLoader(
            dataset=gazeCaptureDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
    else:
        # split dataset into train and val
        train_size = int(0.9 * len(gazeCaptureDataset))
        val_size = len(gazeCaptureDataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(gazeCaptureDataset, [train_size, val_size])
        gazeCaptureDataLoader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )
        valDataLoader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )

    net = gaze_estimation_net.VGG_Gaze_Estimator()
    net.to(device)
    
    # load base checkpoint (pretrained by STED)
    checkpoint = torch.load(base_checkpoint_path)
    net.load_state_dict(checkpoint)

    if finetune_checkpoint_path is not None:
        checkpoint = torch.load(finetune_checkpoint_path)
        net.load_state_dict(checkpoint)

    for param in net.parameters():
        param.requires_grad = True

    # freeze vgg
    if freeze_vgg:
        for param in net.vgg16.parameters():
            param.requires_grad = False

    # train
    lr = 1e-4 if freeze_vgg else 1e-5 # more aggressive lr if vgg is frozen
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.95))

    gazeLoss_per_logging = 0
    headLoss_per_logging = 0
    totalLoss_per_logging = 0
    logging_freq = 50
    validate_freq = 5000
    train_iter = iter(gazeCaptureDataLoader)
    pbar = tqdm.tqdm(range(resume_step, steps), total=steps, initial=resume_step)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(gazeCaptureDataLoader)
            batch = next(train_iter)
        imgs = batch['image_a'].to(device)
        gaze = batch['gaze_a'].to(device)
        head = batch['head_a'].to(device)

        # validate
        if validate and step % validate_freq == 0 and step != 0:
            net.eval()
            with torch.no_grad():
                gaze_val_loss = 0
                head_val_loss = 0
                total_val_loss = 0
                for val_batch in tqdm.tqdm(valDataLoader):
                    imgs = val_batch['image_a'].to(device)
                    gaze = val_batch['gaze_a'].to(device)
                    head = val_batch['head_a'].to(device)

                    gaze_hat, head_hat, _ = net(imgs, feature_out_layers=None)
                    gaze_loss = gaze_angular_loss(gaze, gaze_hat)
                    head_loss = gaze_angular_loss(head, head_hat)
                    loss = gaze_loss + head_loss

                    gaze_val_loss += gaze_loss.item()
                    head_val_loss += head_loss.item()
                    total_val_loss += loss.item()

                gaze_val_loss /= len(valDataLoader)
                head_val_loss /= len(valDataLoader)
                total_val_loss /= len(valDataLoader)

                writer.add_scalar('val_gaze_loss', gaze_val_loss, step)
                writer.add_scalar('val_head_loss', head_val_loss, step)
                writer.add_scalar('val_loss_total', total_val_loss, step)

        # train
        net.train()
        gaze_hat, head_hat, _ = net(imgs, feature_out_layers=None)

        # loss
        gaze_loss = gaze_angular_loss(gaze, gaze_hat)
        head_loss = gaze_angular_loss(head, head_hat)
        loss = gaze_loss + head_loss

        # logging
        gazeLoss_per_logging += gaze_loss.item()
        headLoss_per_logging += head_loss.item()
        totalLoss_per_logging += loss.item()

        if step % logging_freq == 0: # log every 50 steps
            if step == 0: continue
            writer.add_scalar('gaze_loss', gazeLoss_per_logging / logging_freq, step)
            writer.add_scalar('head_loss', headLoss_per_logging / logging_freq, step)
            writer.add_scalar('loss_total', totalLoss_per_logging / logging_freq, step)
            gazeLoss_per_logging = 0
            headLoss_per_logging = 0
            totalLoss_per_logging = 0
        pbar.set_postfix({'loss': loss.item()})
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % output_steps == 0:
            if step == 0: continue
            torch.save(net.state_dict(), f'{checkpoint_output_dir}/at_step_{step:07d}_vggFreeze{freeze_vgg}.pth.tar')
            print(f'Checkpoint saved at step {step}')

        if step == steps:
            break


    torch.save(net.state_dict(), f'{checkpoint_output_dir}/at_step_{step:07d}_vggFreeze{freeze_vgg}.pth.tar')

def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)

def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def gaze_angular_loss(y, y_hat, reduction='mean'):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        print("assuming reduction is None")
    return loss


if __name__ == '__main__':
    main()