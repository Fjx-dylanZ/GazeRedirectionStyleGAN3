import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import tqdm
from collections import OrderedDict
import os
import dataset
from torch.utils.tensorboard import SummaryWriter
import click
from models.redirect_net import RedirectNet, RedirectNetLoss
import numpy as np
import PIL
# --------------------------------------------
@click.command()

# Required parameters
@click.option('--batch', help='Batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--steps', help='Number of steps', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--checkpoint_output_dir', help='Checkpoint output directory', metavar='DIR', type=str, required=True)
@click.option('--h5_dataset_path', help='HDF5 dataset path', metavar='DIR', type=str, required=True)

# Optional parameters
@click.option('--output_steps', help="Number of steps between each checkpoint", metavar='INT', type=click.IntRange(min=1), default=5000, show_default=True)
@click.option('--resume_checkpoint', help='Resume checkpoint path', metavar='DIR', type=str, default=None, show_default=True)
@click.option('--validate', help='Validate', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--validate_steps', help='Number of steps between each validation', metavar='INT', type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--validate_num_samples', help='Number of samples to validate', metavar='INT', type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--log_steps', help='Number of steps between each log', metavar='INT', type=click.IntRange(min=1), default=50, show_default=True)
# python train.py --batch 6 --steps 100000 --checkpoint_output_dir ./training_log/ --h5_dataset_path /root/data/dataset_prep/faze_preprocess/outputs_sted/GazeCapture.h5 --output_steps 2000 --validate True --validate_steps 500
def main(**kwargs):
    config_dict = OrderedDict(kwargs)
    batch_size = config_dict['batch']
    steps = config_dict['steps']
    checkpoint_output_dir = config_dict['checkpoint_output_dir']
    assert os.path.isdir(checkpoint_output_dir), "checkpoint_output_dir must be a directory"
    hdf_file_path = config_dict['h5_dataset_path']
    output_steps = config_dict['output_steps']
    resume_checkpoint_path = config_dict['resume_checkpoint']
    validate = config_dict['validate']
    seed = config_dict['seed']
    validate_steps = config_dict['validate_steps']
    validate_num_samples = config_dict['validate_num_samples']
    log_steps = config_dict['log_steps']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter()
    gazeCaptureDataset = dataset.HDFDataset(
        hdf_file_path=hdf_file_path,
        get_2nd_sample=True # label
    )

    # set random seed
    torch.manual_seed(seed)

    # random split dataset
    train_size = int(0.9 * len(gazeCaptureDataset))
    test_size = len(gazeCaptureDataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(gazeCaptureDataset, [train_size, test_size])

    # initialize model
    net = RedirectNet().to(device)
    net_loss = RedirectNetLoss(
        lambda_gaze=2.5, # start small
        lambda_head_pose=2.5, # start small
        lambda_direct_latent=0.02,
        lambda_image_reconstruction=80,
        lambda_perceptual=70,
        lambda_identity=1.0
    ).to(device)

    # Initialize optimizer
    optimizer = Adam(net.parameters(), lr=1e-5)

    # initialize checkpoint
    checkpoint = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': 0
    }

    # load checkpoint
    if resume_checkpoint_path is not None:
        assert os.path.isfile(resume_checkpoint_path), "resume_checkpoint_path must be a file"
        checkpoint = torch.load(resume_checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
    else:
        start_step = 0

    def get_random_dataloader(dataset, num_samples=None):
        if num_samples is None:
            num_samples = len(dataset)
        else:
            assert num_samples <= len(dataset), "num_samples must be less than or equal to dataset length"
        indices = torch.randperm(len(dataset)).tolist()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:num_samples])
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=8
        )

    # train
    train_iter = iter(get_random_dataloader(train_dataset))
    pbar = tqdm.tqdm(range(start_step, steps), total=steps, initial=start_step)

    train_loss_dict = OrderedDict() # per logging interval

    for step in pbar:

        # test
        if validate and step % validate_steps == 0 and step > 0:
            net.eval()
            with torch.no_grad():
                test_dataloader = get_random_dataloader(test_dataset, num_samples=validate_num_samples)
                val_loss_dict = OrderedDict()
                val_imgs_source = []
                val_imgs_pred = []
                val_imgs_target = []
                for i, test_batch in tqdm.tqdm(enumerate(test_dataloader)):
                    imgs_a = test_batch['image_a'].to(device)
                    imgs_b = test_batch['image_b'].to(device)
                    output_dict = net(imgs_a, imgs_b)
                    loss, loss_dict = net_loss(output_dict)
                    # fill val_loss_dict
                    for key, val in loss_dict.items():
                        if key not in val_loss_dict:
                            val_loss_dict[key] = []
                        try:
                            val_loss_dict[key].append(val.item())
                        except:
                            assert key == 'id_improve'
                            val_loss_dict[key].append(val)
                    if 'total_loss' not in val_loss_dict:
                        val_loss_dict['total_loss'] = []
                    val_loss_dict['total_loss'].append(loss.item())
                    # fill val_imgs_pred and val_imgs_target
                    if i % 40 == 0:
                        val_imgs_source.append(output_dict['id_reference'].detach().cpu())
                        val_imgs_pred.append(output_dict['image_pred'].detach().cpu())
                        val_imgs_target.append(output_dict['image_target'].detach().cpu())

                for key, val in val_loss_dict.items():
                    val_loss_dict[key] = sum(val) / len(val)
                writer.add_scalars('val_loss', val_loss_dict, step)
                # write images
                img_result = torch.cat(val_imgs_source + val_imgs_pred + val_imgs_target, dim=0)
                img_result = return_image(img_result, 3, batch_size * len(val_imgs_source), 256, 256)
                writer.add_images('val_images', img_result, step, dataformats='HWC')

        net.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(get_random_dataloader(train_dataset))
            batch = next(train_iter)
        
        imgs_a = batch['image_a'].to(device)
        imgs_b = batch['image_b'].to(device)

        output_dict = net(imgs_a, imgs_b)
        loss, loss_dict = net_loss(output_dict)

        # fill train_loss_dict
        for key, val in loss_dict.items():
            if key not in train_loss_dict:
                train_loss_dict[key] = []
            try:
                train_loss_dict[key].append(val.item())
            except:
                # id_improve is an eval metric, not a loss
                assert key == 'id_improve'
                train_loss_dict[key].append(val)
        if 'total_loss' not in train_loss_dict:
            train_loss_dict['total_loss'] = []
        train_loss_dict['total_loss'].append(loss.item())
        
        # write images
        if step % (log_steps * 2) == 0 and step > 0:
            img_result = torch.cat([output_dict['id_reference'].detach().cpu(),
                                    output_dict['image_pred'].detach().cpu(), 
                                    output_dict['image_target'].detach().cpu()
                                    ], dim=0)
            img_result = return_image(img_result, 3, batch_size, 256, 256)
            writer.add_images('train_images', img_result, step, dataformats='HWC')

        # write losses
        if step % log_steps == 0 and step > 0:
            for key, val in train_loss_dict.items():
                train_loss_dict[key] = sum(val) / len(val)
            writer.add_scalars('train_loss', train_loss_dict, step)
            train_loss_dict = OrderedDict()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save checkpoint
        if step % output_steps == 0 and step > 0:
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }
            torch.save(checkpoint, os.path.join(checkpoint_output_dir, 'checkpoint_%06d.pth' % step))

        # break if finish training
        if step >= steps:
            break

    # save final checkpoint
    checkpoint = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, os.path.join(checkpoint_output_dir, 'checkpoint_%06d.pth' % step))
        
def return_image(images, gh, gw, H, W):
    np_imgs = []
    for i, image in enumerate(images):
        image = images[i][None,:,:]
        image = (image.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8).cpu().numpy()
        np_imgs.append(np.asarray(PIL.Image.fromarray(image[0], 'RGB').resize((H,W),PIL.Image.LANCZOS)))
    np_imgs = np.stack(np_imgs)
    np_imgs = np_imgs.reshape(gh,gw,H,W,3)
    np_imgs = np_imgs.transpose(0,2,1,3,4)
    np_imgs = np_imgs.reshape(gh*H, gw*W, 3)
    return np_imgs

if __name__ == '__main__':
    main()