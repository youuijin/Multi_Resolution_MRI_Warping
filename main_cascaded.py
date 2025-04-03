import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os, wandb
from utils.model import VoxelMorphUNet, UNet_2input, R2Net_UNet

import argparse

from torch.utils.tensorboard import SummaryWriter

from utils.utils import *
from utils.loss import TrainLoss
from utils.dataset import set_dataloader

wandb.login(key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")

# Training setup
def train_model(image_paths, template_path, loss='NCC', reg='l2', epochs=100, batch_size=1, lr=1e-4, alpha=0.5, alp_sca=1.0, res_level=3, start_epoch=0, saved_path=None):
    log_name = f'ftrain_res{res_level}_loss{loss}({reg}_alpha{alpha})_lr{lr}'
    if reg is None:
        log_name = f'ftrain_res{res_level}_loss{loss}_lr{lr}'
    else:
        if alp_sca!=1.0:
            log_name = f'ftrain_res{res_level}_loss{loss}({reg}_alpha{alpha}_sca{alp_sca})_lr{lr}'
    print('Training:', log_name)
    os.makedirs(f'saved_images/{log_name}', exist_ok=True)
    writer = SummaryWriter(f'./logs/{log_name}')
     # --- WandB Init ---
    wandb.init(
        project="brain-warping",
        name=log_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "reg": reg,
            "alpha": alpha,
            "alp_sca": alp_sca,
            "res_level": res_level
        }
    )

    train_loader, val_loader, save_loader = set_dataloader(image_paths, template_path, batch_size)
    
    if res_level == 1:
        res_k = [1]
    elif res_level == 2:
        res_k = [2, 1]
    elif res_level == 3:
        res_k = [4, 2, 1]
    elif res_level == 5:
        res_k = [16, 8, 4, 2, 1]
    else:
        raise ValueError('res level must be in [1, 2, 3, 5]')

    model = R2Net_UNet()
    
    if start_epoch>0:
        model.load_state_dict(torch.load(saved_path, weights_only=True))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TrainLoss(loss, reg, alpha, alp_sca)

    best_loss = 100000
    cnt = 0

    for epoch in range(epochs):
        if epoch<start_epoch:
            continue
        model.train()
        total_loss = np.array([0. for _ in range(len(res_k))])
        similar_loss = np.array([0. for _ in range(len(res_k))])
        smooth_loss = np.array([0. for _ in range(len(res_k))])
        for (img, template, _, _, _) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
            tot_displace_field = None
            for idx, resolution in enumerate(res_k):
                # downsampling image
                target_spacing = (1.0*resolution, 1.0*resolution, 1.0*resolution)
                cur_img = resample_pytorch_5d(img, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)
                cur_template = resample_pytorch_5d(template, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)

                # upsampling displace field
                if tot_displace_field is not None:
                    tot_displace_field = upsample_deformation(tot_displace_field, cur_img.shape[2:], tot_displace_field.shape[2:])

                    # apply previous displace field
                    P_cur_img = apply_deformation_using_disp(cur_img, tot_displace_field)
                else:
                    P_cur_img = cur_img

                stacked_input = torch.stack((P_cur_img.squeeze(1), cur_template.squeeze(1)), dim=1)

                # get current deformation field
                displace_field = model(stacked_input)
                deformed_cur_img = apply_deformation_using_disp(P_cur_img, displace_field)

                # Edit: using composed deformation field #
                composed_displace_field = compose_displace(tot_displace_field, displace_field) # if None, return

                # calculate loss and train model
                loss, diff_loss, smoo_loss = criterion(deformed_cur_img, cur_template, composed_displace_field, idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss[idx] += loss.item()
                similar_loss[idx] += diff_loss
                smooth_loss[idx] += smoo_loss

                # save deformation field
                new_displace_field = model(stacked_input).clone().detach()
                tot_displace_field = compose_displace(tot_displace_field, new_displace_field)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss.mean() / len(train_loader)}")
        writer.add_scalar('train/loss', total_loss.mean(), epoch)
        writer.add_scalar('train/loss_similar', similar_loss.mean(), epoch)
        writer.add_scalar('train/loss_smooth', smooth_loss.mean(), epoch)
        for idx, resolution in enumerate(res_k):
            writer.add_scalar(f'train/loss_1-{resolution}', total_loss[idx], epoch)
            writer.add_scalar(f'train/loss_similar_1-{resolution}', similar_loss[idx], epoch)
            writer.add_scalar(f'train/loss_smooth_1-{resolution}', smooth_loss[idx], epoch)

        # === wandb Logging (Train) ===
        wandb.log({
            "Train/Total_Loss": total_loss.mean() / len(train_loader),
            "Train/Similarity_Loss": similar_loss.mean(),
            "Train/Regularizer_Loss": smooth_loss.mean(),
            "Epoch": epoch
        })
        for idx, resolution in enumerate(res_k):
            wandb.log({
                f"Train/Total_Loss_res{resolution}": total_loss[idx],
                f"Train/Similarity_Loss_res{resolution}": similar_loss[idx],
                f"Train/Regularizer_Loss_res{resolution}": smooth_loss[idx],
                "Epoch": epoch
            })

        if epoch%5 == 0:
            model.eval()
            total_loss = np.array([0. for _ in range(len(res_k))])
            similar_loss = np.array([0. for _ in range(len(res_k))])
            smooth_loss = np.array([0. for _ in range(len(res_k))])
            with torch.no_grad():
                for (img, template, _, _, _) in val_loader:
                    img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
                    tot_displace_field = None
                    for idx, resolution in enumerate(res_k):
                        # downsampling image
                        target_spacing = (1.0*resolution, 1.0*resolution, 1.0*resolution)
                        cur_img = resample_pytorch_5d(img, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)
                        cur_template = resample_pytorch_5d(template, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)

                        # upsampling deformation field
                        if tot_displace_field is not None:
                            tot_displace_field = upsample_deformation(tot_displace_field, cur_img.shape[2:], tot_displace_field.shape[2:])

                            # apply previous deformation field
                            P_cur_img = apply_deformation_using_disp(cur_img, tot_displace_field)
                        else:
                            P_cur_img = cur_img

                        stacked_input = torch.stack((P_cur_img.squeeze(1), cur_template.squeeze(1)), dim=1)

                        # get current deformation field
                        displace_field = model(stacked_input)
                        deformed_cur_img = apply_deformation_using_disp(P_cur_img, displace_field)

                        # Edit: using composed deformation field #
                        composed_displace_field = compose_displace(tot_displace_field, displace_field) # if None, return

                        # calculate loss and train model
                        # loss = criterion(deformed_cur_img, cur_template, deformation_field)
                        loss, diff_loss, smoo_loss = criterion(deformed_cur_img, cur_template, composed_displace_field, idx)
                        total_loss[idx] += loss.item()
                        similar_loss[idx] += diff_loss
                        smooth_loss[idx] += smoo_loss

                        # save deformation field
                        new_displace_field = model(stacked_input).clone().detach()
                        tot_displace_field = compose_displace(tot_displace_field, new_displace_field)


                writer.add_scalar('val/loss', total_loss.mean(), epoch)
                writer.add_scalar('val/loss_similar', similar_loss.mean(), epoch)
                writer.add_scalar('val/loss_smooth', smooth_loss.mean(), epoch)

                for idx, resolution in enumerate(res_k):
                    writer.add_scalar(f'val/loss_1-{resolution}', total_loss[idx], epoch)
                    writer.add_scalar(f'val/loss_similar_1-{resolution}', similar_loss[idx], epoch)
                    writer.add_scalar(f'val/loss_smooth_1-{resolution}', smooth_loss[idx], epoch)

                # === wandb Logging (Validation) ===
                wandb.log({
                    "Val/Total_Loss": total_loss.mean() / len(val_loader),
                    "Val/Similarity_Loss": similar_loss.mean(),
                    "Val/Regularizer_Loss": smooth_loss.mean(),
                    "Epoch": epoch
                })
                for idx, resolution in enumerate(res_k):
                    wandb.log({
                        f"Val/Total_Loss_res{resolution}": total_loss[idx],
                        f"Val/Similarity_Loss_res{resolution}": similar_loss[idx],
                        f"Val/Regularizer_Loss_res{resolution}": smooth_loss[idx],
                        "Epoch": epoch
                    })
                
                print(f"Epoch {epoch+1}/{epochs}, Valid Loss: {total_loss.mean() / len(val_loader)}")
                if best_loss > total_loss.mean() / len(val_loader):
                    cnt = 0
                    best_loss = total_loss.mean() / len(val_loader)
                    torch.save(model.state_dict(), f'./{log_name}.pt')
                else: 
                    cnt+=1
        if epoch%10 == 0:
            model.eval()
            with torch.no_grad():
                # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(val_loader)}")
                for i, (img, template, img_min, img_max, affine) in enumerate(save_loader):
                    if i>1:
                        break
                    img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
        
                    tot_displace_field = None
                    for idx, resolution in enumerate(res_k):
                        # downsampling image
                        target_spacing = (1.0*resolution, 1.0*resolution, 1.0*resolution)
                        cur_img = resample_pytorch_5d(img, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)
                        cur_template = resample_pytorch_5d(template, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)

                        # upsampling deformation field
                        if tot_displace_field is not None:
                            tot_displace_field = upsample_deformation(tot_displace_field, cur_img.shape[2:], tot_displace_field.shape[2:])

                            # apply previous deformation field
                            P_cur_img = apply_deformation_using_disp(cur_img, tot_displace_field)
                        else:
                            P_cur_img = cur_img

                        # get current deformation field
                        stacked_input = torch.stack((P_cur_img.squeeze(1), cur_template.squeeze(1)), dim=1)

                        # get current deformation field
                        displace_field = model(stacked_input)
                        deformed_cur_img = apply_deformation_using_disp(P_cur_img, displace_field)

                        # save deformation field
                        new_displace_field = model(stacked_input).clone().detach()
                        tot_displace_field = compose_displace(tot_displace_field, new_displace_field)

                        save_deformed_image(deformed_cur_img, f'saved_images/{log_name}/epoch{epoch}_img{i}_brain_res{resolution}.nii.gz', img_min, img_max, affine.squeeze())

                        np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_deform_new_res{resolution}.npy',  new_displace_field.cpu().numpy())
                        np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_deform_tot_res{resolution}.npy', tot_displace_field.cpu().numpy())

            if cnt >= 3:
                # early stop
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="brain_core_zero")
    parser.add_argument("--template_path", type=str, default="MNI_skremove_RAS_cropped.nii")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--saved_path", default=None)

    parser.add_argument("--res_level", type=int, default=3)
    parser.add_argument("--loss", type=str, default="NCC")
    parser.add_argument("--reg", type=str, default=None)
    parser.add_argument("--alpha", type=str, default=None)
    parser.add_argument("--alp_sca", type=float, default=1.0)

    args = parser.parse_args()

    # Example usage
    set_seed(seed=0)
    train_model(args.image_path, args.template_path, loss=args.loss, reg=args.reg, \
                alpha=args.alpha, alp_sca=args.alp_sca, epochs=1000, lr=1e-4, batch_size=1, res_level=args.res_level, saved_path=args.saved_path, start_epoch=args.start_epoch)
    
    wandb.finish()