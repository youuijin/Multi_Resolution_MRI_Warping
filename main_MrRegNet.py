import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os, wandb
from utils.R2Net_model import U_Net_Hyprid

import argparse

from utils.utils import *
from utils.loss import TrainLoss
from utils.dataset import set_dataloader

wandb.login(key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")

# Training setup
def train_model(image_paths, template_path, out_ch, out_lay, loss='MSE', reg='TV', epochs=200, batch_size=1, lr=1e-4, alpha=0.5, alp_sca=1.0, sca_fn='exp', val_interval=5, val_detail=False, start_epoch=0, saved_path=None):
    if out_ch==3:
        if out_lay==1:
            method = 'VM'
        else:
            method = 'MrRegNet'
    else:
        if out_lay==1:
            method = 'Uncertainty_VM'
        else:
            method = 'Uncertainty_MrRegNet'

    if reg is None:
        log_name = f'{method}_loss{loss}_lr{lr}_bs4'
    else:
        log_name = f'{method}_loss{loss}({reg}_alpha{alpha})_lr{lr}_bs4'
        if alp_sca != 1.0:
            log_name = f'{method}_loss{loss}({reg}_alpha{alpha}_{sca_fn}_sca{alp_sca})_lr{lr}_bs4'
    print('Training:', log_name)
    # os.makedirs(f'saved_images/{log_name}', exist_ok=True)

    #  --- WandB Init ---
    wandb.init(
        project="brain-warping-cluster",
        name=log_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "out_channels":out_ch, 
            "out_layers":out_lay,
            "learning_rate": lr,
            "reg": reg,
            "alpha": alpha,
            "alp_sca": alp_sca,
            "sca_fn":sca_fn
        }
    )

    train_loader, val_loader, _ = set_dataloader(image_paths, template_path, batch_size)

    model = U_Net_Hyprid(out_channels=out_ch, out_layers=out_lay) #TODO :select model
    
    if start_epoch>0:
        model.load_state_dict(torch.load(saved_path, weights_only=True))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TrainLoss(loss, reg, alpha, alp_sca, sca_fn)

    best_loss = 100000
    cnt = 0

    for epoch in range(epochs):
        if epoch<start_epoch:
            continue
        model.train()
        total_loss = np.array([0. for _ in range(out_lay)])
        similar_loss = np.array([0. for _ in range(out_lay)])
        smooth_loss = np.array([0. for _ in range(out_lay)])
        for (img, template, _, _, _) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
            stacked_input = torch.cat([img, template], dim=1)

            # get deformation field
            tot_loss = torch.tensor(0., device='cuda')
            disp, disp_res = model(stacked_input)
            for idx, (cur_dis, cur_dis_res) in enumerate(zip(disp, disp_res)):
                cur_source_img = F.interpolate(img, size=cur_dis.shape[2:], mode='nearest')
                cur_target_img = F.interpolate(template, size=cur_dis.shape[2:], mode='nearest') 

                deformed_cur_img = apply_deformation_using_disp(cur_source_img, cur_dis)
                loss, diff_loss, smoo_loss = criterion(deformed_cur_img, cur_target_img, cur_dis_res, idx)
                
                tot_loss += loss
                total_loss[idx] += loss.item()
                similar_loss[idx] += diff_loss
                smooth_loss[idx] += smoo_loss

            # calculate loss and train model
            optimizer.zero_grad()
            (tot_loss/out_lay).backward()
            optimizer.step()

        print(f"Epoch {epoch}/{epochs}, Train Loss: {total_loss.mean() / len(train_loader)}")

        # === wandb Logging (Train) ===
        wandb.log({
            "Train/Total_Loss": total_loss.mean() / len(train_loader),
            "Train/Similarity_Loss": similar_loss.mean()/len(train_loader),
            "Train/Regularizer_Loss": smooth_loss.mean()/len(train_loader),
            "Epoch": epoch
        })
        if out_lay>1:
            for idx, resolution in enumerate([3, 2, 1]):
                wandb.log({
                    f"Train/Total_Loss_layer_{resolution}": total_loss[idx]/len(train_loader),
                    f"Train/Similarity_Loss_layer_{resolution}": similar_loss[idx]/len(train_loader),
                    f"Train/Regularizer_Loss_layer_{resolution}": smooth_loss[idx]/len(train_loader),
                    "Epoch": epoch
                })

        if epoch%val_interval == 0:
            model.eval()
            total_loss = np.array([0. for _ in range(3)])
            similar_loss = np.array([0. for _ in range(3)])
            smooth_loss = np.array([0. for _ in range(3)])
            with torch.no_grad():
                for (img, template, _, _, _) in val_loader:
                    img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
                    stacked_input = torch.cat([img, template], dim=1)

                    # get deformation field
                    tot_loss = torch.tensor(0., device='cuda')
                    disp, disp_res = model(stacked_input)
                    
                    disp, disp_res = model(stacked_input)
                    for idx, (cur_dis, cur_dis_res) in enumerate(zip(disp, disp_res)):
                        cur_source_img = F.interpolate(img, size=cur_dis.shape[2:], mode='nearest')
                        cur_target_img = F.interpolate(template, size=cur_dis.shape[2:], mode='nearest') 

                        deformed_cur_img = apply_deformation_using_disp(cur_source_img, cur_dis)
                        loss, diff_loss, smoo_loss = criterion(deformed_cur_img, cur_target_img, cur_dis_res, idx)
                        
                        tot_loss += loss
                        total_loss[idx] += loss.item()
                        similar_loss[idx] += diff_loss
                        smooth_loss[idx] += smoo_loss

                # === wandb Logging (Validation) ===
                wandb.log({
                    "Val/Total_Loss": total_loss.mean() / len(val_loader),
                    "Val/Similarity_Loss": similar_loss.mean()/len(val_loader),
                    "Val/Regularizer_Loss": smooth_loss.mean()/len(val_loader),
                    "Epoch": epoch
                })
                if out_lay>1:
                    for idx, resolution in enumerate([3, 2, 1]):
                        wandb.log({
                            f"Val/Total_Loss_layer_{resolution}": total_loss[idx]/len(val_loader),
                            f"Val/Similarity_Loss_layer_{resolution}": similar_loss[idx]/len(val_loader),
                            f"Val/Regularizer_Loss_layer_{resolution}": smooth_loss[idx]/len(val_loader),
                            "Epoch": epoch
                        })
                
                print(f"Epoch {epoch}/{epochs}, Valid Loss: {total_loss.mean() / len(val_loader)}")
                if best_loss > total_loss.mean() / len(val_loader):
                    cnt = 0
                    best_loss = total_loss.mean() / len(val_loader)
                    torch.save(model.state_dict(), f'./{log_name}.pt')
                    wandb.save(f'./{log_name}.pt')
                else: 
                    cnt+=1

                if cnt >= 3:
                    # early stop
                    return
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="data/brain_core_zero")
    parser.add_argument("--template_path", type=str, default="MNI_skremove_RAS_cropped.nii")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--saved_path", default=None)

    # training options
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--out_layers", type=int, default=1)
    parser.add_argument("--loss", type=str, default="NCC")
    parser.add_argument("--reg", type=str, default='tv')
    parser.add_argument("--alpha", type=str, default="16.0")
    parser.add_argument("--alp_sca", type=float, default=0.5)
    parser.add_argument("--sca_fn", type=str, default='exp', choices=['exp', 'linear'])

    # validation options
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--val_detail", default=False, action='store_true')

    args = parser.parse_args()

    # Example usage
    set_seed(seed=0)
    train_model(args.image_path, args.template_path, loss=args.loss, reg=args.reg, out_ch=3, out_lay=args.out_layers,\
                alpha=args.alpha, alp_sca=args.alp_sca, sca_fn=args.sca_fn, epochs=200, lr=1e-4, batch_size=1, \
                val_interval=args.val_interval, val_detail=args.val_detail, saved_path=args.saved_path, start_epoch=args.start_epoch)
    
    wandb.finish()