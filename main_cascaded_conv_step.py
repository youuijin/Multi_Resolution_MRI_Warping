import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os, wandb, argparse
from utils.model import R2Net_UNet, FusionWeightNet

from utils.utils import *
from utils.loss import TrainLoss, FusionNetLoss
from utils.dataset import set_dataloader

from evaluation.evaluate_dice_FS_FusionNet import test

wandb.login(key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")

def apply_weight_sum(weights, dis_fields):
    # 각 weight slice
    w1 = weights[:, 0:1, ...]  # (B, 1, H, W, D)
    w2 = weights[:, 1:2, ...]
    w3 = weights[:, 2:3, ...]

    phi1, phi2, phi3 = dis_fields
    phi1 = F.interpolate(phi1, size=phi3.shape[2:], mode='trilinear', align_corners=False)
    phi2 = F.interpolate(phi2, size=phi3.shape[2:], mode='trilinear', align_corners=False)

    # φ weighted sum
    corr_displace_field = w1 * phi1 + w2 * phi2 + w3 * phi3  # (B, 3, H, W, D)

    return corr_displace_field

# Training setup
def train_model(image_paths, template_path, loss='MSE', reg='TV', epochs=200, batch_size=1, lr=1e-4, alpha=0.5, res_level=3, val_interval=5, val_detail=False, start_epoch=0, saved_path=None):
    if saved_path:
        prev_loss = saved_path.split('_')[2][4:]
        epoch_phase1 = int(saved_path.split('_')[4][5:-3])
        log_name = f'Fusion_after{prev_loss}{epoch_phase1}_res{res_level}_loss{loss}({reg}_alpha{alpha})_lr{lr}'
    else:
        epoch_phase1 = epochs
        log_name = f'train_res{res_level}_loss{loss}_lr{lr}_epoch{epochs}.pt'
        
    print('Training:', log_name)
    os.makedirs(f'saved_images/{log_name}', exist_ok=True)

    #  --- WandB Init ---
    wandb.init(
        project="brain-warping-new",
        name=log_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "loss": loss,
            "reg": reg,
            "alpha": alpha,
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
    fus_model = FusionWeightNet().cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_fus = optim.Adam(fus_model.parameters(), lr=lr)

    criterion = TrainLoss(loss)
    criterion_fus = FusionNetLoss(loss, reg=reg, alpha=alpha)

    best_loss = 100000
    cnt = 0

    for epoch in range(epochs):
        if epoch<start_epoch:
            continue

        if epoch<epoch_phase1:
            # train U-Net
            model.train()
            total_loss = torch.tensor([0. for _ in range(len(res_k))], device='cuda')
            similar_loss = np.array([0. for _ in range(len(res_k))])
            smooth_loss = np.array([0. for _ in range(len(res_k))])
        else:
            # train Fusion Net
            model.eval() # Freeze U-Net
            fus_model.train()
            fus_tot_loss, fus_sim_loss, fus_smooth_loss = 0., 0., 0.

        for (img, template, _, _, _) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()

            dis_fields = []
            x2_features = []
            cur_loss = torch.tensor([0.], device='cuda')
            for idx, resolution in enumerate(res_k):
                # downsampling image
                target_spacing = (1.0*resolution, 1.0*resolution, 1.0*resolution)
                cur_img = resample_pytorch_5d(img, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)
                cur_template = resample_pytorch_5d(template, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)

                stacked_input = torch.stack((cur_img.squeeze(1), cur_template.squeeze(1)), dim=1)

                # get current deformation field
                if epoch<epoch_phase1:
                    displace_field = model(stacked_input)
                else:
                    displace_field, feature = model(stacked_input, return_feature=True)
                    x2_features += [feature.detach()]
                dis_fields += [displace_field.detach()]
                deformed_cur_img = apply_deformation_using_disp(cur_img, displace_field)

                # calculate loss and train model
                loss, diff_loss, smoo_loss = criterion(deformed_cur_img, cur_template, displace_field, idx)
                cur_loss += loss
                if epoch<epoch_phase1:
                    total_loss[idx] += loss.item()
                    similar_loss[idx] += diff_loss
                    smooth_loss[idx] += smoo_loss

            if epoch<epoch_phase1:
                # train U-Net
                optimizer.zero_grad()
                (cur_loss/3.).backward()
                optimizer.step()
            else:
                # train Fusion Net
                weights = fus_model(img, dis_fields, x2_features)
                corr_displace_field = apply_weight_sum(weights, dis_fields)

                # 학습용 forward
                warped_img_final = apply_deformation_using_disp(img, corr_displace_field)

                loss_fus, diff_loss_fus, smoo_loss_fus = criterion_fus(warped_img_final, template, corr_displace_field) 

                fus_tot_loss += loss_fus.item()
                fus_sim_loss += diff_loss_fus
                fus_smooth_loss += smoo_loss_fus

                optimizer_fus.zero_grad()
                loss_fus.backward()
                optimizer_fus.step()

        if epoch<epoch_phase1:
            # Train U-Net
            print(f"Epoch {epoch}/{epochs}, Train Loss: {total_loss.mean().item() / len(train_loader)}")

            # === wandb Logging (Train) ===
            wandb.log({
                "Train/Total_Loss": total_loss.mean() / len(train_loader),
                "Train/Similarity_Loss": similar_loss.mean()/len(train_loader),
                "Train/Regularizer_Loss": smooth_loss.mean()/len(train_loader),
                "Epoch": epoch
            })
            for idx, resolution in enumerate(res_k):
                wandb.log({
                    f"Train/Total_Loss_res{resolution}": total_loss[idx]/len(train_loader),
                    f"Train/Similarity_Loss_res{resolution}": similar_loss[idx]/len(train_loader),
                    f"Train/Regularizer_Loss_res{resolution}": smooth_loss[idx]/len(train_loader),
                    "Epoch": epoch
                })
        else:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {fus_tot_loss.item() / len(train_loader)}")
            wandb.log({
                "Train/Fus_Loss": fus_tot_loss/len(train_loader),
                "Train/Fus_Similarity_Loss": fus_sim_loss/len(train_loader),
                "Train/Fus_Regularizer_Loss": fus_smooth_loss/len(train_loader),
                "Epoch": epoch
            })

        if epoch%val_interval == 0:
            model.eval()
            fus_model.eval()
            similar_loss = np.array([0. for _ in range(len(res_k))])
            smooth_loss = np.array([0. for _ in range(len(res_k))])
            neg_rates = np.array([0. for _ in range(len(res_k)+1)])
            total_loss = torch.tensor([0. for _ in range(len(res_k))], device='cuda')
            fus_tot_loss, fus_sim_loss, fus_smooth_loss = 0., 0., 0.

            weights_mean = np.array([0. for _ in range(len(res_k))])
            with torch.no_grad():
                for (img, template, _, _, _) in val_loader:
                    img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
                    dis_fields = []
                    x2_features = []
                    for idx, resolution in enumerate(res_k):
                        # downsampling image
                        target_spacing = (1.0*resolution, 1.0*resolution, 1.0*resolution)
                        cur_img = resample_pytorch_5d(img, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)
                        cur_template = resample_pytorch_5d(template, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)

                        stacked_input = torch.stack((cur_img.squeeze(1), cur_template.squeeze(1)), dim=1)

                        # get current deformation field
                        displace_field, feature = model(stacked_input, return_feature=True)
                        x2_features += [feature.detach()]
                        dis_fields += [displace_field.detach()]
                        deformed_cur_img = apply_deformation_using_disp(cur_img, displace_field)

                        if val_detail:
                            # logging Jacobian determinant
                            neg_rate = calculate_negative_rate(displace_field)
                            neg_rates[idx] += neg_rate

                        # calculate loss and train model
                        loss, diff_loss, smoo_loss = criterion(deformed_cur_img, cur_template, displace_field, idx)
                        total_loss[idx] += loss.item()
                        similar_loss[idx] += diff_loss
                        smooth_loss[idx] += smoo_loss
                    
                    # train Fusion Net
                    weights = fus_model(img, dis_fields, x2_features)
                    corr_displace_field = apply_weight_sum(weights, dis_fields)
                    
                    if val_detail:
                        # logging Jacobian determinant
                        neg_rate = calculate_negative_rate(corr_displace_field)
                        neg_rates[-1] += neg_rate

                    # 학습용 forward
                    warped_img_final = apply_deformation_using_disp(img, corr_displace_field)

                    loss_fus, diff_loss_fus, smoo_loss_fus = criterion_fus(warped_img_final, template, corr_displace_field)  # 마지막 level로 설정

                    fus_tot_loss += loss_fus.item()
                    fus_sim_loss += diff_loss_fus
                    fus_smooth_loss += smoo_loss_fus

                    for i in range(len(res_k)):
                        weights_mean[i] += [weights[:, i:i+1, ...].mean()]

                if val_detail:
                    avg_dice, _ = test(model, fus_model, res_k=res_k)
                    wandb.log({
                        "Val/avg_dice": avg_dice,
                        "Val/neg_rate": neg_rates[:-1].mean()/len(val_loader),
                        "Val/neg_rate_fus":neg_rates[-1]/len(val_loader), #로그추가
                        "Epoch": epoch
                    })

                # === wandb Logging (Validation) ===
                if epoch < epoch_phase1:
                    print(f"Epoch {epoch}/{epochs}, Valid Loss: {total_loss.mean() / len(val_loader)}")
                    
                    wandb.log({
                        "Val/Total_Loss": total_loss.mean() / len(val_loader),
                        "Val/Similarity_Loss": similar_loss.mean()/len(val_loader),
                        "Val/Regularizer_Loss": smooth_loss.mean()/len(val_loader),
                        "Epoch": epoch
                    })
                    for idx, resolution in enumerate(res_k):
                        wandb.log({
                            f"Val/Total_Loss_res{resolution}": total_loss[idx]/len(val_loader),
                            f"Val/Similarity_Loss_res{resolution}": similar_loss[idx]/len(val_loader),
                            f"Val/Regularizer_Loss_res{resolution}": smooth_loss[idx]/len(val_loader),
                            "Epoch": epoch
                        })
                    
                    if best_loss > total_loss.mean() / len(val_loader):
                        cnt = 0
                        best_loss = total_loss.mean() / len(val_loader)
                        torch.save(model.state_dict(), f'./{log_name}.pt')
                    else: 
                        cnt+=1
                else:
                    print(f"Epoch {epoch}/{epochs}, Valid Loss: {fus_tot_loss / len(val_loader)}")

                    wandb.log({
                        "Val/Fus_Loss": fus_tot_loss/len(train_loader),
                        "Val/Fus_Similarity_Loss": fus_sim_loss/len(train_loader),
                        "Val/Fus_Regularizer_Loss": fus_smooth_loss/len(train_loader),
                        "Epoch": epoch
                    })
                    for idx, resolution in enumerate(res_k):
                        wandb.log({
                            f"Val/Weight_res{resolution}": weights_mean[idx]/len(val_loader),
                            "Epoch": epoch
                        })
                    
                    if best_loss > fus_tot_loss / len(val_loader):
                        cnt = 0
                        best_loss = fus_tot_loss / len(val_loader)
                        torch.save(fus_model.state_dict(), f'./{log_name}.pt')
                    else: 
                        cnt+=1

                if cnt >= 3:
                    # early stop
                    return
                
        # save file
        if epoch%15 == 0:
            model.eval()
            with torch.no_grad():
                for i, (img, template, img_min, img_max, affine) in enumerate(save_loader):
                    if i>1:
                        break
                    img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()

                    dis_fields = []
                    x2_features = []
                    for idx, resolution in enumerate(res_k):
                        # downsampling image
                        target_spacing = (1.0*resolution, 1.0*resolution, 1.0*resolution)
                        cur_img = resample_pytorch_5d(img, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)
                        cur_template = resample_pytorch_5d(template, current_spacing=(1.0, 1.0, 1.0), target_spacing=target_spacing)

                        stacked_input = torch.stack((cur_img.squeeze(1), cur_template.squeeze(1)), dim=1)

                        # get current deformation field
                        displace_field, feature = model(stacked_input, return_feature=True)
                        dis_fields += [displace_field.detach()]
                        x2_features += [feature.detach()]

                        if epoch<epoch_phase1:
                            deformed_cur_img = apply_deformation_using_disp(cur_img, displace_field)
                            save_deformed_image(deformed_cur_img, f'saved_images/{log_name}/epoch{epoch}_img{i}_brain_res{resolution}.nii.gz', img_min, img_max, affine.squeeze())
                            np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_deform_res{resolution}.npy',  displace_field.cpu().numpy())

                    if epoch<epoch_phase1:
                        continue

                    # train Fusion Net
                    weights = fus_model(img, dis_fields, x2_features)
                    corr_displace_field = apply_weight_sum(weights, dis_fields)

                    # 학습용 forward
                    warped_img_final = apply_deformation_using_disp(img, corr_displace_field)

                    w1 = weights[:, 0:1, ...]  # (B, 1, H, W, D)
                    w2 = weights[:, 1:2, ...]
                    w3 = weights[:, 2:3, ...]

                    save_deformed_image(warped_img_final, f'saved_images/{log_name}/epoch{epoch}_img{i}_brain_fusion.nii.gz', img_min, img_max, affine.squeeze())

                    np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_deform_fusion.npy',  corr_displace_field.cpu().numpy())

                    np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_weight_4.npy',  w1.cpu().numpy())
                    np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_weight_2.npy',  w2.cpu().numpy())
                    np.save(f'saved_images/{log_name}/epoch{epoch}_img{i}_weight_1.npy',  w3.cpu().numpy())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="brain_core_zero_np")
    parser.add_argument("--template_path", type=str, default="MNI_template/MNI_skremove_RAS_cropped.nii")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--saved_path", default=None)

    # training options
    parser.add_argument("--res_level", type=int, default=3)
    parser.add_argument("--loss", type=str, default="NCC")
    parser.add_argument("--reg", type=str, default=None)
    parser.add_argument("--alpha", type=str, default=None)

    # validation options
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--val_detail", default=False, action='store_true')

    args = parser.parse_args()

    # Example usage
    set_seed(seed=0)
    train_model(args.image_path, args.template_path, loss=args.loss, reg=args.reg, \
                alpha=args.alpha, epochs=200, lr=1e-4, batch_size=1, res_level=args.res_level, \
                val_interval=args.val_interval, val_detail=args.val_detail, saved_path=args.saved_path, start_epoch=args.start_epoch)
    
    wandb.finish()