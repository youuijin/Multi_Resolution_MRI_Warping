import torch, wandb, argparse
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.R2Net_model import U_Net_Hyprid

from utils.utils import *
from utils.loss import MSE_loss, Miccai2018Loss
from utils.dataset import set_dataloader

wandb.login(key="87539aeaa75ad2d8a28ec87d70e5d6ce1277c544")

import matplotlib.pyplot as plt
import wandb
import torch

def transform_slice(img):
    # apply 90-degree CCW rotation + horizontal flip
    return np.fliplr(np.rot90(img, k=1))


def save_middle_slices(img_3d, epoch, idx):
    """
    img_3d: [D, H, W] or [1, D, H, W] or [B, 1, D, H, W] (e.g., torch.Tensor)
    Returns: matplotlib Figure with x, y, z middle slices side-by-side
    """
    if isinstance(img_3d, torch.Tensor):
        img_3d = img_3d.squeeze().detach().cpu().numpy()

    D, H, W = img_3d.shape

    slice_x = transform_slice(img_3d[D // 2, :, :])
    slice_y = transform_slice(img_3d[:, H // 2, :])
    slice_z = transform_slice(img_3d[:, :, W // 2])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(slice_z, cmap='gray')
    axes[0].set_title('Axial (X)')
    axes[1].imshow(slice_y, cmap='gray')
    axes[1].set_title('Coronal (Y)')
    axes[2].imshow(slice_x, cmap='gray')
    axes[2].set_title('Sagittal (Z)')

    for ax in axes:
        ax.axis('off')


    plt.tight_layout()
    wandb.log({f"Media/deformed_slices_img{idx}": wandb.Image(fig)}, step=epoch)
    
    plt.close(fig)
    return fig

# Training setup
def train_model(image_paths, template_path, out_ch, out_lay, image_sigma, prior_lambda, loss='ELBO', reg='TV', epochs=200, batch_size=1, lr=1e-4, alpha=0.5, val_interval=5, start_epoch=0, saved_path=None):
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

    log_name = f'{method}_loss{loss}_{image_sigma}({reg}_{prior_lambda})_lr{lr}_bs{batch_size}'
    
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
            "image_sigma": image_sigma,
            "prior_lambda": prior_lambda
        }
    )

    train_loader, val_loader, _ = set_dataloader(image_paths, template_path, batch_size)

    model = U_Net_Hyprid(out_channels=out_ch, out_layers=out_lay)
    
    if start_epoch>0:
        model.load_state_dict(torch.load(saved_path, weights_only=True))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = Miccai2018Loss(reg=reg, image_sigma=image_sigma, prior_lambda=prior_lambda)

    best_loss = 100000
    best_mse_loss= 100000
    cnt = 0

    for epoch in range(epochs):
        if epoch<start_epoch:
            continue
        model.train()
        total_loss = 0
        total_similar, total_smooth = 0, 0
        uncertainties, smooths = [0. for _ in range(out_lay)], [0. for _ in range(out_lay)]
        for (img, template, _, _, _) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
            stacked_input = torch.cat([img, template], dim=1)

            # get deformation field
            tot_mean, tot_std, means, stds = model(stacked_input) # return lists
            
            # sampling in multi-resolution
            sampling_disp = None
            for m, s in zip(means, stds):
                eps_r = torch.randn_like(m)
                disp_r = m + eps_r * s
                if sampling_disp == None:
                    sampling_disp = disp_r
                else:
                    # upsample previous to match current level
                    upsampled = F.interpolate(sampling_disp, size=disp_r.shape[2:], mode='trilinear', align_corners=True)
                    sampling_disp = disp_r + upsampled
            
            # eps = torch.randn_like(tot_mean[-1])
            # sampling_disp = tot_mean[-1] + eps * tot_std[-1] # reference : VoxelMorph-diff, sampling in training phase

            deformed_cur_img = apply_deformation_using_disp(img, sampling_disp)

            loss, similar, smooth, buff = criterion(deformed_cur_img, template, means, stds, return_all=True) #TODO : expand multi-resolution 
    
            # calculate loss and train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_similar += similar
            total_smooth += smooth
            for i in range(out_lay):
                uncertainties[i] += stds[i].mean()
                smooths[i] += buff[i]

        print(f"Epoch {epoch}/{epochs}, Train Loss: {total_loss / len(train_loader)}")

        # === wandb Logging (Train) ===
        wandb.log({
            "Train/Total_Loss": total_loss / len(train_loader),
            "Train/Similar_Loss": total_similar / len(train_loader),
            "Train/Smooth_Loss": total_smooth / len(train_loader)
        }, step=epoch)
        for i in range(out_lay):
            wandb.log({
                f"Train_Uncert/layer_{out_lay-i}": uncertainties[i] / len(train_loader),
                f"Train_Smooth/layer_{out_lay-i}": smooths[i] / len(train_loader)
            }, step=epoch)

        if epoch%val_interval == 0:
            model.eval()
            total_loss = 0
            total_MSE_loss = 0
            total_similar, total_smooth = 0, 0
            uncertainties, smooths = [0. for _ in range(out_lay)], [0. for _ in range(out_lay)]
            with torch.no_grad():
                for idx, (img, template, _, _, _) in enumerate(val_loader):
                    img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda()
                    stacked_input = torch.cat([img, template], dim=1)

                    # get deformation field
                    tot_mean, tot_std, means, stds = model(stacked_input)

                    deformed_cur_img = apply_deformation_using_disp(img, tot_mean[-1]) # using mean as deformation field (no sampling)

                    loss, similar, smooth, buff = criterion(deformed_cur_img, template, means, stds, return_all=True)
                    sim_loss = MSE_loss(deformed_cur_img, template)
            
                    total_loss += loss.item()
                    total_MSE_loss += sim_loss.item()
                    total_similar += similar
                    total_smooth += smooth

                    for i in range(out_lay):
                        uncertainties[i] += stds[i].mean()
                        smooths[i] += buff[i]

                    if idx < 2:
                        save_middle_slices(deformed_cur_img, epoch, idx)

                # === wandb Logging (Validation) ===
                wandb.log({
                    "Val/Total_Loss": total_loss / len(val_loader),
                    "Val/MSE_Loss": total_MSE_loss / len(val_loader),
                    "Val/Similar_Loss": total_similar / len(val_loader),
                    "Val/Smooth_Loss": total_smooth / len(val_loader),
                }, step=epoch)
                for i in range(out_lay):
                    wandb.log({
                        f"Val_Uncert/layer_{out_lay-i}": uncertainties[i] / len(val_loader),
                        f"Val_Smooth/layer_{out_lay-i}": smooths[i] / len(val_loader),
                    }, step=epoch)

                print(f"Epoch {epoch}/{epochs}, Valid Loss: {total_loss / len(val_loader)}")
                if best_loss > total_loss / len(val_loader):
                    cnt = 0
                    best_loss = total_loss / len(val_loader)
                    torch.save(model.state_dict(), f'./{log_name}_total.pt')
                    # wandb.save( f'./{log_name}_total.pt')
        
                if best_mse_loss > total_MSE_loss / len(val_loader):
                    cnt = 0
                    best_loss = total_loss / len(val_loader)
                    torch.save(model.state_dict(), f'./{log_name}_mse.pt')
                    # wandb.save( f'./{log_name}_mse.pt')
                else: 
                    cnt+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="data/brain_core_zero")
    parser.add_argument("--template_path", type=str, default="MNI_skremove_RAS_cropped.nii")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--saved_path", default=None)

    # training options
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--out_layers", type=int, default=1)
    parser.add_argument("--loss", type=str, default="Bayes")
    parser.add_argument("--reg", type=str, default='tv', choices=['tv', 'atv'])
    parser.add_argument("--image_sigma", type=float, default=0.02)
    parser.add_argument("--prior_lambda", type=float, default=10.0)

    # validation options
    parser.add_argument("--val_interval", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    # Example usage
    set_seed(seed=0)
    train_model(args.image_path, args.template_path, loss=args.loss, reg=args.reg, out_ch=6, out_lay=args.out_layers, \
                image_sigma=args.image_sigma, prior_lambda=args.prior_lambda, epochs=200, lr=1e-4, batch_size=args.batch_size, \
                val_interval=args.val_interval, saved_path=args.saved_path, start_epoch=args.start_epoch)
    
    wandb.finish()